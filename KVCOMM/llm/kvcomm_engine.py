"""Utilities to manipulate DynamicCache and coordinate KV anchor workflows.

This module extends `transformers.cache_utils.DynamicCache` with operations for
safe slicing, concatenation, selection, splitting by placeholder spans, and
device movement. It also defines `KVCOMMEngine`, which manages per-request
state and anchor selection/updates used by LLMChat for KV reuse and dense
prefill.
"""
from __future__ import annotations

import copy
import threading
from collections.abc import MutableMapping
from collections.abc import Sequence
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import DynamicCache

from KVCOMM.llm.token_ops import concat
from KVCOMM.utils.log import logger

_MISSING = object()
_DELETED = object()

def _is_layered_cache(cache: DynamicCache) -> bool:
    """Return True if the cache uses the newer `layers` structure."""
    return hasattr(cache, "layers")


def _get_layer_count(cache: DynamicCache) -> int:
    """Return number of transformer layers tracked in the cache."""
    if _is_layered_cache(cache):
        return len(cache.layers)
    return len(getattr(cache, "key_cache", []))


def _get_layer_kv(cache: DynamicCache, idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return key/value tensors (or None) for a given layer index."""
    if _is_layered_cache(cache):
        layer = cache.layers[idx]
        return getattr(layer, "keys", None), getattr(layer, "values", None)
    key_cache = getattr(cache, "key_cache", [])
    value_cache = getattr(cache, "value_cache", [])
    key = key_cache[idx] if idx < len(key_cache) else None
    value = value_cache[idx] if idx < len(value_cache) else None
    return key, value


def _set_layer_kv(
    cache: DynamicCache,
    idx: int,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
) -> None:
    """Assign key/value tensors to a specific layer, updating metadata if present."""
    if _is_layered_cache(cache):
        layer = cache.layers[idx]
        layer.keys = key
        layer.values = value
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = bool(isinstance(key, torch.Tensor) and key.numel() > 0)
        if hasattr(layer, "dtype") and isinstance(key, torch.Tensor):
            layer.dtype = key.dtype
        if hasattr(layer, "device") and isinstance(key, torch.Tensor):
            layer.device = key.device
        if hasattr(layer, "cumulative_length") and isinstance(key, torch.Tensor):
            layer.cumulative_length = key.shape[-2]
    else:
        key_cache = getattr(cache, "key_cache")
        value_cache = getattr(cache, "value_cache")
        key_cache[idx] = key
        value_cache[idx] = value


def _stack_cache_tensors(cache: DynamicCache) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return stacked key/value tensors when all layers are dense tensors."""
    layer_count = _get_layer_count(cache)
    if layer_count == 0:
        return None
    keys: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    for idx in range(layer_count):
        key, value = _get_layer_kv(cache, idx)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            return None
        keys.append(key)
        values.append(value)
    if not keys:
        return None
    try:
        key_stack = torch.stack(keys)
        value_stack = torch.stack(values)
    except RuntimeError:
        return None
    return key_stack, value_stack


def _assign_stack_to_cache(cache: DynamicCache, key_stack: torch.Tensor, value_stack: torch.Tensor) -> None:
    """Overwrite cache layers with stacked tensors maintaining per-layer metadata."""
    layer_count = _get_layer_count(cache)
    if _is_layered_cache(cache):
        if layer_count != key_stack.shape[0]:
            raise ValueError("Layer count mismatch while assigning stacked cache tensors.")
        for idx in range(layer_count):
            layer = cache.layers[idx]
            layer.keys = key_stack[idx]
            layer.values = value_stack[idx]
            if hasattr(layer, "is_initialized"):
                layer.is_initialized = key_stack[idx].shape[-2] > 0
            if hasattr(layer, "dtype"):
                layer.dtype = key_stack[idx].dtype
            if hasattr(layer, "device"):
                layer.device = key_stack[idx].device
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = key_stack[idx].shape[-2]
    else:
        cache.key_cache = list(key_stack)
        cache.value_cache = list(value_stack)


def _layer_is_empty(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None:
        return True
    if isinstance(tensor, list):
        return len(tensor) == 0
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() == 0 or tensor.shape[-2] == 0
    return False


def _layer_length(tensor: Optional[torch.Tensor]) -> int:
    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
        return int(tensor.shape[-2])
    return 0


def _normalize_indices(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """Convert None/negative indices into bounded absolute [start, end)."""
    seq_len = _safe_seq_len(cache)
    if start is None:
        start = 0
    elif start < 0:
        start = seq_len + start
    if end is None:
        end = seq_len
    elif end < 0:
        end = seq_len + end
    start = max(0, min(seq_len, start))
    end = max(0, min(seq_len, end))
    return start, end


def _safe_seq_len(cache: DynamicCache) -> int:
    """Best-effort sequence length from cache (APIs vary across transformers)."""
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            length = getter()
        except TypeError:
            length = getter(0)
        if length is not None:
            return int(length)

    for idx in range(_get_layer_count(cache)):
        key, _ = _get_layer_kv(cache, idx)
        if not _layer_is_empty(key):
            return _layer_length(key)
    return 0


def _clone_tensor_or_empty(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or isinstance(tensor, list):
        return tensor
    return tensor.clone()


def _empty_like_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or isinstance(tensor, list):
        return tensor
    return tensor[..., :0, :].clone()


def _ensure_same_layout(cache: DynamicCache, other: DynamicCache) -> None:
    if _get_layer_count(cache) != _get_layer_count(other):
        raise ValueError("Layer count mismatch between DynamicCache objects.")


def _set_seen_tokens(cache: DynamicCache, length: int) -> None:
    try:
        setattr(cache, "_seen_tokens", int(length))
    except Exception:
        pass


def _copy_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = type(cache)()
    if _is_layered_cache(cache):
        new_cache.layers = []
        for idx in range(len(cache.layers)):
            original_layer = cache.layers[idx]
            cloned_layer = copy.deepcopy(original_layer)
            if hasattr(cloned_layer, "keys") and isinstance(cloned_layer.keys, torch.Tensor):
                cloned_layer.keys = cloned_layer.keys.clone()
            if hasattr(cloned_layer, "values") and isinstance(cloned_layer.values, torch.Tensor):
                cloned_layer.values = cloned_layer.values.clone()
            new_cache.layers.append(cloned_layer)
    else:
        new_cache.key_cache = []
        new_cache.value_cache = []
        for idx in range(len(cache.key_cache)):
            key, value = cache.key_cache[idx], cache.value_cache[idx]
            new_cache.key_cache.append(_clone_tensor_or_empty(key))
            new_cache.value_cache.append(_clone_tensor_or_empty(value))
    for attr in ("offloading", "only_non_sliding", "prefetch_stream", "layer_class_to_replicate"):
        if hasattr(cache, attr):
            setattr(new_cache, attr, getattr(cache, attr))
    if hasattr(cache, "_seen_tokens"):
        _set_seen_tokens(new_cache, getattr(cache, "_seen_tokens"))
    return new_cache


def _slice_inplace(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> DynamicCache:
    """In-place slice of KV cache along sequence dimension."""
    start, end = _normalize_indices(cache, start, end)
    if start >= end:
        for idx in range(_get_layer_count(cache)):
            key, value = _get_layer_kv(cache, idx)
            _set_layer_kv(cache, idx, _empty_like_tensor(key), _empty_like_tensor(value))
        _set_seen_tokens(cache, 0)
        return cache

    stacked = _stack_cache_tensors(cache)
    if stacked is not None:
        key_stack, value_stack = stacked
        slice_start = min(start, key_stack.shape[-2])
        slice_end = min(end, key_stack.shape[-2])
        if slice_end <= slice_start:
            new_key_stack = key_stack[..., :0, :].clone()
            new_value_stack = value_stack[..., :0, :].clone()
        else:
            new_key_stack = key_stack[..., slice_start:slice_end, :].clone()
            new_value_stack = value_stack[..., slice_start:slice_end, :].clone()
        _assign_stack_to_cache(cache, new_key_stack, new_value_stack)
    else:
        for idx in range(_get_layer_count(cache)):
            key, value = _get_layer_kv(cache, idx)
            if _layer_is_empty(key):
                continue
            current_len = _layer_length(key)
            slice_start = min(start, current_len)
            slice_end = min(end, current_len)
            if slice_end <= slice_start:
                new_key = _empty_like_tensor(key)
                new_value = _empty_like_tensor(value)
            else:
                new_key = key[..., slice_start:slice_end, :].clone()
                new_value = value[..., slice_start:slice_end, :].clone()
            _set_layer_kv(cache, idx, new_key, new_value)

    _set_seen_tokens(cache, end - start)
    return cache


def _slice_functional(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> DynamicCache:
    """Return a sliced copy of the KV cache."""
    copied = _copy_cache(cache)
    return _slice_inplace(copied, start, end)


def _concat_tensors(
    base: Optional[torch.Tensor],
    additions: Sequence[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    if isinstance(base, torch.Tensor) and base.shape[-2] > 0:
        tensors.append(base)
    for tensor in additions:
        if isinstance(tensor, torch.Tensor) and tensor.shape[-2] > 0:
            tensors.append(tensor)
    if not tensors:

        for candidate in [base, *additions]:
            if isinstance(candidate, torch.Tensor):
                return candidate[..., :0, :].clone()
        return base
    first_tensor = tensors[0]
    other_tensors = [t if t is first_tensor else t.to(first_tensor.device) for t in tensors[1:]]
    return torch.cat([first_tensor] + other_tensors, dim=-2)


def _ensure_cache_sequence(
    caches: Union[DynamicCache, Sequence[DynamicCache], None]
) -> List[DynamicCache]:
    if caches is None:
        return []
    if isinstance(caches, (list, tuple)):
        return [cache for cache in caches if cache is not None]
    return [caches]


def _concat_inplace(cache: DynamicCache, others: Sequence[DynamicCache]) -> DynamicCache:
    """In-place concatenate multiple caches along sequence dimension."""
    if not others:
        return cache

    usable = [other for other in others if other is not None]
    if not usable:
        return cache

    for other in usable:
        _ensure_same_layout(cache, other)

    base_stack = _stack_cache_tensors(cache)
    other_stacks = [_stack_cache_tensors(other) for other in usable]

    if base_stack is not None and all(stack is not None for stack in other_stacks):
        base_keys, base_values = base_stack
        other_keys = [stack[0] for stack in other_stacks]                          
        other_values = [stack[1] for stack in other_stacks]                          
        key_stack = torch.cat([base_keys] + other_keys, dim=-2)
        value_stack = torch.cat([base_values] + other_values, dim=-2)
        _assign_stack_to_cache(cache, key_stack, value_stack)
        _set_seen_tokens(cache, key_stack.shape[-2])
        return cache

    for idx in range(_get_layer_count(cache)):
        base_key, base_value = _get_layer_kv(cache, idx)
        other_keys = []
        other_values = []
        for other in usable:
            key, value = _get_layer_kv(other, idx)
            other_keys.append(key)
            other_values.append(value)

        new_key = _concat_tensors(base_key, other_keys)
        new_value = _concat_tensors(base_value, other_values)
        _set_layer_kv(cache, idx, new_key, new_value)

    new_length = _safe_seq_len(cache)
    _set_seen_tokens(cache, new_length)
    return cache


def _concat_functional(cache: DynamicCache, others: Sequence[DynamicCache]) -> DynamicCache:
    """Return a new cache that is the concatenation of base and others."""
    copied = _copy_cache(cache)
    return _concat_inplace(copied, others)


def _replace_inplace(cache: DynamicCache, start: int, end: int, real: DynamicCache) -> DynamicCache:
    """In-place replace [start, end) with the content from `real`."""
    left = cache.slice(start=0, end=start)
    middle = real.copy()
    right = cache.slice(start=end, end=None)
    replaced = left.concat([middle, right])
    if _is_layered_cache(cache):
        cache.layers = replaced.layers
    else:
        cache.key_cache = replaced.key_cache
        cache.value_cache = replaced.value_cache
    _set_seen_tokens(cache, _safe_seq_len(cache))
    return cache


def _replace_functional(cache: DynamicCache, start: int, end: int, real: DynamicCache) -> DynamicCache:
    """Return a copy of cache with [start, end) replaced by `real`."""
    copied = _copy_cache(cache)
    return _replace_inplace(copied, start, end, real)


def _select_indices(cache: DynamicCache, indices: torch.Tensor) -> DynamicCache:
    """Select positions by index tensor, preserving layout and metadata."""
    stacked = _stack_cache_tensors(cache)
    if stacked is not None:
        key_stack, value_stack = stacked
        selected_keys = key_stack[..., indices, :].clone()
        selected_values = value_stack[..., indices, :].clone()
        _assign_stack_to_cache(cache, selected_keys, selected_values)
        _set_seen_tokens(cache, indices.shape[-1])
        return cache

    for idx in range(_get_layer_count(cache)):
        key, value = _get_layer_kv(cache, idx)
        if _layer_is_empty(key):
            continue
        _set_layer_kv(cache, idx, key[..., indices, :].clone(), value[..., indices, :].clone())
    _set_seen_tokens(cache, indices.shape[-1])
    return cache


def _to_device(cache: DynamicCache, device: Union[str, torch.device]) -> DynamicCache:
    """Move all tensors in the cache to the specified device."""
    if _is_layered_cache(cache):
        for layer in cache.layers:
            if isinstance(getattr(layer, "keys", None), torch.Tensor):
                layer.keys = layer.keys.to(device)
            if isinstance(getattr(layer, "values", None), torch.Tensor):
                layer.values = layer.values.to(device)
    else:
        for idx in range(len(cache.key_cache)):
            if isinstance(cache.key_cache[idx], torch.Tensor):
                cache.key_cache[idx] = cache.key_cache[idx].to(device)
            if isinstance(cache.value_cache[idx], torch.Tensor):
                cache.value_cache[idx] = cache.value_cache[idx].to(device)
    return cache


def _split_cache_by_placeholders(
    cache: DynamicCache,
    placeholder_dict: Dict[str, Tuple[int, int]],
) -> Tuple[List[DynamicCache], List[DynamicCache]]:
    """Split a cache into placeholder and prefix segments per provided spans."""
    if not placeholder_dict:
        return [], [cache.copy()]

    total_len = _safe_seq_len(cache)
    intervals: List[Tuple[int, int, bool]] = []
    last = 0
    for start, end in sorted(placeholder_dict.values(), key=lambda pair: pair[0]):
        if start > last:
            intervals.append((last, start, False))
        intervals.append((start, end, True))
        last = end
    if last < total_len:
        intervals.append((last, total_len, False))

    placeholder_caches: List[DynamicCache] = []
    prefix_caches: List[DynamicCache] = []
    for start, end, is_placeholder in intervals:
        segment = cache.slice(start=start, end=end)
        segment_length = max(end - start, 0)
        _set_seen_tokens(segment, segment_length)
        if is_placeholder:
            placeholder_caches.append(segment)
        else:
            prefix_caches.append(segment)
    return placeholder_caches, prefix_caches


def _elementwise_binary_op(
    cache: DynamicCache,
    other: DynamicCache,
    op,
) -> DynamicCache:
    """Apply an elementwise binary op to two caches layer-by-layer."""
    _ensure_same_layout(cache, other)
    base_stack = _stack_cache_tensors(cache)
    other_stack = _stack_cache_tensors(other)
    if base_stack is not None and other_stack is not None:
        result = _copy_cache(cache)
        key_stack = op(base_stack[0], other_stack[0])
        value_stack = op(base_stack[1], other_stack[1])
        _assign_stack_to_cache(result, key_stack, value_stack)
        _set_seen_tokens(result, key_stack.shape[-2])
        return result

    result = type(cache)()
    if _is_layered_cache(cache):
        result.layers = []
    else:
        result.key_cache = []
        result.value_cache = []
    for idx in range(_get_layer_count(cache)):
        key_a, value_a = _get_layer_kv(cache, idx)
        key_b, value_b = _get_layer_kv(other, idx)
        if _layer_is_empty(key_a):
            new_key = _clone_tensor_or_empty(key_b)
            new_value = _clone_tensor_or_empty(value_b)
        elif _layer_is_empty(key_b):
            new_key = _clone_tensor_or_empty(key_a)
            new_value = _clone_tensor_or_empty(value_a)
        else:
            new_key = op(key_a, key_b)
            new_value = op(value_a, value_b)
        if _is_layered_cache(cache):
            layer = copy.deepcopy(cache.layers[idx])
            layer.keys = new_key
            layer.values = new_value
            if hasattr(layer, "is_initialized"):
                layer.is_initialized = not _layer_is_empty(new_key)
            if hasattr(layer, "dtype") and isinstance(new_key, torch.Tensor):
                layer.dtype = new_key.dtype
            if hasattr(layer, "device") and isinstance(new_key, torch.Tensor):
                layer.device = new_key.device
            result.layers.append(layer)
        else:
            result.key_cache.append(new_key)
            result.value_cache.append(new_value)
    _set_seen_tokens(result, _safe_seq_len(cache))
    return result


def _split_cache(cache: DynamicCache, sizes: Sequence[int]) -> List[DynamicCache]:
    """Split a cache into multiple segments by the given lengths (sum sizes)."""
    offsets = []
    start = 0
    for size in sizes:
        offsets.append((start, start + size))
        start += size
    return [cache.slice(start=s, end=e) for s, e in offsets]


def _install_dynamic_cache_extensions() -> None:
    """Monkey-patch DynamicCache with convenience methods used by KVCOMM."""
    if getattr(DynamicCache, "_kvcomm_extensions_installed", False):
        return

    DynamicCache._normalize_slice_indices = lambda self, start=None, end=None: _normalize_indices(self, start, end)
    DynamicCache.slice_ = lambda self, start=None, end=None: _slice_inplace(self, start, end)
    DynamicCache.slice = lambda self, start=None, end=None: _slice_functional(self, start, end)
    DynamicCache.concat_ = lambda self, other: _concat_inplace(self, _ensure_cache_sequence(other))
    DynamicCache.concat = lambda self, other: _concat_functional(self, _ensure_cache_sequence(other))
    DynamicCache.replace_ = lambda self, start, end, real: _replace_inplace(self, start, end, real)
    DynamicCache.replace = lambda self, start, end, real: _replace_functional(self, start, end, real)
    DynamicCache.select_indices = lambda self, indices: _select_indices(self, indices)
    DynamicCache.to = lambda self, device: _to_device(self, device)
    DynamicCache.copy = lambda self: _copy_cache(self)
    DynamicCache.split_cache_by_placeholders = lambda self, placeholder_dict: _split_cache_by_placeholders(
        self, placeholder_dict
    )
    DynamicCache.__add__ = lambda self, other: _elementwise_binary_op(self, other, torch.add)
    DynamicCache.__sub__ = lambda self, other: _elementwise_binary_op(self, other, torch.sub)
    DynamicCache.split = lambda self, sizes: _split_cache(self, sizes)
    DynamicCache._kvcomm_extensions_installed = True


_install_dynamic_cache_extensions()


def _clone_default(value: Any) -> Any:
    if isinstance(value, (dict, list, set, tuple)):
        return copy.deepcopy(value)
    return copy.copy(value) if hasattr(value, "__copy__") else value


class _ScopedDict(MutableMapping):
    """Request-scoped view over a shared dictionary with deferred commits."""

    def __init__(self, base: Dict[str, Any]):
        self._base = base
        self._local: Dict[str, Any] = {}

    def _ensure_local(self, key: str) -> None:
        if key in self._local:
            return
        if key in self._base:
            self._local[key] = copy.deepcopy(self._base[key])

    def __getitem__(self, key: str) -> Any:
        if key in self._local:
            value = self._local[key]
            if value is _DELETED:
                raise KeyError(key)
            return value
        if key in self._base:
            value = copy.deepcopy(self._base[key])
            self._local[key] = value
            if value is _DELETED:
                raise KeyError(key)
            return value
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._local[key] = value

    def __delitem__(self, key: str) -> None:
        self.pop(key)

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> List[str]:
        merged = set(self._base.keys()) | set(self._local.keys())
        return [
            key
            for key in merged
            if self._local.get(key, None) is not _DELETED
        ]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def values(self):
        for _, value in self.items():
            yield value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: Any = None):
        if key in self._local:
            value = self._local[key]
            if value is _DELETED:
                new_value = _clone_default(default)
                self._local[key] = new_value
                return new_value
            return value
        if key in self._base:
            value = copy.deepcopy(self._base[key])
            self._local[key] = value
            return value
        new_value = _clone_default(default)
        self._local[key] = new_value
        return new_value

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        self._ensure_local(key)
        if key not in self._local:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self._local[key]
        if value is _DELETED:
            if default is _MISSING:
                raise KeyError(key)
            return default
        self._local[key] = _DELETED
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._local:
            return self._local[key] is not _DELETED
        return key in self._base

    def commit(self) -> None:
        for key, value in self._local.items():
            if value is _DELETED:
                self._base.pop(key, None)
            else:
                self._base[key] = value
        self._local.clear()


class _RequestState:
    """Container tracking deferred mutations for a single request."""

    def __init__(
        self,
        request_uid: str,
        anchor_dict: Dict[str, Any],
        anchor_len_dict: Dict[str, Any],
        anchor_info_dict: Dict[str, Any],
        weight_dict: Dict[str, Any],
        anchors: Dict[str, Any],
        global_anchor_info_dict: Dict[str, Any],
    ):
        self.request_uid = request_uid
        self.anchor_dict = _ScopedDict(anchor_dict)
        self.anchor_len_dict = _ScopedDict(anchor_len_dict)
        self.anchor_info_dict = _ScopedDict(anchor_info_dict)
        self.weight_dict = _ScopedDict(weight_dict)
        self.anchors = _ScopedDict(anchors)
        self.global_anchor_info = _ScopedDict(global_anchor_info_dict)

    def commit(self) -> None:
        self.anchor_dict.commit()
        self.anchor_len_dict.commit()
        self.anchor_info_dict.commit()
        self.weight_dict.commit()
        self.anchors.commit()
        self.global_anchor_info.commit()


class KVCOMMEngine:
    """Central coordinator for anchor-related KV cache interactions."""

    anchors: Dict[str, Any] = {}
    anchor_dict: Dict[str, Any] = {}
    anchor_len_dict: Dict[str, Any] = {}
    anchor_info_dict: Dict[str, Any] = {}
    weight_dict: Dict[str, Any] = {}
    global_anchor_info_dict: Dict[str, Any] = {}

    _request_lock = threading.Lock()
    _request_states: Dict[str, _RequestState] = {}
    _active_requests: set[str] = set()
    _staged_commits: List[_RequestState] = []

    def __init__(self, llm: "LLMChat"):
        self.llm = llm
        self._warning_prefix = "[KVCOMMEngine]"

    def _log_warning(self, message: str) -> None:
        logger.opt(colors=True).warning("<yellow>{}</yellow> {}", self._warning_prefix, message)

    @staticmethod
    def _stack_cache_tensors(cache: DynamicCache) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(cache.key_cache), torch.stack(cache.value_cache)

    @staticmethod
    def _placeholder_length(cache: DynamicCache) -> int:
        return cache.key_cache[0].shape[-2]

    def _rotate_segment_caches(self, segment_meta: Dict[str, Any]) -> Tuple[DynamicCache, DynamicCache]:
        rotated_placeholder = self.apply_rotary_pos_emb(
            segment_meta["ph_cache"],
            offset=segment_meta["start"] - segment_meta["drop_num"] + segment_meta["offset_before"],
            drop_num=segment_meta["drop_num"],
        )
        rotated_prefix = self.apply_rotary_pos_emb(
            segment_meta["pf_kv"],
            offset=segment_meta["offset_after"],
        )
        return rotated_placeholder, rotated_prefix

    @classmethod
    def _get_request_state(cls, request_uid: str) -> _RequestState:
        """Return or create a request-scoped state container under a lock."""
        if not request_uid:
            raise ValueError("request_uid must be provided for scoped anchor updates.")
        with cls._request_lock:
            state = cls._request_states.get(request_uid)
            if state is None:
                state = _RequestState(
                    request_uid,
                    cls.anchor_dict,
                    cls.anchor_len_dict,
                    cls.anchor_info_dict,
                    cls.weight_dict,
                    cls.anchors,
                    cls.global_anchor_info_dict,
                )
                cls._request_states[request_uid] = state
                cls._active_requests.add(request_uid)
            return state

    @classmethod
    def finalize_request(cls, request_uid: str) -> None:
        """Stage a request state for commit; flush when last active request ends."""
        if not request_uid:
            return
        with cls._request_lock:
            state = cls._request_states.pop(request_uid, None)
            if state is None:
                return
            cls._staged_commits.append(state)
            cls._active_requests.discard(request_uid)
            if not cls._active_requests:
                cls._commit_staged_states_locked()

    @classmethod
    def _commit_staged_states_locked(cls) -> None:
        """Commit all staged request states into the global dictionaries."""
        if not cls._staged_commits:
            return
        for state in cls._staged_commits:
            state.commit()
        cls._staged_commits.clear()

    def resolve_request_state(self, request_uid: str) -> _RequestState:
        """Public alias to access or create the request-scoped state."""
        return self._get_request_state(request_uid)

    def get_request_state(self, request_uid: str) -> _RequestState:
        """Return the request state; identical to resolve_request_state."""
        return self.resolve_request_state(request_uid)

    @staticmethod
    def anchor_signature(anchor_list: List[Dict[str, Any]]) -> Tuple[int, ...]:
        """Create a lightweight fingerprint for the active anchors."""
        return tuple(id(anchor) for anchor in anchor_list)

    def _get_cached_anchor_weights(
        self,
        request_uid: str,
        ph_id: str,
        message: str,
        signature: Tuple[int, ...],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Look up cached anchor weights for a specific placeholder/message/signature."""
        state = self.resolve_request_state(request_uid)
        bucket = state.weight_dict.get(ph_id)
        if bucket is None:
            return None
        entry = bucket.get(message)
        if not entry:
            return None
        if entry.get("anchor_signature") != signature:
            return None
        return entry

    def _set_cached_anchor_weights(
        self,
        request_uid: str,
        ph_id: str,
        message: str,
        entry: Dict[str, torch.Tensor],
    ) -> None:
        """Store computed anchor weights for reuse within the same request."""
        state = self.resolve_request_state(request_uid)
        bucket = state.weight_dict.setdefault(ph_id, {})
        bucket[message] = entry

    @staticmethod
    def _select_anchor_indices(anchor_list: List[Dict[str, Any]], placeholder_len: int) -> List[int]:
        """Pick anchors whose placeholder span covers the current placeholder length."""
        return [
            idx
            for idx, anchor in enumerate(anchor_list)
            if anchor["ph_key_embedding"].shape[-2] >= placeholder_len
        ]

    def _compute_anchor_weight_entry(
        self,
        anchor_list: List[Dict[str, Any]],
        anchor_indices: List[int],
        real_key_embedding: torch.Tensor,
        real_value_embedding: torch.Tensor,
        placeholder_len: int,
        temperature: float,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute attention-like weights between the real placeholder segment and stored anchors."""
        if not anchor_indices:
            return None
        used_anchors = [anchor_list[idx] for idx in anchor_indices]

        anchor_key_placeholder = torch.stack(
            [anchor["ph_key_embedding"][..., -placeholder_len:, :] for anchor in used_anchors]
        )
        anchor_value_placeholder = torch.stack(
            [anchor["ph_value_embedding"][..., -placeholder_len:, :] for anchor in used_anchors]
        )
        real_key_placeholder = real_key_embedding[..., -placeholder_len:, :]
        real_value_placeholder = real_value_embedding[..., -placeholder_len:, :]

        sims_key_prefix = (real_key_placeholder.unsqueeze(0) - anchor_key_placeholder).norm(2, dim=-2)
        weights_key_prefix = torch.softmax(-sims_key_prefix.float() / temperature, dim=0).unsqueeze(-2)

        sims_val_prefix = (real_value_placeholder.unsqueeze(0) - anchor_value_placeholder).norm(2, dim=-2)
        weights_value_prefix = torch.softmax(-sims_val_prefix.float() / temperature, dim=0).unsqueeze(-2)

        anchor_key_placeholder = torch.stack(
            [anchor["ph_key_embedding"][..., :placeholder_len, :] for anchor in used_anchors]
        )
        anchor_value_placeholder = torch.stack(
            [anchor["ph_value_embedding"][..., :placeholder_len, :] for anchor in used_anchors]
        )

        sims_key_placeholder = (
            real_key_placeholder.unsqueeze(0) - anchor_key_placeholder
        ).abs().mean(dim=(-5, -4, -3, -1), keepdim=True)
        weights_key_placeholder = torch.softmax(-sims_key_placeholder.float() / temperature, dim=0)

        sims_val_placeholder = (
            real_value_placeholder.unsqueeze(0) - anchor_value_placeholder
        ).abs().mean(dim=(-5, -4, -3, -1), keepdim=True)
        weights_value_placeholder = torch.softmax(-sims_val_placeholder.float() / temperature, dim=0)

        return {
            "anchor_index": anchor_indices,
            "weights_key_for_prefix": weights_key_prefix.detach(),
            "weights_value_for_prefix": weights_value_prefix.detach(),
            "weights_key_for_placeholder": weights_key_placeholder.detach(),
            "weights_value_for_placeholder": weights_value_placeholder.detach(),
        }

    def offset_kv_cache_pair(
        self,
        ph_id: str,
        message: str,
        request_uid: str,
        base_placeholder_cache: DynamicCache,
        base_prefix_cache: DynamicCache,
        anchor_list: List[Dict],
        temperature: int = 1,
    ) -> Tuple[DynamicCache, DynamicCache]:
        """Blend base caches with anchor deltas weighted by similarity."""
        placeholder_len = int(base_placeholder_cache._seen_tokens)
        if placeholder_len <= 0:
            self._log_warning("real_placeholder_kv_cache has no tokens, skip updating.")
            return base_placeholder_cache, base_prefix_cache

        real_key_embedding, real_value_embedding = self._stack_cache_tensors(base_placeholder_cache)

        anchor_signature = self.anchor_signature(anchor_list)

        cache_entry = self._get_cached_anchor_weights(
            request_uid,
            ph_id,
            message=message,
            signature=anchor_signature,
        )

        if cache_entry is None:
            anchor_index = self._select_anchor_indices(anchor_list, placeholder_len)
            if not anchor_index:
                if anchor_list:
                    self._log_warning(
                        f"No anchors cover placeholder {ph_id} for Agent {self.llm.node_id} ({self.llm.role})."
                    )
                return base_placeholder_cache.copy(), base_prefix_cache.copy()

            cache_entry = self._compute_anchor_weight_entry(
                anchor_list,
                anchor_index,
                real_key_embedding,
                real_value_embedding,
                placeholder_len,
                float(temperature),
            )
            if cache_entry is None:
                return base_placeholder_cache.copy(), base_prefix_cache.copy()
            cache_entry["anchor_signature"] = anchor_signature
            cache_entry["placeholder_len"] = placeholder_len
            self._set_cached_anchor_weights(request_uid, ph_id, message, cache_entry)
        else:
            anchor_index = cache_entry["anchor_index"]
            if not anchor_index:
                return base_placeholder_cache.copy(), base_prefix_cache.copy()

        weights_key_for_prefix = cache_entry["weights_key_for_prefix"]
        weights_value_for_prefix = cache_entry["weights_value_for_prefix"]
        weights_key_for_placeholder = cache_entry["weights_key_for_placeholder"]
        weights_value_for_placeholder = cache_entry["weights_value_for_placeholder"]

        prefix_key_delta_stack = torch.stack(
            [anchor_list[i][f"{self.llm.node_id}_pf_key_delta"] for i in anchor_index]
        )
        layer_total_delta_key_for_prefix = (
            weights_key_for_prefix * prefix_key_delta_stack
        ).sum(0)

        prefix_value_delta_stack = torch.stack(
            [anchor_list[i][f"{self.llm.node_id}_pf_value_delta"] for i in anchor_index]
        )
        layer_total_value_delta_for_prefix = (
            weights_value_for_prefix * prefix_value_delta_stack
        ).sum(0)

        placeholder_key_delta_stack = torch.stack(
            [
                anchor_list[i][f"{self.llm.node_id}_ph_key_delta"][..., :placeholder_len, :]
                for i in anchor_index
            ]
        )
        layer_total_delta_key_for_placeholder = (
            weights_key_for_placeholder * placeholder_key_delta_stack
        ).sum(0)

        placeholder_value_delta_stack = torch.stack(
            [
                anchor_list[i][f"{self.llm.node_id}_ph_value_delta"][..., :placeholder_len, :]
                for i in anchor_index
            ]
        )
        layer_total_value_delta_for_placeholder = (
            weights_value_for_placeholder * placeholder_value_delta_stack
        ).sum(0)

        new_placeholder_cache = type(base_placeholder_cache)()
        updated_placeholder_key = (
            real_key_embedding + layer_total_delta_key_for_placeholder.to(real_key_embedding.dtype)
        )
        updated_placeholder_key[0] = real_key_embedding[0]
        new_placeholder_cache.key_cache = list(updated_placeholder_key)

        updated_placeholder_value = (
            real_value_embedding + layer_total_value_delta_for_placeholder.to(real_value_embedding.dtype)
        )
        updated_placeholder_value[0] = real_value_embedding[0]
        new_placeholder_cache.value_cache = list(updated_placeholder_value)
        new_placeholder_cache._seen_tokens = base_placeholder_cache._seen_tokens

        base_prefix_key, base_prefix_value = self._stack_cache_tensors(base_prefix_cache)

        new_prefix_cache = type(base_prefix_cache)()
        updated_prefix_key = base_prefix_key + layer_total_delta_key_for_prefix.to(base_prefix_key.dtype)
        updated_prefix_key[0] = base_prefix_key[0]
        new_prefix_cache.key_cache = list(updated_prefix_key)

        updated_prefix_value = (
            base_prefix_value + layer_total_value_delta_for_prefix.to(base_prefix_value.dtype)
        )
        updated_prefix_value[0] = base_prefix_value[0]
        new_prefix_cache.value_cache = list(updated_prefix_value)
        new_prefix_cache._seen_tokens = base_prefix_cache._seen_tokens

        return new_placeholder_cache, new_prefix_cache

    def predict_as_anchor(
        self,
        candidate_kv_cache: DynamicCache,
        anchor_kv_cache_list: List[Dict],
        anchor_len_list: List[Tuple[int, int]],
        anchor_activated_list: List[int],
        top_p: float = 0.9,
        entropy_eps: float = 1e-40,
        test_time: bool = False,
    ) -> Tuple[bool, List[int]]:
        if len(anchor_kv_cache_list) in [0, 1]:
            return True, anchor_activated_list

        if test_time:
            torch.cuda.synchronize()
            start_time = perf_counter()
        k = candidate_kv_cache.value_cache[0].shape[-2]
        anchor_available = [i for i, (j, _accum_j) in enumerate(anchor_len_list) if j >= k]

        if len(anchor_len_list) != len(anchor_kv_cache_list):
            self._log_warning(
                "The length of anchor_len_list is not equal to the length of anchor_available, "
                f"with {len(anchor_len_list)} and {len(anchor_available)}."
            )
            return True, anchor_activated_list

        if len(anchor_available) > 1:
            candidate_value_embedding = torch.stack(candidate_kv_cache.value_cache)[..., :k, :]
            anchor_value_embedding = torch.stack(
                [anchor_kv_cache_list[i]["ph_value_embedding"][..., :k, :] for i in anchor_available]
            )
            diff = (candidate_value_embedding.unsqueeze(0) - anchor_value_embedding).norm(2, dim=(1, 2, 3, 4, 5))
            sim = torch.softmax(-diff.float(), dim=0)
            threshold = self.llm.config.threshold
            entropy = -(sim * (sim + entropy_eps).log2()).sum()
            if entropy > threshold * torch.log2(torch.tensor(sim.shape[0])):
                logger.opt(colors=True).debug(
                    f"<yellow>Entropy {entropy:.4f} exceeds threshold {threshold * torch.log2(torch.tensor(sim.shape[0])):.4f}, "
                    "skip activating anchors.</yellow>"
                )
                if test_time:
                    torch.cuda.synchronize()
                    end_time = perf_counter()
                    logger.opt(colors=True).debug(
                        f"<cyan>Latency for Anchor prediction: {end_time - start_time} s</cyan>"
                    )
                return True, anchor_activated_list
            sorted_sim, sorted_indices = torch.sort(sim, descending=True)
            cumulative_sum = torch.cumsum(sorted_sim, dim=0)
            cutoff_index_candidates = (cumulative_sum < top_p).nonzero(as_tuple=True)[0]
            cutoff_index = cutoff_index_candidates[-1] if len(cutoff_index_candidates) > 0 else len(sorted_sim) - 1
            selected_indices = sorted_indices[:cutoff_index + 1]
            for i in selected_indices:
                if anchor_available[i] >= len(anchor_activated_list):
                    self._log_warning(
                        "anchor_available index "
                        f"{anchor_available[i]} out of range for anchor_activated_list with length "
                        f"{len(anchor_activated_list)}"
                    )
                    continue
                anchor_activated_list[anchor_available[i]] += 1
            if test_time:
                torch.cuda.synchronize()
                end_time = perf_counter()
                logger.opt(colors=True).debug(
                    f"<cyan>Latency for Anchor prediction: {end_time - start_time} s</cyan>"
                )
            return False, anchor_activated_list
        logger.opt(colors=True).debug("<yellow>No available anchors to activate.</yellow>")
        return True, anchor_activated_list

    def update_anchor(self, request_uid: str, ph_id: str, window_length: int = 5) -> None:
        """
        Update the anchor list by filtering out the least frequent anchors in the oldest anchor set.
        """
        state = self.resolve_request_state(request_uid)
        anchor_store = state.anchors.setdefault(ph_id, {})
        anchor_info_dict = state.anchor_info_dict.setdefault(ph_id, {})
        info_list = list(anchor_info_dict.values())[:window_length]
        if not info_list:
            return
        min_idx = info_list.index(min(info_list))
        message = list(anchor_info_dict.keys())[min_idx]
        anchor_store.pop(message, None)
        state.anchor_len_dict.setdefault(ph_id, {}).pop(message, None)
        freq = anchor_info_dict.pop(message, None)
        state.global_anchor_info.setdefault(ph_id, {}).pop(message, None)
        self._log_warning(
            f"Removed anchor for message '{message}' in {self.llm.node_id} ({self.llm.role}) due to low frequency: {freq}"
        )

    def set_anchor(
        self,
        request_uid: str,
        message: str,
        ph_id_list: List[str],
        real_placeholder_cache_list: List[DynamicCache],
        real_prefix_cache_list: List[DynamicCache],
        base_placeholder_cache_list: List[DynamicCache],
        base_prefix_cache_list: List[DynamicCache],
        max_anchor_num: int = 20,
        window_length: int = 5,
    ) -> Dict[str, List[List[Dict]]]:
        """Populate or update anchor store with per-placeholder deltas for a message."""
        state = self.resolve_request_state(request_uid)
        anchor_store = state.anchors

        n = len(real_placeholder_cache_list)
        real_pf = real_prefix_cache_list[-n:]
        base_pf = base_prefix_cache_list[-n:]
        anchor_flags = {
            ph_id: state.anchor_dict.setdefault(ph_id, {})
            for ph_id in ph_id_list
        }

        def _should_materialise(ph_id: str) -> bool:
            return anchor_flags[ph_id].get(message) is True

        def _make_anchor(i, ph_id, real_ph, base_ph, real_pf, base_pf):
            ph_key_real, ph_val_real = self._stack_cache_tensors(real_ph)
            ph_key_base, ph_val_base = self._stack_cache_tensors(base_ph)
            pf_key_real, pf_val_real = self._stack_cache_tensors(real_pf)
            pf_key_base, pf_val_base = self._stack_cache_tensors(base_pf)

            entry = {
                "ph_key_embedding": ph_key_base,
                "ph_value_embedding": ph_val_base,
                f"{self.llm.node_id}_ph_key_delta": ph_key_real - ph_key_base,
                f"{self.llm.node_id}_ph_value_delta": ph_val_real - ph_val_base,
                f"{self.llm.node_id}_pf_key_delta": pf_key_real - pf_key_base,
                f"{self.llm.node_id}_pf_value_delta": pf_val_real - pf_val_base,
            }
            return i, entry

        args = [
            (i, ph_id, real_ph, base_ph, real_pf_i, base_pf_i)
            for i, (ph_id, real_ph, base_ph, real_pf_i, base_pf_i) in enumerate(
                zip(ph_id_list, real_placeholder_cache_list, base_placeholder_cache_list, real_pf, base_pf)
            )
            if _should_materialise(ph_id)
        ]

        if not args:
            return anchor_store

        results = list(self.llm._map_in_pool(_make_anchor, args, timeout=30))
        results.sort(key=lambda x: x[0])
        anchor_dict = {i: entry for i, entry in results}

        accumulate_len = 0
        for i in range(n):
            placeholder_len = self._placeholder_length(real_placeholder_cache_list[i])
            if i not in anchor_dict:
                accumulate_len += placeholder_len
                continue
            entry = anchor_dict[i]
            over_store = len(anchor_store.setdefault(ph_id_list[i], {})) > max_anchor_num
            if over_store:
                self.update_anchor(request_uid, ph_id_list[i], window_length)

            if message not in anchor_store[ph_id_list[i]]:
                anchor_store[ph_id_list[i]][message] = entry
                info_bucket = state.anchor_info_dict.setdefault(ph_id_list[i], {})
                info_bucket[message] = 0

                length_bucket = state.anchor_len_dict.setdefault(ph_id_list[i], {})
                length_bucket[message] = [
                    placeholder_len,
                    accumulate_len,
                ]

                state.global_anchor_info.setdefault(ph_id_list[i], {}).setdefault(
                    message,
                    [0, placeholder_len],
                )
            else:
                anchor_store[ph_id_list[i]][message].update(entry)
            accumulate_len += placeholder_len

        return anchor_store
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rotate_tensor(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        """Apply RoPE rotation using provided cos/sin tables."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (tensor * cos) + (self._rotate_half(tensor) * sin)

    def apply_rotary_pos_emb(
        self,
        ph_cache: DynamicCache,
        offset: int,
        drop_num: int = 0,
    ) -> DynamicCache:
        """Rotate placeholder cache keys by absolute offset (with optional drop)."""
        rotate_emb = self.llm.model.model.rotary_emb
        if drop_num > 0:
            new_ph_cache = ph_cache.copy().slice_(start=drop_num)
        else:
            new_ph_cache = ph_cache.copy()
        position_ids = (
            torch.ones(new_ph_cache._seen_tokens, dtype=torch.long)
            .unsqueeze(0)
            .to(self.llm.model.device)
            * offset
        )
        cos, sin = rotate_emb(new_ph_cache.key_cache[0], position_ids)

        kv = torch.stack(new_ph_cache.key_cache, dim=0)
        kv_rot = self.rotate_tensor(kv, cos, sin)
        new_ph_cache.key_cache = list(kv_rot)
        return new_ph_cache

    def fetch_shared_cache(
        self,
        ph_id: str,
        message: str,
    ) -> Tuple[DynamicCache, Dict[str, torch.Tensor], int]:
        """Retrieve shared KV cache and ids for a placeholder given message context."""
        shared_memory = self.llm._shared_kv_cache_memory

        if "user_question" in ph_id:
            return (
                shared_memory["input"][message][-1],
                shared_memory["input_ids"][message][-1],
                shared_memory["input_drop_num"][message][-1],
            )

        type_str, node_id, *rest = ph_id.split("_")
        is_current = (rest and rest[0] == "current")

        key_prefix = "condition" if type_str == "condition" else "response"
        slot_idx = -1 if is_current else -2

        node_memory = shared_memory[node_id]

        def _get_slot(bucket_key: str):
            bucket = node_memory.get(bucket_key, {})
            values = bucket.get(message)
            if not values:
                return None
            try:
                return values[slot_idx]
            except IndexError:
                return None

        ph_cache = _get_slot(key_prefix)
        ph_cache_ids = _get_slot(f"{key_prefix}_ids")
        drop_num = _get_slot(f"{key_prefix}_drop_num")

        if ph_cache is None:
            raise RuntimeError(
                f"fetch_shared_cache: placeholder {ph_id} for message='{message}' not found."
            )

        return ph_cache, ph_cache_ids, drop_num

    @staticmethod
    def trim_token_ids(ids_dict: Dict[str, torch.Tensor], drop_num: int) -> Dict[str, torch.Tensor]:
        if drop_num == 0:
            return ids_dict
        return {
            key: None if value is None else value[:, drop_num:]
            for key, value in ids_dict.items()
        }

    def update_kv_cache_segment(
        self,
        request_uid: str,
        message: str,
        m: Dict[str, Any],
        anchors_for_ph: List[Dict],
    ) -> Tuple[int, DynamicCache, Dict[str, torch.Tensor]]:
        """Rotate and offset a single placeholder/prefix segment for kv_reuse mode."""
        new_ph, new_pf = self._rotate_segment_caches(m)
        new_ph, new_pf = self.offset_kv_cache_pair(
            m["ph_id"], message, request_uid, new_ph, new_pf, anchors_for_ph, temperature=1
        )

        seg_cache = new_ph.concat_([new_pf])
        seg_token_ids = concat(self.trim_token_ids(m["ph_cache_ids"], m["drop_num"]), m["pf_ids"])

        return m["idx"], seg_cache, seg_token_ids

    def process_anchor(
        self,
        message: str,
        m: Dict[str, Any],
    ) -> Tuple[int, DynamicCache, Dict[str, torch.Tensor]]:
        """Rotate and concatenate a single segment for dense_prefill mode."""
        new_ph, new_pf = self._rotate_segment_caches(m)
        seg_cache = new_ph.concat_([new_pf])
        seg_token_ids = concat(self.trim_token_ids(m["ph_cache_ids"], m["drop_num"]), m["pf_ids"])

        return m["idx"], seg_cache, seg_token_ids
