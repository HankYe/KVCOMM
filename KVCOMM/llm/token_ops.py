import torch
from typing import Optional

def _normalize_slice_indices(
    token_inputs, start: Optional[int], end: Optional[int]
):
    """Convert negative indices and None into concrete slice bounds."""
    total = token_inputs['input_ids'].shape[-1]
    if start is None:
        start = 0
    elif start < 0:
        start = total + start
    if end is None:
        end = total
    elif end < 0:
        end = total + end
    return start, end


def slice_(token_inputs, start: Optional[int] = None, end: Optional[int] = None):
    """
    In-place slice of the token inputs along the time dimension.
    Negative indexing is allowed.

    Examples:
        - slice_(end=-10) removes the last 10 tokens.
        - slice_(start=5, end=15) keeps tokens 5..14.
    """
    start, end = _normalize_slice_indices(token_inputs, start, end)
    new_length = end - start

    if new_length <= 0:

        if token_inputs['input_ids'] is not None:
            batch = token_inputs['input_ids'].size(0)
            token_inputs['input_ids'] = token_inputs['input_ids'].new_empty((batch, 0))
        if token_inputs['attention_mask'] is not None:
            batch = token_inputs['attention_mask'].size(0)
            token_inputs['attention_mask'] = token_inputs['attention_mask'].new_empty((batch, 0))
        if 'position_ids' in token_inputs.keys() and token_inputs['position_ids'] is not None:
            batch = token_inputs['position_ids'].size(0)
            token_inputs['position_ids'] = token_inputs['position_ids'].new_empty((batch, 0))
        return token_inputs

    if token_inputs['input_ids'] is not None:
        token_inputs['input_ids'] = token_inputs['input_ids'][..., start:end]
    if token_inputs['attention_mask'] is not None:
        token_inputs['attention_mask'] = token_inputs['attention_mask'][..., start:end]
    if 'position_ids' in token_inputs.keys() and token_inputs['position_ids'] is not None:
        token_inputs['position_ids'] = token_inputs['position_ids'][..., start:end]

    return token_inputs


def slice(token_inputs, start: Optional[int] = None, end: Optional[int] = None):
    """
    Returns a new TokenInput with input_ids and attention_mask sliced along the time dimension.
    Negative indexing is allowed.
    """
    new_obj = type(token_inputs)()
    start, end = _normalize_slice_indices(token_inputs, start, end)
    new_length = end - start

    if new_length <= 0:
        return new_obj

    if token_inputs['input_ids'] is not None:
        new_obj['input_ids'] = token_inputs['input_ids'][..., start:end]
    if token_inputs['attention_mask'] is not None:
        new_obj['attention_mask'] = token_inputs['attention_mask'][..., start:end]
    if 'position_ids' in token_inputs.keys() and token_inputs['position_ids'] is not None:
        new_obj['position_ids'] = token_inputs['position_ids'][..., start:end]

    return new_obj

def concat_(token_inputs, other):
    """
    In-place concatenation of the token inputs from another TokenInput along the time dimension.
    """
    if other is None:
        return token_inputs
    if not isinstance(other, list):
        others = [other]
    else:
        others = other
    if token_inputs['input_ids'] is None or token_inputs['input_ids'].size(-1) == 0:
        token_inputs['input_ids'] = torch.cat([o['input_ids'] for o in others if o['input_ids'] is not None], dim=-1)
    elif others[0]['input_ids'] is not None and others[0]['input_ids'].size(-1) != 0:
        token_inputs['input_ids'] = torch.cat([token_inputs['input_ids']] + [o['input_ids'] for o in others if o['input_ids'] is not None], dim=-1)

    if token_inputs['attention_mask'] is None or token_inputs['attention_mask'].size(-1) == 0:
        token_inputs['attention_mask'] = torch.cat([o['attention_mask'] for o in others if o['attention_mask'] is not None], dim=-1)
    elif others[0]['attention_mask'] is not None and others[0]['attention_mask'].size(-1) != 0:
        token_inputs['attention_mask'] = torch.cat([token_inputs['attention_mask']] + [o['attention_mask'] for o in others if o['attention_mask'] is not None], dim=-1)

    if 'position_ids' in token_inputs.keys() and 'position_ids' in others[0].keys():
        if token_inputs['position_ids'] is None or token_inputs['position_ids'].size(-1) == 0:
            token_inputs['position_ids'] = torch.cat([o['position_ids'] for o in others if o['position_ids'] is not None], dim=-1)
        elif others[0]['position_ids'] is not None and others[0]['position_ids'].size(-1) != 0:
            token_inputs['position_ids'] = torch.cat([token_inputs['position_ids']] + [o['position_ids'] for o in others if o['position_ids'] is not None], dim=-1)

    return token_inputs

def concat(token_inputs, other):
    """
    Returns a new TokenInput instance with token inputs concatenated along the time dimension.
    """
    new_obj = type(token_inputs)()
    if not isinstance(other, list):
        others = [other]
    else:
        others = other

    if token_inputs['input_ids'] is None or token_inputs['input_ids'].size(-1) == 0:
        new_obj['input_ids'] = torch.cat([o['input_ids'] for o in others if o['input_ids'] is not None], dim=-1)
    elif others[0]['input_ids'] is None or others[0]['input_ids'].size(-1) == 0:
        new_obj['input_ids'] = token_inputs['input_ids']
    else:
        new_obj['input_ids'] = torch.cat([token_inputs['input_ids']] + [o['input_ids'] for o in others if o['input_ids'] is not None], dim=-1)

    if token_inputs['attention_mask'] is None or token_inputs['attention_mask'].size(-1) == 0:
        new_obj['attention_mask'] = torch.cat([o['attention_mask'] for o in others if o['attention_mask'] is not None], dim=-1)
    elif others[0]['attention_mask'] is None or others[0]['attention_mask'].size(-1) == 0:
        new_obj['attention_mask'] = token_inputs['attention_mask']
    else:
        new_obj['attention_mask'] = torch.cat([token_inputs['attention_mask']] + [o['attention_mask'] for o in others if o['attention_mask'] is not None], dim=-1)

    if 'position_ids' in token_inputs.keys() and 'position_ids' in others[0].keys():
        if token_inputs['position_ids'] is None or token_inputs['position_ids'].size(-1) == 0:
            new_obj['position_ids'] = torch.cat([o['position_ids'] for o in others if o['position_ids'] is not None], dim=-1)
        elif others[0]['position_ids'] is None or others[0]['position_ids'].size(-1) == 0:
            new_obj['position_ids'] = token_inputs['position_ids']
        else:
            new_obj['position_ids'] = torch.cat([token_inputs['position_ids']] + [o['position_ids'] for o in others if o['position_ids'] is not None], dim=-1)
    return new_obj


def replace_(token_inputs, start: int, end: int, real):
    """In-place replace the slice [start, end) with token inputs from `real`."""
    left = slice(token_inputs, 0, start)
    right = slice(token_inputs, end)

    if (not isinstance(left, dict) and left.data == {}) or (isinstance(left, dict) and left == {}):
        left = concat_(real, right)
    elif (not isinstance(right, dict) and right.data == {}) or (isinstance(right, dict) and right == {}):
        left = concat_(left, real)
    else:
        left = concat_(left, real)
        left = concat_(left, right)

    token_inputs['input_ids'] = left['input_ids']
    token_inputs['attention_mask'] = left['attention_mask']
    if 'position_ids' in token_inputs.keys() and 'position_ids' in left.keys():
        token_inputs['position_ids'] = left['position_ids']
    return token_inputs

def replace(token_inputs, start: int, end: int, real):
    """
    Returns a new TokenInput where the slice [start, end) is replaced by `real`.
    """
    left = slice(token_inputs, 0, start)
    right = slice(token_inputs, end)

    if (not isinstance(left, dict) and left.data == {}) or (isinstance(left, dict) and left == {}):
        return concat_(real, right)
    if (not isinstance(right, dict) and right.data == {}) or (isinstance(right, dict) and right == {}):
        return concat_(left, real)

    left = concat_(left, real)
    left = concat_(left, right)
    return left
