from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict
import os


@dataclass(frozen=True)
class KVCommConfig:
    """
    Configuration for KV communication and scheduling.

    Args:
        threshold (float): The threshold for the KV communication.
        thread_pool_workers (int): The number of threads to use for the thread pool.
        worker_timeout (float): The timeout for the worker.
    """
    threshold: float = 0.3
    max_anchor_num: int = 20
    window_size: int = 5
    thread_pool_workers: int = 8
    worker_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "KVCommConfig":
        """Create a config from environment variables with safe defaults."""
        return cls(
            threshold=float(os.environ.get("THRESHOLD", cls.threshold)),
            max_anchor_num=int(os.environ.get("MAX_ANCHOR_NUM", cls.max_anchor_num)),
            window_size=int(os.environ.get("WINDOW_SIZE", cls.window_size)),
            thread_pool_workers=int(os.environ.get("KVCOMM_THREAD_WORKERS", cls.thread_pool_workers)),
            worker_timeout=float(os.environ.get("KVCOMM_WORKER_TIMEOUT", cls.worker_timeout)),
        ).validate()

    def apply_overrides(self, **overrides: Any) -> "KVCommConfig":
        """Return a copy with provided non-None fields overridden."""
        current: Dict[str, Any] = asdict(self)
        for key, value in overrides.items():
            if value is None or key not in current:
                continue
            current[key] = value
        return replace(self, **current).validate()

    def validate(self) -> "KVCommConfig":
        """Validate value ranges and return self."""
        if self.thread_pool_workers <= 0:
            raise ValueError("thread_pool_workers must be positive")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        return self
