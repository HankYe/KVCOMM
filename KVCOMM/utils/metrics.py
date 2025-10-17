from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger


@dataclass(slots=True)
class GenerationResult:
    """Container describing a single model generation outcome."""

    text: str
    mode: str
    ttft: float
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestMetricsRecorder:
    """Tracks per-request agent outputs, reuse rates, and mode-specific TTFT."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._total_calls: int = 0
        self._total_reuse_calls: int = 0
        self._ttft_stats: Dict[str, Dict[str, float]] = {}

    def start_request(
        self,
        *,
        request_uid: str,
        batch_index: Optional[int],
        task: Optional[str],
        execution_mode: str,
    ) -> None:
        """Initialise bookkeeping for a new request."""
        with self._lock:
            self._requests[request_uid] = {
                "batch_index": batch_index,
                "task": task,
                "execution_mode": execution_mode,
                "agents": [],
                "kv_reuse_count": 0,
                "total_count": 0,
            }

    def record_agent_output(
        self,
        *,
        request_uid: str,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        generation: GenerationResult | None,
    ) -> None:
        """Log an agent's generation output and update reuse statistics."""
        if generation is None:
            return

        with self._lock:
            request_entry = self._requests.setdefault(
                request_uid,
                {
                    "batch_index": None,
                    "task": None,
                    "execution_mode": "unknown",
                    "agents": [],
                    "kv_reuse_count": 0,
                    "total_count": 0,
                },
            )

            agent_record = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "text": generation.text,
            }
            if generation.metadata:
                agent_record["metadata"] = generation.metadata
            agents_list = request_entry["agents"]
            replaced = False
            for idx, existing in enumerate(agents_list):
                if existing["agent_id"] == agent_id:
                    agents_list[idx] = agent_record
                    replaced = True
                    break
            if not replaced:
                agents_list.append(agent_record)


            request_entry["total_count"] = len(agents_list)
            request_entry["kv_reuse_count"] = sum(
                1 for entry in agents_list if entry.get("mode") == "kv_reuse"
            )


            stats = self._ttft_stats.setdefault(
                generation.mode, {"sum": 0.0, "count": 0.0}
            )
            stats["sum"] += generation.ttft
            stats["count"] += 1
            avg_ttft = stats["sum"] / stats["count"] if stats["count"] else 0.0

            ttft_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "mode_avg_ttft": avg_ttft,
            }
            metadata = generation.metadata or {}
            preprocess_latency = metadata.get("preprocess_latency")
            if preprocess_latency is not None:
                ttft_payload["preprocess_latency"] = preprocess_latency
            generation_ttft = metadata.get("generation_ttft")
            if generation_ttft is not None:
                ttft_payload["generation_ttft"] = generation_ttft
            logger.opt(colors=True).info(
                "<cyan>[TTFT:{mode}]</cyan> {}",
                json.dumps(ttft_payload, ensure_ascii=False),
                mode=generation.mode,
            )

            output_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "text": generation.text,
            }
            logger.opt(colors=True).info(
                "<green>[AGENT OUTPUT]</green> {}",
                json.dumps(output_payload, ensure_ascii=False),
            )

    def finalize_request(self, request_uid: str) -> Optional[float]:
        """Compute and log per-request reuse statistics."""
        with self._lock:
            request_entry = self._requests.pop(request_uid, None)
            if request_entry is None:
                return None

            kv_reuse = request_entry.get("kv_reuse_count", 0)
            total = request_entry.get("total_count", 0)
            reuse_rate = (kv_reuse / total) if total else 0.0

            payload = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "task": request_entry.get("task"),
                "execution_mode": request_entry.get("execution_mode"),
                "reuse_rate": reuse_rate,
                "kv_reuse_count": kv_reuse,
                "total_agents": total,
            }
            logger.opt(colors=True).info(
                "<magenta>[REQUEST REUSE]</magenta> {}",
                json.dumps(payload, ensure_ascii=False),
            )

            self._total_calls += total
            self._total_reuse_calls += kv_reuse
            return reuse_rate

    def log_cumulative(self, *, batch_index: Optional[int]) -> float:
        """Log the cumulative average reuse rate across all processed requests."""
        with self._lock:
            if self._total_calls == 0:
                cumulative = 0.0
            else:
                cumulative = self._total_reuse_calls / self._total_calls

            payload = {
                "batch_index": batch_index,
                "cumulative_reuse_rate": cumulative,
                "kv_reuse_calls": self._total_reuse_calls,
                "total_agent_calls": self._total_calls,
            }
            logger.opt(colors=True).info(
                "<yellow>[CUMULATIVE REUSE]</yellow> {}",
                json.dumps(payload, ensure_ascii=False),
            )
            return cumulative


metrics_recorder = RequestMetricsRecorder()
