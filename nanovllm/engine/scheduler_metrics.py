"""Instrumentation-only scheduler counters. Does not affect scheduling decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchedulerMetrics:
    num_kv_blocks: int = 1
    scheduler_iterations: int = 0
    prefill_iterations: int = 0
    decode_iterations: int = 0
    mixed_iterations: int = 0
    scheduled_prefill_tokens: int = 0
    scheduled_decode_tokens: int = 0
    chunked_prefill_count: int = 0
    preemption_count: int = 0
    preempted_request_ids: set[int] = field(default_factory=set)
    recomputed_tokens: int = 0
    max_running_requests: int = 0
    max_waiting_requests: int = 0
    allocation_failures: int = 0
    peak_kv_blocks_used: int = 0
    peak_kv_utilization: float = 0.0
    _kv_util_sum: float = 0.0
    _kv_util_count: int = 0

    def reset(self) -> None:
        self.scheduler_iterations = 0
        self.prefill_iterations = 0
        self.decode_iterations = 0
        self.mixed_iterations = 0
        self.scheduled_prefill_tokens = 0
        self.scheduled_decode_tokens = 0
        self.chunked_prefill_count = 0
        self.preemption_count = 0
        self.preempted_request_ids.clear()
        self.recomputed_tokens = 0
        self.max_running_requests = 0
        self.max_waiting_requests = 0
        self.allocation_failures = 0
        self.peak_kv_blocks_used = 0
        self.peak_kv_utilization = 0.0
        self._kv_util_sum = 0.0
        self._kv_util_count = 0

    def observe_queues(self, num_waiting: int, num_running: int, used_blocks: int) -> None:
        self.max_waiting_requests = max(self.max_waiting_requests, num_waiting)
        self.max_running_requests = max(self.max_running_requests, num_running)
        self.peak_kv_blocks_used = max(self.peak_kv_blocks_used, used_blocks)
        util = used_blocks / max(self.num_kv_blocks, 1)
        self.peak_kv_utilization = max(self.peak_kv_utilization, util)
        self._kv_util_sum += util
        self._kv_util_count += 1

    @property
    def average_kv_utilization(self) -> float:
        if self._kv_util_count == 0:
            return 0.0
        return self._kv_util_sum / self._kv_util_count

    def to_dict(self) -> dict:
        return {
            "scheduler_iterations": self.scheduler_iterations,
            "prefill_iterations": self.prefill_iterations,
            "decode_iterations": self.decode_iterations,
            "mixed_iterations": self.mixed_iterations,
            "scheduled_prefill_tokens": self.scheduled_prefill_tokens,
            "scheduled_decode_tokens": self.scheduled_decode_tokens,
            "chunked_prefill_count": self.chunked_prefill_count,
            "preemption_count": self.preemption_count,
            "preempted_requests": len(self.preempted_request_ids),
            "recomputed_tokens": self.recomputed_tokens,
            "max_running_requests": self.max_running_requests,
            "max_waiting_requests": self.max_waiting_requests,
            "allocation_failures": self.allocation_failures,
            "peak_KV_blocks_used": self.peak_kv_blocks_used,
            "average_KV_utilization": self.average_kv_utilization,
            "peak_KV_utilization": self.peak_kv_utilization,
        }
