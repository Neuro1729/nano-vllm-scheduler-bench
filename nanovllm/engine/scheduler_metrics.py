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
    preemption_count: int = 0
    preempted_request_ids: set[int] = field(default_factory=set)
    recomputed_tokens: int = 0
    max_running_requests: int = 0
    max_waiting_requests: int = 0
    # Legacy: on real this equals failed candidates (HOL break, ≤1 per step).
    allocation_failures: int = 0
    # Comparable across policies: iterations with ≥1 waiting can_allocate==-1.
    allocation_failure_steps: int = 0
    # Every waiting request that failed can_allocate.
    allocation_failed_candidates: int = 0
    # Waiting requests skipped to continue scan (0 on real; parity HOL-skip).
    hol_skipped_requests: int = 0
    waiting_candidates_examined: int = 0
    peak_kv_blocks_used: int = 0
    peak_kv_utilization: float = 0.0
    _kv_util_sum: float = 0.0
    _kv_util_count: int = 0
    _had_alloc_failure_this_iteration: bool = False

    def reset(self) -> None:
        self.scheduler_iterations = 0
        self.prefill_iterations = 0
        self.decode_iterations = 0
        self.mixed_iterations = 0
        self.preemption_count = 0
        self.preempted_request_ids.clear()
        self.recomputed_tokens = 0
        self.max_running_requests = 0
        self.max_waiting_requests = 0
        self.allocation_failures = 0
        self.allocation_failure_steps = 0
        self.allocation_failed_candidates = 0
        self.hol_skipped_requests = 0
        self.waiting_candidates_examined = 0
        self.peak_kv_blocks_used = 0
        self.peak_kv_utilization = 0.0
        self._kv_util_sum = 0.0
        self._kv_util_count = 0
        self._had_alloc_failure_this_iteration = False

    def begin_iteration(self) -> None:
        self._had_alloc_failure_this_iteration = False

    def examine_waiting_candidate(self) -> None:
        self.waiting_candidates_examined += 1

    def record_allocation_failure(self, *, hol_skipped: bool) -> None:
        self.allocation_failures += 1
        self.allocation_failed_candidates += 1
        if not self._had_alloc_failure_this_iteration:
            self._had_alloc_failure_this_iteration = True
            self.allocation_failure_steps += 1
        if hol_skipped:
            self.hol_skipped_requests += 1

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

    @property
    def allocation_candidate_failure_rate(self) -> float:
        if self.waiting_candidates_examined == 0:
            return 0.0
        return self.allocation_failed_candidates / self.waiting_candidates_examined

    def to_dict(self) -> dict:
        return {
            "scheduler_iterations": self.scheduler_iterations,
            "prefill_iterations": self.prefill_iterations,
            "decode_iterations": self.decode_iterations,
            "mixed_iterations": self.mixed_iterations,
            "preemption_count": self.preemption_count,
            "preempted_requests": len(self.preempted_request_ids),
            "recomputed_tokens": self.recomputed_tokens,
            "max_running_requests": self.max_running_requests,
            "max_waiting_requests": self.max_waiting_requests,
            "allocation_failures": self.allocation_failures,
            "allocation_failure_steps": self.allocation_failure_steps,
            "allocation_failed_candidates": self.allocation_failed_candidates,
            "hol_skipped_requests": self.hol_skipped_requests,
            "waiting_candidates_examined": self.waiting_candidates_examined,
            "allocation_candidate_failure_rate": self.allocation_candidate_failure_rate,
            "peak_KV_blocks_used": self.peak_kv_blocks_used,
            "average_KV_utilization": self.average_kv_utilization,
            "peak_KV_utilization": self.peak_kv_utilization,
        }
