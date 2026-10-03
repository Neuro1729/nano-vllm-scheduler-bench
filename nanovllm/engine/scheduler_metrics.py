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
    # Legacy: historically counted can_allocate==-1 hits. On real this is ~1/step
    # (HOL break); on parity it counts every HOL-skipped candidate. Prefer the
    # explicit metrics below for cross-branch comparisons.
    allocation_failures: int = 0
    # Comparable across policies: iterations where any waiting admit saw can_allocate==-1.
    allocation_failure_steps: int = 0
    # Every waiting request that failed can_allocate (parity may be >> real).
    allocation_failed_candidates: int = 0
    # Waiting requests skipped so a later candidate could be tried (parity only).
    hol_skipped_requests: int = 0
    # Waiting admission candidates inspected (heads considered in the admit loop).
    waiting_candidates_examined: int = 0
    # Times WAITING admission selected a non-head candidate (SJF reorders).
    waiting_reorders: int = 0
    # sjf_aging: successful admits chosen because age >= threshold.
    aging_promotions: int = 0
    max_waiting_age_steps: int = 0
    _admission_age_sum: int = 0
    _admission_age_count: int = 0
    # Policy name / threshold copied from Config for benchmark JSON.
    scheduler_policy: str = "fcfs"
    scheduler_aging_threshold: int = 128
    # MLFQ config mirror + counters
    mlfq_q0_quantum: int = 256
    mlfq_q1_quantum: int = 1024
    mlfq_boost_interval: int = 256
    mlfq_q0_service_tokens: int = 0
    mlfq_q1_service_tokens: int = 0
    mlfq_q2_service_tokens: int = 0
    mlfq_q0_admissions: int = 0
    mlfq_q1_demotions: int = 0
    mlfq_q2_demotions: int = 0
    mlfq_priority_boosts: int = 0
    mlfq_requests_boosted: int = 0
    mlfq_max_q0_size: int = 0
    mlfq_max_q1_size: int = 0
    mlfq_max_q2_size: int = 0
    completed_in_q0: int = 0
    completed_in_q1: int = 0
    completed_in_q2: int = 0
    _completion_level_sum: int = 0
    _completion_level_count: int = 0
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
        self.scheduled_prefill_tokens = 0
        self.scheduled_decode_tokens = 0
        self.chunked_prefill_count = 0
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
        self.waiting_reorders = 0
        self.aging_promotions = 0
        self.max_waiting_age_steps = 0
        self._admission_age_sum = 0
        self._admission_age_count = 0
        self.mlfq_q0_service_tokens = 0
        self.mlfq_q1_service_tokens = 0
        self.mlfq_q2_service_tokens = 0
        self.mlfq_q0_admissions = 0
        self.mlfq_q1_demotions = 0
        self.mlfq_q2_demotions = 0
        self.mlfq_priority_boosts = 0
        self.mlfq_requests_boosted = 0
        self.mlfq_max_q0_size = 0
        self.mlfq_max_q1_size = 0
        self.mlfq_max_q2_size = 0
        self.completed_in_q0 = 0
        self.completed_in_q1 = 0
        self.completed_in_q2 = 0
        self._completion_level_sum = 0
        self._completion_level_count = 0
        # Keep policy/config mirrors across reset.
        self.peak_kv_blocks_used = 0
        self.peak_kv_utilization = 0.0
        self._kv_util_sum = 0.0
        self._kv_util_count = 0
        self._had_alloc_failure_this_iteration = False

    def begin_iteration(self) -> None:
        """Call once at the start of each scheduler.schedule()."""
        self._had_alloc_failure_this_iteration = False

    def examine_waiting_candidate(self) -> None:
        self.waiting_candidates_examined += 1

    def observe_waiting_age(self, age: int) -> None:
        self.max_waiting_age_steps = max(self.max_waiting_age_steps, age)

    def record_admission_age(self, age: int, *, aging_promoted: bool) -> None:
        self._admission_age_sum += age
        self._admission_age_count += 1
        if aging_promoted:
            self.aging_promotions += 1

    def record_allocation_failure(self, *, hol_skipped: bool) -> None:
        """Record a waiting-admission can_allocate(seq) == -1 event.

        Args:
            hol_skipped: True if the scheduler skipped this request and continued
                scanning (parity). False if it stopped at HOL (real).
        """
        self.allocation_failures += 1
        self.allocation_failed_candidates += 1
        if not self._had_alloc_failure_this_iteration:
            self._had_alloc_failure_this_iteration = True
            self.allocation_failure_steps += 1
        if hol_skipped:
            self.hol_skipped_requests += 1

    def observe_mlfq_queue_sizes(self, q0: int, q1: int, q2: int) -> None:
        self.mlfq_max_q0_size = max(self.mlfq_max_q0_size, q0)
        self.mlfq_max_q1_size = max(self.mlfq_max_q1_size, q1)
        self.mlfq_max_q2_size = max(self.mlfq_max_q2_size, q2)

    def record_mlfq_service(self, level: int, tokens: int) -> None:
        if tokens <= 0:
            return
        if level == 0:
            self.mlfq_q0_service_tokens += tokens
        elif level == 1:
            self.mlfq_q1_service_tokens += tokens
        else:
            self.mlfq_q2_service_tokens += tokens

    def record_mlfq_demotion(self, new_level: int) -> None:
        if new_level == 1:
            self.mlfq_q1_demotions += 1
        elif new_level == 2:
            self.mlfq_q2_demotions += 1

    def record_completion_level(self, level: int) -> None:
        level = max(0, min(2, level))
        self._completion_level_sum += level
        self._completion_level_count += 1
        if level == 0:
            self.completed_in_q0 += 1
        elif level == 1:
            self.completed_in_q1 += 1
        else:
            self.completed_in_q2 += 1

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

    @property
    def mean_age_at_admission(self) -> float:
        if self._admission_age_count == 0:
            return 0.0
        return self._admission_age_sum / self._admission_age_count

    @property
    def mean_completion_level(self) -> float:
        if self._completion_level_count == 0:
            return 0.0
        return self._completion_level_sum / self._completion_level_count

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
            "allocation_failure_steps": self.allocation_failure_steps,
            "allocation_failed_candidates": self.allocation_failed_candidates,
            "hol_skipped_requests": self.hol_skipped_requests,
            "waiting_candidates_examined": self.waiting_candidates_examined,
            "waiting_reorders": self.waiting_reorders,
            "aging_promotions": self.aging_promotions,
            "max_waiting_age_steps": self.max_waiting_age_steps,
            "mean_age_at_admission": self.mean_age_at_admission,
            "scheduler_policy": self.scheduler_policy,
            "scheduler_aging_threshold": self.scheduler_aging_threshold,
            "mlfq_q0_quantum": self.mlfq_q0_quantum,
            "mlfq_q1_quantum": self.mlfq_q1_quantum,
            "mlfq_boost_interval": self.mlfq_boost_interval,
            "mlfq_q0_service_tokens": self.mlfq_q0_service_tokens,
            "mlfq_q1_service_tokens": self.mlfq_q1_service_tokens,
            "mlfq_q2_service_tokens": self.mlfq_q2_service_tokens,
            "mlfq_q0_admissions": self.mlfq_q0_admissions,
            "mlfq_q1_demotions": self.mlfq_q1_demotions,
            "mlfq_q2_demotions": self.mlfq_q2_demotions,
            "mlfq_priority_boosts": self.mlfq_priority_boosts,
            "mlfq_requests_boosted": self.mlfq_requests_boosted,
            "mlfq_max_q0_size": self.mlfq_max_q0_size,
            "mlfq_max_q1_size": self.mlfq_max_q1_size,
            "mlfq_max_q2_size": self.mlfq_max_q2_size,
            "completed_in_q0": self.completed_in_q0,
            "completed_in_q1": self.completed_in_q1,
            "completed_in_q2": self.completed_in_q2,
            "mean_completion_level": self.mean_completion_level,
            "allocation_candidate_failure_rate": self.allocation_candidate_failure_rate,
            "peak_KV_blocks_used": self.peak_kv_blocks_used,
            "average_KV_utilization": self.average_kv_utilization,
            "peak_KV_utilization": self.peak_kv_utilization,
        }
