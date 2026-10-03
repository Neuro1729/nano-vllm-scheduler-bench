"""Instrumentation semantics for allocation failure metrics (no scheduling changes)."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(num_blocks: int = 2, max_batched_tokens: int = 32, max_seqs: int = 4) -> Scheduler:
    Sequence.block_size = 256
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = max_seqs
    sched.max_num_batched_tokens = max_batched_tokens
    sched.eos = -1
    sched.block_size = 256
    sched.block_manager = BlockManager(num_blocks, 256)
    sched.waiting = deque()
    sched.running = deque()
    sched.scheduler_policy = "fcfs"
    sched.metrics = SchedulerMetrics(num_kv_blocks=num_blocks)
    sched.metrics.scheduler_policy = "fcfs"
    return sched


def _seq(n: int) -> Sequence:
    return Sequence(list(range(n)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))


def test_metrics_helpers_step_vs_candidates():
    m = SchedulerMetrics(num_kv_blocks=4)
    m.begin_iteration()
    m.examine_waiting_candidate()
    m.record_allocation_failure(hol_skipped=True)
    m.examine_waiting_candidate()
    m.record_allocation_failure(hol_skipped=True)
    assert m.allocation_failure_steps == 1
    assert m.allocation_failed_candidates == 2
    assert m.hol_skipped_requests == 2
    assert m.allocation_failures == 2  # legacy mirrors candidates here
    assert m.waiting_candidates_examined == 2
    assert m.allocation_candidate_failure_rate == 1.0

    m.begin_iteration()
    m.examine_waiting_candidate()
    m.record_allocation_failure(hol_skipped=False)
    assert m.allocation_failure_steps == 2
    assert m.allocation_failed_candidates == 3
    assert m.hol_skipped_requests == 2  # real-style stop does not skip


def test_parity_hol_skip_inflates_candidates_not_steps():
    sched = _make_scheduler(num_blocks=1)
    hog = _seq(4)
    sched.block_manager.allocate(hog, 0)
    assert len(sched.block_manager.free_block_ids) == 0

    blocked_a = _seq(4)
    blocked_b = _seq(4)
    ok = _seq(4)
    ok.block_table = list(hog.block_table)
    ok.num_cached_tokens = 0
    ok.status = SequenceStatus.WAITING

    sched.add(blocked_a)
    sched.add(blocked_b)
    sched.waiting.append(ok)

    out = sched.schedule()
    assert ok in out.seqs
    # Two blocked waiting requests failed allocate; one step.
    assert sched.metrics.allocation_failed_candidates == 2
    assert sched.metrics.allocation_failure_steps == 1
    assert sched.metrics.hol_skipped_requests == 2
    assert sched.metrics.waiting_candidates_examined >= 3
    assert sched.metrics.allocation_failures == 2
    d = sched.metrics.to_dict()
    assert d["allocation_failure_steps"] == 1
    assert d["allocation_failed_candidates"] == 2
    assert d["hol_skipped_requests"] == 2
    assert "allocation_candidate_failure_rate" in d


def test_compare_results_marks_failed_candidates_as_info():
    from benchmarks.compare_results import _change, _global_alloc_scan_rows, _global_pressure_rows

    real = {
        "scheduler": {
            "allocation_failure_steps": 10,
            "allocation_failed_candidates": 10,
            "hol_skipped_requests": 0,
            "waiting_candidates_examined": 100,
            "allocation_candidate_failure_rate": 0.1,
            "allocation_failures": 10,
            "preemption_count": 5,
            "preempted_requests": 3,
            "recomputed_tokens": 100,
            "peak_KV_utilization": 0.9,
            "average_KV_utilization": 0.5,
            "peak_KV_blocks_used": 40,
        }
    }
    parity = {
        "scheduler": {
            "allocation_failure_steps": 8,
            "allocation_failed_candidates": 500,
            "hol_skipped_requests": 490,
            "waiting_candidates_examined": 600,
            "allocation_candidate_failure_rate": 500 / 600,
            "allocation_failures": 500,
            "preemption_count": 4,
            "preempted_requests": 2,
            "recomputed_tokens": 80,
            "peak_KV_utilization": 0.8,
            "average_KV_utilization": 0.4,
            "peak_KV_blocks_used": 35,
        }
    }
    pressure = {name: j for name, _, _, j in _global_pressure_rows(real, parity)}
    assert pressure["Alloc failure steps"] == "lower"
    scan = {name: j for name, _, _, j in _global_alloc_scan_rows(real, parity)}
    assert scan["Alloc failed candidates"] == "info"
    assert scan["HOL skipped requests"] == "info"
    assert scan["Alloc failures (legacy)"] == "info"
    # Info judgment must not say better/worse even if candidates explode.
    tip = _change(10, 500, "info")
    assert "(info)" in tip
    assert "better" not in tip and "worse" not in tip
