"""Instrumentation semantics for allocation failure metrics (no GPU / triton)."""

from nanovllm.engine.scheduler_metrics import SchedulerMetrics


def test_metrics_helpers_step_vs_candidates():
    m = SchedulerMetrics(num_kv_blocks=4)
    m.begin_iteration()
    m.examine_waiting_candidate()
    m.record_allocation_failure(hol_skipped=False)
    m.examine_waiting_candidate()
    m.record_allocation_failure(hol_skipped=False)
    # Same iteration: steps stay 1.
    assert m.allocation_failure_steps == 1
    assert m.allocation_failed_candidates == 2
    assert m.hol_skipped_requests == 0
    assert m.allocation_failures == 2
    assert m.waiting_candidates_examined == 2

    m.begin_iteration()
    m.record_allocation_failure(hol_skipped=True)
    assert m.allocation_failure_steps == 2
    assert m.hol_skipped_requests == 1
    d = m.to_dict()
    assert d["allocation_failure_steps"] == 2
    assert d["allocation_failed_candidates"] == 3
    assert "allocation_candidate_failure_rate" in d


def test_real_style_hol_stop_does_not_count_skip():
    """real records hol_skipped=False; steps == candidates when ≤1 failure/step."""
    m = SchedulerMetrics(num_kv_blocks=1)
    for _ in range(3):
        m.begin_iteration()
        m.examine_waiting_candidate()
        m.record_allocation_failure(hol_skipped=False)
        # Second waiter is never examined under real HOL break.
    assert m.allocation_failure_steps == 3
    assert m.allocation_failed_candidates == 3
    assert m.hol_skipped_requests == 0
    assert m.waiting_candidates_examined == 3


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
    assert scan["Alloc failures (legacy)"] == "info"
    tip = _change(10, 500, "info")
    assert "(info)" in tip
    assert "better" not in tip and "worse" not in tip
