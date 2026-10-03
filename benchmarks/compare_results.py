#!/usr/bin/env python3
"""Compare two scheduler benchmark JSON result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal


Judgment = Literal["lower", "higher", "info"]


def _get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def _change(real: Any, dev: Any, judgment: Judgment) -> str:
    if real is None or dev is None:
        return "n/a"
    try:
        real_f = float(real)
        dev_f = float(dev)
    except (TypeError, ValueError):
        return "n/a"
    if real_f == 0:
        return "n/a"
    pct = (dev_f - real_f) / abs(real_f) * 100.0
    if judgment == "info":
        tip = "info"
    elif abs(pct) < 0.05:
        tip = "same"
    elif judgment == "lower":
        tip = "better" if pct < 0 else "worse"
    else:
        tip = "better" if pct > 0 else "worse"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% ({tip})"


def _print_table(title: str, rows: list[tuple[str, Any, Any, Judgment]], note: str = "") -> None:
    print()
    print(title)
    if note:
        print(f"  {note}")
    print(f"{'Metric':<32} {'real':>14} {'dev':>14} {'change':>22}")
    print("-" * 84)
    for name, real, dev, judgment in rows:
        print(
            f"{name:<32} {_fmt(real):>14} {_fmt(dev):>14} "
            f"{_change(real, dev, judgment):>22}"
        )


def _rows(real: dict, dev: dict, pairs: list[tuple[str, str, Judgment]]) -> list[tuple[str, Any, Any, Judgment]]:
    return [(name, _get(real, path), _get(dev, path), judgment) for name, path, judgment in pairs]


def _global_pressure_rows(real: dict, dev: dict) -> list[tuple[str, Any, Any, Judgment]]:
    """Cross-branch-comparable KV / preemption pressure metrics."""
    return _rows(
        real,
        dev,
        [
            ("Alloc failure steps", "scheduler.allocation_failure_steps", "lower"),
            ("Preemptions", "scheduler.preemption_count", "lower"),
            ("Preempted requests", "scheduler.preempted_requests", "lower"),
            ("Recomputed tokens", "scheduler.recomputed_tokens", "lower"),
            ("Peak KV util", "scheduler.peak_KV_utilization", "lower"),
            ("Avg KV util", "scheduler.average_KV_utilization", "info"),
            ("Peak KV blocks used", "scheduler.peak_KV_blocks_used", "info"),
        ],
    )


def _global_alloc_scan_rows(real: dict, dev: dict) -> list[tuple[str, Any, Any, Judgment]]:
    """Policy-dependent allocation scan counts; do not treat as better/worse."""
    return _rows(
        real,
        dev,
        [
            ("Alloc failed candidates", "scheduler.allocation_failed_candidates", "info"),
            ("HOL skipped requests", "scheduler.hol_skipped_requests", "info"),
            ("Waiting candidates examined", "scheduler.waiting_candidates_examined", "info"),
            ("Alloc candidate fail rate", "scheduler.allocation_candidate_failure_rate", "info"),
            ("Alloc failures (legacy)", "scheduler.allocation_failures", "info"),
        ],
    )


def _global_perf_rows(real: dict, dev: dict) -> list[tuple[str, Any, Any, Judgment]]:
    return _rows(
        real,
        dev,
        [
            ("Output tok/s", "throughput.output_token_throughput_tok_s", "higher"),
            ("Prompt tok/s", "throughput.prompt_token_throughput_tok_s", "higher"),
            ("Total tok/s", "throughput.total_token_throughput_tok_s", "higher"),
            ("Request/s", "throughput.request_throughput_req_s", "higher"),
            ("Wall clock s", "throughput.wall_clock_time_s", "lower"),
            ("Mean TTFT ms", "latency.ttft_ms.mean", "lower"),
            ("P50 TTFT ms", "latency.ttft_ms.p50", "lower"),
            ("P95 TTFT ms", "latency.ttft_ms.p95", "lower"),
            ("P99 TTFT ms", "latency.ttft_ms.p99", "lower"),
            ("Mean TPOT ms", "latency.tpot_ms.mean", "lower"),
            ("P95 TPOT ms", "latency.tpot_ms.p95", "lower"),
            ("P99 TPOT ms", "latency.tpot_ms.p99", "lower"),
            ("Mean E2E ms", "latency.e2e_latency_ms.mean", "lower"),
            ("P95 E2E ms", "latency.e2e_latency_ms.p95", "lower"),
            ("P99 E2E ms", "latency.e2e_latency_ms.p99", "lower"),
            ("Mixed iterations", "scheduler.mixed_iterations", "info"),
            ("Prefill iterations", "scheduler.prefill_iterations", "info"),
            ("Decode iterations", "scheduler.decode_iterations", "info"),
            ("Sched prefill toks", "scheduler.scheduled_prefill_tokens", "info"),
            ("Sched decode toks", "scheduler.scheduled_decode_tokens", "info"),
            ("Chunked prefills", "scheduler.chunked_prefill_count", "info"),
        ],
    )


def _class_rows(real: dict, dev: dict, cls: str) -> list[tuple[str, Any, Any, Judgment]]:
    base = f"classes.{cls}"
    return _rows(
        real,
        dev,
        [
            ("Mean TTFT ms", f"{base}.latency.ttft_ms.mean", "lower"),
            ("P50 TTFT ms", f"{base}.latency.ttft_ms.p50", "lower"),
            ("P95 TTFT ms", f"{base}.latency.ttft_ms.p95", "lower"),
            ("P99 TTFT ms", f"{base}.latency.ttft_ms.p99", "lower"),
            ("Mean TPOT ms", f"{base}.latency.tpot_ms.mean", "lower"),
            ("P99 TPOT ms", f"{base}.latency.tpot_ms.p99", "lower"),
            ("Mean E2E ms", f"{base}.latency.e2e_latency_ms.mean", "lower"),
            ("P95 E2E ms", f"{base}.latency.e2e_latency_ms.p95", "lower"),
            ("Output tok/s", f"{base}.throughput.output_token_throughput_tok_s", "higher"),
            ("Preemptions", f"{base}.scheduler.preemption_count", "lower"),
            ("Recomputed tokens", f"{base}.scheduler.recomputed_tokens", "lower"),
        ],
    )


def compare(real_path: Path, dev_path: Path) -> int:
    real = json.loads(real_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))

    print("Comparison metadata")
    print(f"  real: branch={_get(real, 'metadata.branch')} commit={_get(real, 'metadata.git_commit')}")
    print(f"        gpu={_get(real, 'metadata.gpu_name')} CUDA_VISIBLE_DEVICES={_get(real, 'metadata.cuda_visible_devices')}")
    print(f"        workload={_get(real, 'metadata.workload')} seed={_get(real, 'metadata.seed')}")
    print(f"  dev:  branch={_get(dev, 'metadata.branch')} commit={_get(dev, 'metadata.git_commit')}")
    print(f"        gpu={_get(dev, 'metadata.gpu_name')} CUDA_VISIBLE_DEVICES={_get(dev, 'metadata.cuda_visible_devices')}")
    print(f"        workload={_get(dev, 'metadata.workload')} seed={_get(dev, 'metadata.seed')}")

    if _get(real, "metadata.seed") != _get(dev, "metadata.seed"):
        print("WARNING: seeds differ; comparison may be unfair.")
    if _get(real, "metadata.workload") != _get(dev, "metadata.workload"):
        print("WARNING: workloads differ; comparison may be unfair.")
    if _get(real, "metadata.gpu_name") != _get(dev, "metadata.gpu_name"):
        print("WARNING: GPU models differ; run crossover swap before drawing conclusions.")

    _print_table(
        "KV / preemption pressure (cross-branch comparable)",
        _global_pressure_rows(real, dev),
        note="Prefer these for real vs parity pressure comparisons.",
    )
    _print_table(
        "Allocation scan diagnostics (policy-dependent; not better/worse)",
        _global_alloc_scan_rows(real, dev),
        note=(
            "failed_candidates/hol_skipped reflect admit scan policy; "
            "parity HOL-skip can legitimately inflate them vs real."
        ),
    )
    _print_table("Throughput / latency / batch mix", _global_perf_rows(real, dev))

    classes = sorted(set(real.get("classes", {})) | set(dev.get("classes", {})))
    for cls in classes:
        _print_table(f"Class: {cls}", _class_rows(real, dev, cls))

    print()
    print("No single overall score is assigned; inspect trade-offs above.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare Nano-vLLM scheduler benchmark results")
    p.add_argument("real_json", type=Path, help="Baseline/real result JSON")
    p.add_argument("dev_json", type=Path, help="Experimental/dev result JSON")
    args = p.parse_args(argv)
    return compare(args.real_json, args.dev_json)


if __name__ == "__main__":
    raise SystemExit(main())
