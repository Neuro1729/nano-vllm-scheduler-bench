#!/usr/bin/env python3
"""Compare two scheduler benchmark JSON result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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


def _change(real: Any, dev: Any, lower_is_better: bool) -> str:
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
    # Annotate direction qualitatively without a single score.
    if abs(pct) < 0.05:
        tip = "same"
    elif lower_is_better:
        tip = "better" if pct < 0 else "worse"
    else:
        tip = "better" if pct > 0 else "worse"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% ({tip})"


def _print_table(title: str, rows: list[tuple[str, Any, Any, bool]]) -> None:
    print()
    print(title)
    print(f"{'Metric':<28} {'real':>14} {'dev':>14} {'change':>22}")
    print("-" * 80)
    for name, real, dev, lower_is_better in rows:
        print(
            f"{name:<28} {_fmt(real):>14} {_fmt(dev):>14} "
            f"{_change(real, dev, lower_is_better):>22}"
        )


def _global_rows(real: dict, dev: dict) -> list[tuple[str, Any, Any, bool]]:
    pairs = [
        ("Output tok/s", "throughput.output_token_throughput_tok_s", False),
        ("Prompt tok/s", "throughput.prompt_token_throughput_tok_s", False),
        ("Total tok/s", "throughput.total_token_throughput_tok_s", False),
        ("Request/s", "throughput.request_throughput_req_s", False),
        ("Wall clock s", "throughput.wall_clock_time_s", True),
        ("Mean TTFT ms", "latency.ttft_ms.mean", True),
        ("P50 TTFT ms", "latency.ttft_ms.p50", True),
        ("P95 TTFT ms", "latency.ttft_ms.p95", True),
        ("P99 TTFT ms", "latency.ttft_ms.p99", True),
        ("Mean TPOT ms", "latency.tpot_ms.mean", True),
        ("P95 TPOT ms", "latency.tpot_ms.p95", True),
        ("P99 TPOT ms", "latency.tpot_ms.p99", True),
        ("Mean E2E ms", "latency.e2e_latency_ms.mean", True),
        ("P95 E2E ms", "latency.e2e_latency_ms.p95", True),
        ("P99 E2E ms", "latency.e2e_latency_ms.p99", True),
        ("Preemptions", "scheduler.preemption_count", True),
        ("Preempted requests", "scheduler.preempted_requests", True),
        ("Recomputed tokens", "scheduler.recomputed_tokens", True),
        ("Alloc failures", "scheduler.allocation_failures", True),
        ("Peak KV util", "scheduler.peak_KV_utilization", True),
        ("Avg KV util", "scheduler.average_KV_utilization", False),
    ]
    rows = []
    for name, path, lower in pairs:
        rows.append((name, _get(real, path), _get(dev, path), lower))
    return rows


def _class_rows(real: dict, dev: dict, cls: str) -> list[tuple[str, Any, Any, bool]]:
    base = f"classes.{cls}"
    pairs = [
        ("Mean TTFT ms", f"{base}.latency.ttft_ms.mean", True),
        ("P50 TTFT ms", f"{base}.latency.ttft_ms.p50", True),
        ("P95 TTFT ms", f"{base}.latency.ttft_ms.p95", True),
        ("P99 TTFT ms", f"{base}.latency.ttft_ms.p99", True),
        ("Mean TPOT ms", f"{base}.latency.tpot_ms.mean", True),
        ("P99 TPOT ms", f"{base}.latency.tpot_ms.p99", True),
        ("Mean E2E ms", f"{base}.latency.e2e_latency_ms.mean", True),
        ("P95 E2E ms", f"{base}.latency.e2e_latency_ms.p95", True),
        ("Output tok/s", f"{base}.throughput.output_token_throughput_tok_s", False),
        ("Preemptions", f"{base}.scheduler.preemption_count", True),
        ("Recomputed tokens", f"{base}.scheduler.recomputed_tokens", True),
    ]
    return [(n, _get(real, p), _get(dev, p), low) for n, p, low in pairs]


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

    _print_table("Global metrics", _global_rows(real, dev))

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
