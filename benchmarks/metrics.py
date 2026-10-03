"""Aggregate per-request and global benchmark metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "count": 0}
    ordered = sorted(values)
    return {
        "mean": float(mean(ordered)),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": float(ordered[-1]),
        "count": len(ordered),
    }


def request_latency_fields(req: dict[str, Any]) -> dict[str, float]:
    arrival = req["arrival_time"]
    admit = req["first_scheduler_admission_time"]
    first = req["first_token_time"]
    finish = req["finish_time"]
    output_tokens = max(int(req["output_tokens"]), 0)

    queue_latency_ms = max(0.0, (admit - arrival) * 1000.0) if admit > 0 else 0.0
    ttft_ms = max(0.0, (first - arrival) * 1000.0) if first > 0 else 0.0
    e2e_ms = max(0.0, (finish - arrival) * 1000.0) if finish > 0 else 0.0
    if first > 0 and finish >= first:
        tpot_ms = ((finish - first) * 1000.0) / max(output_tokens - 1, 1)
    else:
        tpot_ms = 0.0
    return {
        "queue_latency_ms": queue_latency_ms,
        "TTFT_ms": ttft_ms,
        "E2E_latency_ms": e2e_ms,
        "TPOT_ms": tpot_ms,
    }


def enrich_requests(raw_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for req in raw_requests:
        item = dict(req)
        item.update(request_latency_fields(item))
        enriched.append(item)
    return enriched


def latency_block(requests: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        "queue_latency_ms": summarize([r["queue_latency_ms"] for r in requests]),
        "ttft_ms": summarize([r["TTFT_ms"] for r in requests]),
        "e2e_latency_ms": summarize([r["E2E_latency_ms"] for r in requests]),
        "tpot_ms": summarize([r["TPOT_ms"] for r in requests]),
    }


def throughput_block(
    requests: list[dict[str, Any]],
    wall_clock_time_s: float,
) -> dict[str, float]:
    prompt_tokens = sum(int(r["prompt_tokens"]) for r in requests)
    output_tokens = sum(int(r["output_tokens"]) for r in requests)
    total_tokens = prompt_tokens + output_tokens
    wall = max(wall_clock_time_s, 1e-12)
    return {
        "wall_clock_time_s": wall_clock_time_s,
        "completed_requests": len(requests),
        "request_throughput_req_s": len(requests) / wall,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_token_throughput_tok_s": prompt_tokens / wall,
        "output_token_throughput_tok_s": output_tokens / wall,
        "total_token_throughput_tok_s": total_tokens / wall,
    }


def class_blocks(requests: list[dict[str, Any]], wall_clock_time_s: float) -> dict[str, dict]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for req in requests:
        by_class.setdefault(req["workload_class"], []).append(req)
    out = {}
    for cls, cls_reqs in sorted(by_class.items()):
        out[cls] = {
            "throughput": throughput_block(cls_reqs, wall_clock_time_s),
            "latency": latency_block(cls_reqs),
            "scheduler": {
                "preemption_count": sum(int(r["num_preemptions"]) for r in cls_reqs),
                "recomputed_tokens": sum(int(r["num_recomputed_tokens"]) for r in cls_reqs),
                "completed_requests": len(cls_reqs),
            },
        }
    return out


def build_result(
    metadata: dict[str, Any],
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    scheduler_metrics: dict[str, Any],
    wall_clock_time_s: float,
) -> dict[str, Any]:
    enriched = enrich_requests(requests)
    return {
        "metadata": metadata,
        "config": config,
        "throughput": throughput_block(enriched, wall_clock_time_s),
        "latency": latency_block(enriched),
        "scheduler": scheduler_metrics,
        "classes": class_blocks(enriched, wall_clock_time_s),
        "requests": enriched,
    }
