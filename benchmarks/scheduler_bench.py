#!/usr/bin/env python3
"""Scheduler-focused benchmark runner for Nano-vLLM (branch-neutral)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

# Allow `python benchmarks/scheduler_bench.py` from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.metrics import build_result
from benchmarks.workloads import describe_scaled_ranges, generate_workload
from nanovllm import LLM, SamplingParams


# Nano-vLLM rejects temperature=0; use a tiny positive value with ignore_eos for fixed lengths.
NEAR_GREEDY_TEMPERATURE = 1e-5


def _git_metadata() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
    }


def _gpu_metadata() -> dict[str, object]:
    info: dict[str, object] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": None,
        "gpu_count_visible": 0,
        "cuda_device_index": 0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count_visible"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_bytes"] = int(props.total_memory)
            info["gpu_multi_processor_count"] = int(props.multi_processor_count)
    except Exception as exc:  # pragma: no cover - best-effort metadata
        info["gpu_error"] = str(exc)
    return info


def _output_path_for_run(base: Path, run_idx: int, seed: int, runs: int) -> Path:
    if runs == 1:
        return base
    stem = base.stem
    return base.with_name(f"{stem}_run{run_idx}_seed{seed}{base.suffix}")


def _collect_finished_requests(llm: LLM) -> list[dict]:
    # Finished sequences are removed from scheduler queues; track via side channel.
    return list(getattr(llm, "_bench_finished_requests"))


def _attach_finish_collector(llm: LLM) -> None:
    """Wrap scheduler.postprocess to retain finished request records in memory."""
    llm._bench_finished_requests = []
    original = llm.scheduler.postprocess

    def postprocess(seqs, token_ids, is_prefill=None):
        # Compat: parity dropped the is_prefill arg; ignore if provided.
        if is_prefill is None:
            original(seqs, token_ids)
        else:
            try:
                original(seqs, token_ids, is_prefill)
            except TypeError:
                original(seqs, token_ids)
        for seq in seqs:
            if not seq.is_finished:
                continue
            llm._bench_finished_requests.append(
                {
                    "request_id": seq.client_request_id if seq.client_request_id >= 0 else seq.seq_id,
                    "seq_id": seq.seq_id,
                    "workload_class": seq.workload_class,
                    "prompt_tokens": seq.num_prompt_tokens,
                    "requested_output_tokens": seq.requested_output_tokens,
                    "output_tokens": seq.num_completion_tokens,
                    "arrival_time": seq.arrival_time,
                    "first_scheduler_admission_time": seq.first_admission_time,
                    "first_token_time": seq.first_token_time,
                    "finish_time": seq.finish_time,
                    "num_preemptions": seq.num_preemptions,
                    "num_scheduler_steps": seq.num_scheduler_steps,
                    "num_recomputed_tokens": seq.num_recomputed_tokens,
                }
            )

    llm.scheduler.postprocess = postprocess


def _reset_between_runs(llm: LLM) -> None:
    llm.scheduler.metrics.reset()
    llm.scheduler.block_manager.clear_prefix_cache()
    llm._bench_finished_requests = []
    assert llm.scheduler.is_finished(), "scheduler not idle between runs"


def run_once(
    llm: LLM,
    workload: str,
    num_requests: int,
    seed: int,
    run_id: int,
    temperature: float,
) -> dict:
    max_model_len = llm.config.max_model_len
    vocab_size = getattr(llm.config.hf_config, "vocab_size", 32000)

    # Generate workload BEFORE timed inference.
    specs = generate_workload(
        name=workload,
        num_requests=num_requests,
        seed=seed,
        max_model_len=max_model_len,
        vocab_size=vocab_size,
        run_id=run_id,
    )

    sampling_params = [
        SamplingParams(
            temperature=temperature,
            max_tokens=spec.max_tokens,
            ignore_eos=True,
        )
        for spec in specs
    ]

    _reset_between_runs(llm)

    arrival_time = perf_counter()
    for spec, sp in zip(specs, sampling_params):
        llm.add_request(
            spec.prompt_token_ids,
            sp,
            workload_class=spec.workload_class,
            arrival_time=arrival_time,
            client_request_id=spec.request_id,
        )

    # Timed region: inference only (model already loaded/warmed).
    t0 = perf_counter()
    while not llm.is_finished():
        llm.step()
    # Ensure GPU work is complete before stopping the clock.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    wall = perf_counter() - t0

    requests = _collect_finished_requests(llm)
    if len(requests) != num_requests:
        raise RuntimeError(f"expected {num_requests} completed requests, got {len(requests)}")

    git = _git_metadata()
    gpu = _gpu_metadata()
    metadata = {
        **git,
        **gpu,
        "model": str(llm.config.model),
        "seed": seed,
        "run_id": run_id,
        "workload": workload,
        "num_requests": num_requests,
        "temperature": temperature,
        "ignore_eos": True,
        "prefix_cache_policy": "cleared_between_runs; distinct synthetic prompts",
        "scheduler_policy": getattr(llm.config, "scheduler_policy", "fcfs"),
        "scheduler_aging_threshold": getattr(llm.config, "scheduler_aging_threshold", 128),
        "mlfq_q0_quantum": getattr(llm.config, "mlfq_q0_quantum", 256),
        "mlfq_q1_quantum": getattr(llm.config, "mlfq_q1_quantum", 1024),
        "mlfq_boost_interval": getattr(llm.config, "mlfq_boost_interval", 256),
    }
    config = {
        "max_model_len": llm.config.max_model_len,
        "max_num_batched_tokens": llm.config.max_num_batched_tokens,
        "max_num_seqs": llm.config.max_num_seqs,
        "gpu_memory_utilization": llm.config.gpu_memory_utilization,
        "tensor_parallel_size": llm.config.tensor_parallel_size,
        "enforce_eager": llm.config.enforce_eager,
        "kvcache_block_size": llm.config.kvcache_block_size,
        "num_kvcache_blocks": llm.config.num_kvcache_blocks,
        "scheduler_policy": getattr(llm.config, "scheduler_policy", "fcfs"),
        "scheduler_aging_threshold": getattr(llm.config, "scheduler_aging_threshold", 128),
        "mlfq_q0_quantum": getattr(llm.config, "mlfq_q0_quantum", 256),
        "mlfq_q1_quantum": getattr(llm.config, "mlfq_q1_quantum", 1024),
        "mlfq_boost_interval": getattr(llm.config, "mlfq_boost_interval", 256),
        "scaled_workload_ranges": describe_scaled_ranges(llm.config.max_model_len),
    }
    return build_result(
        metadata=metadata,
        config=config,
        requests=requests,
        scheduler_metrics=llm.scheduler.metrics.to_dict(),
        wall_clock_time_s=wall,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nano-vLLM scheduler benchmark")
    p.add_argument("--workload", required=True, choices=["interactive", "rag", "long_generation", "mixed"])
    p.add_argument("--model", required=True, help="Local Hugging Face model directory")
    p.add_argument("--num-requests", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=1, help="Repeated trials; seeds are seed..seed+runs-1")
    p.add_argument("--output", type=Path, required=True, help="JSON output path (run suffix added if --runs>1)")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--temperature", type=float, default=NEAR_GREEDY_TEMPERATURE)
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--experiment-label", type=str, default="", help="Optional label for crossover experiments")
    p.add_argument(
        "--scheduler-policy",
        type=str,
        default="fcfs",
        choices=["fcfs", "sjf", "sjf_aging", "mlfq"],
        help="Scheduler policy (dev experiments; default fcfs matches parity)",
    )
    p.add_argument(
        "--aging-threshold",
        type=int,
        default=128,
        help="For sjf_aging: waiting-age steps before starvation promotion (default 128)",
    )
    p.add_argument("--mlfq-q0-quantum", type=int, default=256, help="MLFQ Q0 token quantum")
    p.add_argument("--mlfq-q1-quantum", type=int, default=1024, help="MLFQ Q1 token quantum")
    p.add_argument(
        "--mlfq-boost-interval",
        type=int,
        default=256,
        help="MLFQ priority boost every N scheduler iterations (0 disables)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.temperature <= 1e-10:
        raise SystemExit(
            "Nano-vLLM rejects temperature<=1e-10; use a tiny positive value "
            f"(default {NEAR_GREEDY_TEMPERATURE})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.aging_threshold < 1:
        raise SystemExit("--aging-threshold must be >= 1")
    if args.mlfq_q0_quantum < 1 or args.mlfq_q1_quantum < 1:
        raise SystemExit("--mlfq-q0-quantum and --mlfq-q1-quantum must be >= 1")
    if args.mlfq_boost_interval < 0:
        raise SystemExit("--mlfq-boost-interval must be >= 0")

    llm_kwargs = {
        "enforce_eager": args.enforce_eager,
        "scheduler_policy": args.scheduler_policy,
        "scheduler_aging_threshold": args.aging_threshold,
        "mlfq_q0_quantum": args.mlfq_q0_quantum,
        "mlfq_q1_quantum": args.mlfq_q1_quantum,
        "mlfq_boost_interval": args.mlfq_boost_interval,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    print("Loading model (not included in measured time)...")
    llm = LLM(args.model, **llm_kwargs)
    _attach_finish_collector(llm)

    print("Warmup (not included in measured time)...")
    warm_len = min(args.warmup_tokens, max(1, llm.config.max_model_len // 4))
    llm.generate(
        [[1, 2, 3, 4]],
        SamplingParams(temperature=args.temperature, max_tokens=warm_len, ignore_eos=True),
        use_tqdm=False,
    )
    llm.scheduler.metrics.reset()
    llm.scheduler.block_manager.clear_prefix_cache()

    written: list[Path] = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        print(f"Run {run_idx + 1}/{args.runs} workload={args.workload} seed={seed}")
        result = run_once(
            llm=llm,
            workload=args.workload,
            num_requests=args.num_requests,
            seed=seed,
            run_id=run_idx,
            temperature=args.temperature,
        )
        if args.experiment_label:
            result["metadata"]["experiment_label"] = args.experiment_label
        out_path = _output_path_for_run(args.output, run_idx, seed, args.runs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        written.append(out_path)
        thr = result["throughput"]
        lat = result["latency"]["ttft_ms"]
        print(
            f"  wall={thr['wall_clock_time_s']:.3f}s "
            f"out_tok/s={thr['output_token_throughput_tok_s']:.1f} "
            f"ttft_p50={lat['p50']:.2f}ms ttft_p99={lat['p99']:.2f}ms "
            f"preempts={result['scheduler']['preemption_count']}"
        )

    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
