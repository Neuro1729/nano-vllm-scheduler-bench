# Five-seed FCFS vs SJF experiment (historical)

Summarized GPU results for the first FCFS vs SJF comparison under the mixed workload.
This file records **provided numerical summaries only**. Raw per-run JSON remains on the GPU machine until copied separately.

## Code snapshot (do not mislabel)

| Role | Commit | Notes |
|------|--------|-------|
| parity baseline | `1d0eea4` | instrumentation cleanup; modern baseline |
| experimental SJF (`dev`) | `aa599e5` | `experiment: add shortest-job-first admission policy` |

These runs were generated **before** `sjf_aging` (`b7dc177` and later). Do not attribute this data to aging-enabled builds.

## Hardware / configuration

```text
GPU: NVIDIA RTX A5000
GPU selected with: CUDA_VISIBLE_DEVICES=1

Model: /models/Qwen3-0.6B
Workload: mixed
Requests: 256
Runs per seed/policy: 1
max_model_len: 4096

Policies: fcfs, sjf
Seeds: 42, 43, 44, 45, 46
```

### SJF definition used

```text
shortest-known-work-first WAITING admission

score:
  remaining_compute_tokens =
    max(0, num_tokens - num_computed_tokens)

tie-break: lower seq_id / arrival order
```

SJF does **not** predict true future generation length. It is not classic total-runtime SJF.

### Run order (alternating to reduce order bias)

```text
seed 42: FCFS -> SJF
seed 43: SJF  -> FCFS
seed 44: FCFS -> SJF
seed 45: SJF  -> FCFS
seed 46: FCFS -> SJF
```

## Raw per-seed headline results

See also `fcfs_vs_sjf_5seed.csv`.

| seed | policy | wall_s | output_tok_s | ttft_p50_ms | ttft_p99_ms | preemptions |
|------|--------|--------|--------------|-------------|-------------|-------------|
| 42 | fcfs | 56.760 | 2432.7 | 4606.92 | 18419.50 | 71 |
| 42 | sjf | 53.135 | 2598.7 | 1008.47 | 18793.74 | 17 |
| 43 | fcfs | 56.745 | 2435.5 | 2889.62 | 17085.99 | 65 |
| 43 | sjf | 53.711 | 2573.1 | 989.93 | 18026.44 | 19 |
| 44 | fcfs | 59.062 | 2498.5 | 2873.78 | 16258.60 | 64 |
| 44 | sjf | 57.356 | 2572.8 | 1020.92 | 17319.14 | 17 |
| 45 | fcfs | 61.634 | 2421.9 | 3138.28 | 17141.24 | 61 |
| 45 | sjf | 59.978 | 2488.8 | 1025.57 | 21218.19 | 23 |
| 46 | fcfs | 60.228 | 2406.7 | 3301.15 | 17812.68 | 60 |
| 46 | sjf | 58.198 | 2490.6 | 1097.78 | 19508.87 | 17 |

## Five-seed aggregate (descriptive means)

Descriptive averages over five matched seeds. **Not** confidence intervals or significance tests.

| Metric | FCFS | SJF | approximate change |
|--------|------|-----|--------------------|
| Output tok/s | 2439.1 | 2544.8 | +4.3% |
| Wall time | 58.89 s | 56.48 s | -4.1% |
| TTFT p50 | 3362 ms | 1029 ms | -69.4% |
| TTFT p99 | 17344 ms | 18973 ms | +9.4% worse |
| Preemptions | 64.2 | 18.6 | -71.0% |

## Repeated observation: p99 TTFT worse under SJF in every seed

| seed | FCFS p99 (ms) | SJF p99 (ms) | SJF vs FCFS |
|------|---------------|--------------|-------------|
| 42 | 18419.50 | 18793.74 | ≈ +2.0% worse |
| 43 | 17085.99 | 18026.44 | ≈ +5.5% worse |
| 44 | 16258.60 | 17319.14 | ≈ +6.5% worse |
| 45 | 17141.24 | 21218.19 | ≈ +23.8% worse |
| 46 | 17812.68 | 19508.87 | ≈ +9.5% worse |

## Detailed seed-42 findings

### Global FCFS -> SJF

| Metric | FCFS | SJF | change |
|--------|------|-----|--------|
| Output tok/s | 2432.68 | 2598.65 | +6.8% |
| Wall (s) | 56.760 | 53.135 | -6.4% |
| Mean TTFT (ms) | 7404.56 | 2766.24 | -62.6% |
| P50 TTFT (ms) | 4606.92 | 1008.47 | -78.1% |
| P95 TTFT (ms) | 16823.74 | 14122.26 | -16.1% |
| P99 TTFT (ms) | 18419.50 | 18793.74 | +2.0% worse |
| Mean TPOT (ms) | 48.481 | 45.915 | -5.3% |
| P99 TPOT (ms) | 90.735 | 68.667 | -24.3% |
| Mean E2E (ms) | 26127.86 | 22061.60 | -15.6% |
| P99 E2E (ms) | 56423.54 | 52984.53 | -6.1% |

### Scheduler behavior (seed 42)

| Metric | FCFS | SJF |
|--------|------|-----|
| scheduler iterations | 2107 | 1941 |
| mixed iterations | 95 | 57 |
| decode iterations | 2011 | 1883 |
| preemptions | 71 | 17 |
| preempted requests | 49 | 11 |
| recomputed tokens | 49945 | 40659 |
| max running requests | 161 | 220 |
| waiting reorders | 0 | 258 |
| average KV utilization | 0.5992 | 0.6504 |
| peak KV utilization | 1.0 | 1.0 |

`waiting_reorders=258` under SJF indicates WAITING admission order actually changed relative to FCFS.

### Per-class (seed 42)

**Interactive**

| Metric | FCFS | SJF |
|--------|------|-----|
| Mean TTFT (ms) | 7108.58 | 572.31 |
| P50 TTFT (ms) | 4779.47 | 469.08 |
| P95 TTFT (ms) | 15520.71 | 733.60 |
| P99 TTFT (ms) | 16434.33 | 1008.47 |
| Mean E2E (ms) | 16114.98 | 9287.68 |
| Preemptions | 42 | 0 |
| Recomputed tokens | 11505 | 0 |

**Long-generation**

| Metric | FCFS | SJF |
|--------|------|-----|
| Mean TTFT (ms) | 6813.75 | 941.54 |
| P50 TTFT (ms) | 4434.38 | 1008.47 |
| P99 TTFT (ms) | 17672.61 | 1405.98 |
| Mean E2E (ms) | 44110.80 | 41077.97 |
| Preemptions | 16 | 0 |
| Recomputed tokens | 11340 | 0 |

**RAG** (fairness / tail tradeoff visible)

| Metric | FCFS | SJF | note |
|--------|------|-----|------|
| Mean TTFT (ms) | 7716.88 | 8357.12 | +8.3% worse |
| P50 TTFT (ms) | 4434.38 | 8764.75 | +97.7% worse |
| P95 TTFT (ms) | 17867.57 | 18493.82 | +3.5% worse |
| P99 TTFT (ms) | 18508.79 | 19728.18 | +6.6% worse |
| Mean TPOT (ms) | 57.798 | 46.292 | better |
| Mean E2E (ms) | 16151.85 | 15178.97 | better |

**Heavy**

| Metric | FCFS | SJF |
|--------|------|-----|
| Mean TTFT (ms) | 9336.89 | 2163.86 |
| P50 TTFT (ms) | 10297.06 | 2230.76 |
| P99 TTFT (ms) | 18364.81 | 2904.41 |
| Mean E2E (ms) | 46883.23 | 43627.78 |
| Preemptions | 1 | 8 |
| Recomputed tokens | 1079 | 16012 |

## Interpretation (cautious)

Under the tested mixed workload, shortest-known-work-first admission substantially improved median TTFT and throughput and greatly reduced the number of preemptions overall.

However, tail TTFT (p99) was worse than FCFS in all five tested seeds, indicating a fairness/starvation tradeoff.

Seed-42 per-class data showed that RAG requests in particular experienced worse TTFT under SJF while interactive requests benefited dramatically.

This does **not** establish statistical significance, does **not** claim SJF is universally better, and does **not** treat the policy as true total-runtime SJF (future generated length is unknown).

This observation motivates the next experiment: **SJF + aging / starvation prevention** (`sjf_aging`).

## Raw JSON location (GPU machine only)

Not included in this repository commit. On the GPU host at the time of the runs:

```text
/results/dev_fcfs_seed42.json
/results/dev_fcfs_seed43.json
/results/dev_fcfs_seed44.json
/results/dev_fcfs_seed45.json
/results/dev_fcfs_seed46.json

/results/dev_sjf_seed42.json
/results/dev_sjf_seed43.json
/results/dev_sjf_seed44.json
/results/dev_sjf_seed45.json
/results/dev_sjf_seed46.json
```

Upload/copy separately if bit-exact JSON archival is required.
