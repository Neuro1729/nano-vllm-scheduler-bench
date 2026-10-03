# Experimental scheduler policies (`dev`)

`real` and `parity` stay frozen. `dev` = parity + experimental policies.

## Policy summary

| Policy | Idea |
|--------|------|
| **FCFS** | Arrival-order baseline (parity WAITING queue + running-first) |
| **SJF** | Uses **known** remaining compute (`num_tokens - num_computed`) |
| **SJF+Aging** | SJF plus starvation prevention by waiting-age threshold |
| **MLFQ** | Does **not** know job size; infers behavior from **scheduled-token** service |

Do not claim MLFQ is better until GPU results exist.

## Config

```text
--scheduler-policy fcfs|sjf|sjf_aging|mlfq   (default: fcfs)
--aging-threshold N                          (default: 128; sjf_aging)
--mlfq-q0-quantum 256
--mlfq-q1-quantum 1024
--mlfq-boost-interval 256                    (0 disables boost)
```

## SJF score

```text
score = remaining_prefill_tokens
      = max(0, num_tokens - num_computed_tokens)   # while prefill/recompute
```

## Waiting age (`sjf_aging`)

| Event | Age effect |
|-------|------------|
| Enter WAITING (`add`) | start at 0 |
| Each `schedule()` while still WAITING | `+1` |
| HOL skip / lose SJF contest | age **kept** |
| Admitted to RUNNING | reset to 0 |
| Recompute preemption back to WAITING | **restart at 0** |

## MLFQ (implemented)

Three levels: Q0 (highest) → Q1 → Q2 (bottom).

- New requests start in **Q0**.
- Service = scheduled tokens (prefill + decode).
- Exhaust Q0 quantum → demote to Q1; exhaust Q1 → Q2; Q2 does not demote.
- Prefill chunks are clipped to **remaining quantum** (cannot silently burn multiple levels in one slice).
- Within a level: deterministic round-robin (deque rotate across `schedule()` calls).
- Higher levels are fully preferred while they have eligible work.
- Demotion is priority-only: **no** KV free / recompute.
- Every `mlfq_boost_interval` scheduler iterations, unfinished requests return to Q0 (KV kept).
- True KV-pressure preemption **retains** `mlfq_level` and `mlfq_service_in_level`.

Recommended first GPU baseline:

```text
Q0=256, Q1=1024, boost=256
```

Compare on seed 42 against FCFS, SJF, and SJF+Aging-64.
