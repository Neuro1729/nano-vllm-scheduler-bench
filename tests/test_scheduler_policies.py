"""Deterministic FCFS control tests for scheduler policy abstraction (no GPU)."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.scheduler_policy import (
    remaining_prefill_score,
    select_waiting_index,
)
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(
    policy: str = "fcfs",
    max_batched_tokens: int = 64,
    max_seqs: int = 8,
    num_blocks: int = 256,
) -> Scheduler:
    Sequence.block_size = 256
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = max_seqs
    sched.max_num_batched_tokens = max_batched_tokens
    sched.eos = -1
    sched.block_size = 256
    sched.scheduler_policy = policy
    sched.block_manager = BlockManager(num_blocks, 256)
    sched.waiting = deque()
    sched.running = deque()
    sched.metrics = SchedulerMetrics(num_kv_blocks=num_blocks)
    sched.metrics.scheduler_policy = policy
    return sched


def _seq(n_prompt: int, max_tokens: int = 4) -> Sequence:
    return Sequence(
        list(range(n_prompt)),
        SamplingParams(temperature=1e-5, max_tokens=max_tokens, ignore_eos=True),
    )


def _admission_order(policy: str) -> list[int]:
    sched = _make_scheduler(policy=policy, max_batched_tokens=32, max_seqs=1)
    a = _seq(4000)
    b = _seq(100)
    c = _seq(500)
    sched.add(a)
    sched.add(b)
    sched.add(c)

    order: list[int] = []
    seen: set[int] = set()
    for _ in range(5000):
        if len(order) == 3:
            break
        out = sched.schedule()
        assert out.num_seqs == 1
        seq = out.seqs[0]
        if seq.seq_id not in seen:
            seen.add(seq.seq_id)
            order.append(seq.num_prompt_tokens)

        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if seq.num_computed_tokens >= seq.num_tokens:
            seq.is_prefill = False
            seq.status = SequenceStatus.FINISHED
            sched.block_manager.deallocate(seq)
            sched.running.clear()
    assert len(order) == 3
    return order


def test_fcfs_admits_in_arrival_order():
    """Control: policy=fcfs preserves parity-style A -> B -> C admission."""
    assert _admission_order("fcfs") == [4000, 100, 500]


def test_fcfs_selection_index_is_always_head():
    a = _seq(4000)
    b = _seq(100)
    c = _seq(500)
    waiting = deque([a, b, c])
    assert select_waiting_index("fcfs", waiting) == 0
    assert remaining_prefill_score(b) < remaining_prefill_score(a)


def test_fcfs_does_not_count_waiting_reorders():
    sched = _make_scheduler(policy="fcfs", max_batched_tokens=32)
    for n in (4000, 100, 500):
        sched.add(_seq(n))
    sched.schedule()
    assert sched.metrics.waiting_reorders == 0
    assert sched.metrics.to_dict()["scheduler_policy"] == "fcfs"
