"""Regression: recompute accounting after destructive KV preemption."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(num_blocks: int = 64, max_batched_tokens: int = 128, max_seqs: int = 16) -> Scheduler:
    Sequence.block_size = 256
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = max_seqs
    sched.max_num_batched_tokens = max_batched_tokens
    sched.eos = -1
    sched.block_size = 256
    sched.block_manager = BlockManager(num_blocks, 256)
    sched.waiting = deque()
    sched.running = deque()
    sched.metrics = SchedulerMetrics(num_kv_blocks=num_blocks)
    return sched


def _drive_until_finished(sched: Scheduler, max_steps: int = 500) -> None:
    for _ in range(max_steps):
        if sched.is_finished():
            return
        out = sched.schedule()
        assert out.num_seqs > 0
        fake_tokens = [7] * out.num_seqs
        sched.postprocess(out.seqs, fake_tokens)
    raise AssertionError("scheduler did not finish within max_steps")


def test_preempted_decode_must_recompute_all_known_tokens():
    """Fails on 7ee2d8d: remaining_prefill used prompt-only, so after a prefix-sized
    allocate progress, waiting head could have remaining==0 and stall the queue.
    """
    sched = _make_scheduler(num_blocks=8, max_batched_tokens=64)
    prompt = list(range(100))
    seq = Sequence(prompt, SamplingParams(temperature=1e-5, max_tokens=5, ignore_eos=True))
    sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = seq.num_prompt_tokens
    for t in (201, 202, 203):
        seq.append_token(t)
    seq.num_cached_tokens = seq.num_tokens - 1
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    sched.running.append(seq)

    sched.running.remove(seq)
    sched.preempt(seq)
    assert seq.status == SequenceStatus.WAITING
    assert seq.is_prefill is True
    assert seq.num_computed_tokens == 0
    assert not seq.block_table
    assert seq.remaining_compute_tokens == seq.num_tokens == 103
    assert not (seq.status == SequenceStatus.WAITING and seq.remaining_compute_tokens == 0)

    out = sched.schedule()
    assert out.num_seqs == 1
    assert seq in out.seqs
    assert seq.is_prefill is True
    assert seq.num_scheduled_tokens > 0
    assert seq.num_scheduled_tokens == min(64, 103)


def test_prefix_allocate_cannot_stall_waiting_with_zero_remaining():
    """Simulate allocate() setting cached tokens above prompt length via block hits.

    Old remaining_prefill_tokens = prompt - computed became 0 and broke waiting.
    """
    sched = _make_scheduler(num_blocks=8, max_batched_tokens=32)
    # 300 tokens => 2 blocks; one full cached block is 256 tokens (>= prompt of 100).
    seq = Sequence(list(range(100)), SamplingParams(temperature=1e-5, max_tokens=4, ignore_eos=True))
    for t in range(200):
        seq.append_token(1000 + t)
    assert seq.num_tokens == 300
    seq.is_prefill = True
    seq.status = SequenceStatus.WAITING
    seq.num_computed_tokens = 0
    sched.waiting.append(seq)

    # Manually mimic allocate() after a 1-block prefix hit.
    seq.block_table = [sched.block_manager._allocate_block(), sched.block_manager._allocate_block()]
    seq.num_cached_tokens = 256  # >= prompt (100), < num_tokens (300)
    assert seq.remaining_prefill_tokens == 44
    old_remaining = max(0, seq.num_prompt_tokens - seq.num_computed_tokens)
    assert old_remaining == 0

    out = sched.schedule()
    assert out.num_seqs == 1
    assert seq.num_scheduled_tokens == 32  # clipped to budget, not stalled


def test_preempted_request_eventually_finishes_under_tiny_kv():
    sched = _make_scheduler(num_blocks=2, max_batched_tokens=32, max_seqs=4)
    seqs = []
    for i in range(3):
        seq = Sequence(
            list(range(i * 10, i * 10 + 8)),
            SamplingParams(temperature=1e-5, max_tokens=3, ignore_eos=True),
        )
        seqs.append(seq)
        sched.add(seq)

    _drive_until_finished(sched, max_steps=400)
    assert all(seq.is_finished for seq in seqs)
    assert sched.is_finished()


def test_invariant_no_waiting_zero_remaining_unfinished():
    """WAITING + unfinished + remaining_compute==0 must not persist after preempt."""
    sched = _make_scheduler(num_blocks=4, max_batched_tokens=16)
    seq = Sequence(list(range(20)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.append_token(1)
    seq.append_token(2)
    seq.num_cached_tokens = seq.num_tokens - 1
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    sched.running.append(seq)

    # Match scheduler.preempt call sites: seq is removed from running first.
    sched.running.remove(seq)
    sched.preempt(seq)
    assert seq.status == SequenceStatus.WAITING
    assert seq.remaining_compute_tokens > 0
    assert seq.num_computed_tokens == 0
    assert not seq.block_table

    out = sched.schedule()
    assert out.num_seqs >= 1
    assert seq.status == SequenceStatus.RUNNING


def test_prefix_cache_reuse_after_preempt_still_schedules_tail():
    """After recompute preemption, prefix-cache hits reduce work but must not zero it out."""
    sched = _make_scheduler(num_blocks=8, max_batched_tokens=64)
    # Keep a hashed donor resident so prefix lookup stays valid (allocate(0) on a
    # free hashed block would reset it and drop the hash entry).
    donor = Sequence(list(range(300)), SamplingParams(temperature=1e-5, max_tokens=1, ignore_eos=True))
    sched.block_manager.allocate(donor, 0)
    donor.num_scheduled_tokens = 256
    donor.num_cached_tokens = 0
    sched.block_manager.hash_blocks(donor)
    donor.num_cached_tokens = 256
    donor.status = SequenceStatus.RUNNING
    donor.is_prefill = False

    victim = Sequence(list(range(300)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    sched.block_manager.allocate(victim, 0)
    victim.num_cached_tokens = victim.num_prompt_tokens
    victim.append_token(999)
    victim.num_cached_tokens = victim.num_tokens - 1
    victim.status = SequenceStatus.RUNNING
    victim.is_prefill = False
    sched.running.append(victim)

    sched.running.remove(victim)
    sched.preempt(victim)
    assert victim.num_computed_tokens == 0

    out = sched.schedule()
    assert victim in out.seqs
    # Prefix should restore 256 cached tokens; schedule the remaining tail.
    assert victim.num_computed_tokens == 256
    assert victim.num_scheduled_tokens == min(64, 301 - 256)
    # Valid prefix reuse must remain available (not wiped by the recompute reset).
    assert donor.num_computed_tokens == 256
    assert donor.block_table
