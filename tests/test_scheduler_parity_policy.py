"""Shared token-budget / mixed / HOL-skip scheduler tests (no GPU)."""

from collections import deque
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(max_batched_tokens: int = 16, max_seqs: int = 8, num_blocks: int = 128) -> Scheduler:
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


def _seq(n_prompt: int, max_tokens: int = 4) -> Sequence:
    return Sequence(
        list(range(n_prompt)),
        SamplingParams(temperature=1e-5, max_tokens=max_tokens, ignore_eos=True),
    )


def test_shared_budget_never_exceeded():
    sched = _make_scheduler(max_batched_tokens=10, max_seqs=8)
    for _ in range(5):
        sched.add(_seq(20))
    out = sched.schedule()
    assert out.total_scheduled_tokens <= 10
    assert out.total_scheduled_tokens == 10


def test_any_prefill_can_be_chunked():
    sched = _make_scheduler(max_batched_tokens=10, max_seqs=8)
    a = _seq(8)
    b = _seq(8)
    sched.add(a)
    sched.add(b)
    out = sched.schedule()
    # Old policy would refuse to admit B when it cannot fit fully.
    # Shared-budget parity chunks B into the remaining 2 tokens.
    assert out.num_prefill_seqs == 2
    assert a.num_scheduled_tokens == 8
    assert b.num_scheduled_tokens == 2
    assert out.total_scheduled_tokens == 10


def test_running_decode_before_waiting_prefill_mixed():
    sched = _make_scheduler(max_batched_tokens=8, max_seqs=8)
    decode = _seq(4, max_tokens=3)
    # Pretend prompt already fully computed and first token appended.
    sched.block_manager.allocate(decode, 0)
    decode.num_cached_tokens = decode.num_prompt_tokens
    decode.append_token(99)
    decode.num_cached_tokens = decode.num_prompt_tokens  # KV through prompt; last token pending
    # Align with engine invariants after first completion token:
    decode.num_cached_tokens = len(decode) - 1
    decode.status = SequenceStatus.RUNNING
    decode.is_prefill = False
    sched.running.append(decode)

    waiting = _seq(20)
    sched.add(waiting)

    out = sched.schedule()
    assert out.is_mixed
    assert out.num_decode_seqs == 1
    assert out.num_prefill_seqs >= 1
    assert decode in out.seqs and decode.is_prefill is False
    assert decode.num_scheduled_tokens == 1
    assert out.total_scheduled_tokens <= 8
    assert sched.metrics.mixed_iterations == 1


def test_hol_skip_blocked_waiting_head():
    sched = _make_scheduler(max_batched_tokens=32, max_seqs=4, num_blocks=2)
    # Fill KV with one resident decode-ready sequence occupying both blocks... 
    # With block_size 256, each seq needs 1 block. Exhaust free blocks.
    hog = _seq(10)
    sched.block_manager.allocate(hog, 0)
    # Consume remaining free blocks so a new allocate fails.
    while sched.block_manager.free_block_ids:
        sched.block_manager.free_block_ids.popleft()

    blocked = _seq(8)
    ok = _seq(8)
    # Give `ok` a pre-built block_table so it does not need allocate().
    ok.block_table = [0]  # reuse hog's block id only for scheduling metadata in this unit test
    ok.num_cached_tokens = 0
    ok.status = SequenceStatus.WAITING

    sched.add(blocked)  # head: needs allocate -> fail -> skip
    sched.waiting.append(ok)

    out = sched.schedule()
    # Blocked head skipped; request with existing table can still be chunk-scheduled.
    assert ok in out.seqs
    assert blocked in sched.waiting  # re-queued after skip
    assert sched.metrics.allocation_failures >= 1
    assert sched.metrics.allocation_failure_steps >= 1
    assert sched.metrics.allocation_failed_candidates >= 1
    assert sched.metrics.hol_skipped_requests >= 1


def test_decode_progress_while_long_prefill_present():
    sched = _make_scheduler(max_batched_tokens=5, max_seqs=8)
    decode = _seq(3, max_tokens=5)
    sched.block_manager.allocate(decode, 0)
    decode.append_token(7)
    decode.num_cached_tokens = len(decode) - 1
    decode.status = SequenceStatus.RUNNING
    decode.is_prefill = False
    sched.running.append(decode)

    long_p = _seq(50)
    sched.add(long_p)

    outs = []
    for _ in range(3):
        out = sched.schedule()
        outs.append(out)
        # Advance computed tokens as postprocess would for scheduled work.
        for seq in out.seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            scheduled = seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill_chunk:
                continue
            if seq is decode:
                seq.append_token(1)
                # after append, cached should lag by 1 until next decode writes KV
                seq.num_cached_tokens = len(seq) - 1

    assert any(o.is_mixed or o.num_decode_seqs == 1 for o in outs)
    assert decode.num_completion_tokens >= 1
