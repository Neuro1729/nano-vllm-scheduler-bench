"""Regression: scheduler must make progress after KV preemption / HOL blocks."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(num_blocks: int, max_batched_tokens: int = 64, max_seqs: int = 8) -> Scheduler:
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


def test_preempt_self_admits_recompute_same_step():
    """a1b5bca failure shape: sole running decode cannot append and preempts itself.

    Old code:
      preempted_this_step=True -> skip waiting -> assert scheduled_seqs
    Fixed code:
      empty scheduled after preempt -> still admit waiting (recompute).
    """
    sched = _make_scheduler(num_blocks=1, max_batched_tokens=64)
    # len==1 => next decode append needs a new block (1 % 256 == 1).
    seq = Sequence([42], SamplingParams(temperature=1e-5, max_tokens=4, ignore_eos=True))
    sched.block_manager.allocate(seq, 0)
    assert len(sched.block_manager.free_block_ids) == 0
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    assert len(seq) % 256 == 1
    assert not sched.block_manager.can_append(seq)
    sched.running.append(seq)

    out = sched.schedule()
    assert out.num_seqs > 0
    assert out.total_scheduled_tokens > 0
    assert sched.metrics.preemption_count >= 1
    assert seq in out.seqs
    assert seq.is_prefill is True  # recomputation prefill after preempt


def test_preempt_other_then_schedule_decode():
    """Decode needing a block preempts another resident and then schedules."""
    sched = _make_scheduler(num_blocks=2, max_batched_tokens=32)
    a = Sequence([1], SamplingParams(temperature=1e-5, max_tokens=4, ignore_eos=True))
    b = Sequence([2], SamplingParams(temperature=1e-5, max_tokens=4, ignore_eos=True))
    sched.block_manager.allocate(a, 0)
    sched.block_manager.allocate(b, 0)
    assert len(sched.block_manager.free_block_ids) == 0

    for seq in (a, b):
        seq.num_cached_tokens = seq.num_prompt_tokens
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        assert len(seq) % 256 == 1
        sched.running.append(seq)

    out = sched.schedule()
    assert out.num_seqs > 0
    assert out.total_scheduled_tokens > 0
    assert sched.metrics.preemption_count >= 1
    assert a in out.seqs and a.is_prefill is False


def test_hol_skip_schedules_later_waiting():
    sched = _make_scheduler(num_blocks=1, max_batched_tokens=16)
    hog = Sequence(list(range(10)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    sched.block_manager.allocate(hog, 0)
    assert len(sched.block_manager.free_block_ids) == 0

    blocked = Sequence(list(range(20, 30)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    ok = Sequence(list(range(40, 50)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    ok.block_table = list(hog.block_table)
    ok.num_cached_tokens = 0
    ok.status = SequenceStatus.WAITING

    sched.add(blocked)
    sched.waiting.append(ok)

    out = sched.schedule()
    assert ok in out.seqs
    assert out.total_scheduled_tokens > 0
    assert blocked in sched.waiting


def test_old_waiting_gate_vs_new_liveness_gate():
    preempted_this_step = True
    scheduled_seqs: list[object] = []
    new_gate = (not preempted_this_step) or (not scheduled_seqs)
    old_gate = not preempted_this_step
    assert old_gate is False
    assert new_gate is True
