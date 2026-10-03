"""Scheduler tests for in-progress prefill accounting (no GPU)."""

from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(max_batched_tokens: int = 8, max_seqs: int = 4, num_blocks: int = 64) -> Scheduler:
    Sequence.block_size = 256
    cfg = SimpleNamespace(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_batched_tokens,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
    )
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = cfg.max_num_seqs
    sched.max_num_batched_tokens = cfg.max_num_batched_tokens
    sched.eos = cfg.eos
    sched.block_size = cfg.kvcache_block_size
    sched.block_manager = BlockManager(cfg.num_kvcache_blocks, cfg.kvcache_block_size)
    from collections import deque
    from nanovllm.engine.scheduler_metrics import SchedulerMetrics

    sched.waiting = deque()
    sched.running = deque()
    sched.scheduler_policy = "fcfs"
    sched.metrics = SchedulerMetrics(num_kv_blocks=num_blocks)
    sched.metrics.scheduler_policy = "fcfs"
    return sched


def test_partial_prefill_moves_to_running():
    sched = _make_scheduler(max_batched_tokens=4, max_seqs=4)
    prompt = list(range(10))
    seq = Sequence(prompt, SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    sched.add(seq)

    out = sched.schedule()
    assert out.is_prefill_only is True
    assert out.num_seqs == 1
    assert out.seqs[0] is seq
    assert seq.status == SequenceStatus.RUNNING
    assert seq.is_prefill_chunk is True  # prompt not fully computed yet
    assert seq.num_scheduled_tokens == 4
    assert seq in sched.running
    assert seq not in sched.waiting

    # Simulate model step advancing computed tokens without finishing prompt.
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    assert seq.is_prefill_chunk is True

    out2 = sched.schedule()
    assert out2.is_prefill_only is True
    assert out2.seqs[0] is seq
    assert seq.num_scheduled_tokens == 4  # next chunk clipped to budget
    assert seq in sched.running
