"""Deterministic sequence-state tests for parity accounting (no GPU)."""

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def test_computed_tokens_alias_and_prefill_chunk():
    Sequence.block_size = 256
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=1e-5, max_tokens=8, ignore_eos=True))
    assert seq.num_prompt_tokens == 5
    assert seq.num_computed_tokens == 0
    assert seq.is_prefill_chunk is True
    assert seq.remaining_prefill_tokens == 5

    seq.num_computed_tokens = 3
    assert seq.num_cached_tokens == 3
    assert seq.is_prefill_chunk is True
    assert seq.remaining_prefill_tokens == 2

    seq.num_computed_tokens = 5
    assert seq.is_prefill_chunk is False
    assert seq.remaining_prefill_tokens == 0


def test_recompute_covers_generated_tokens_not_just_prompt():
    Sequence.block_size = 256
    seq = Sequence(list(range(10)), SamplingParams(temperature=1e-5, max_tokens=8, ignore_eos=True))
    seq.append_token(100)
    seq.append_token(101)
    assert seq.num_tokens == 12
    assert seq.num_prompt_tokens == 10
    # After destructive preemption accounting reset:
    seq.is_prefill = True
    seq.num_computed_tokens = 0
    assert seq.is_prefill_chunk is True
    assert seq.remaining_compute_tokens == 12  # prompt + generated


def test_running_partial_prefill_status_fields():
    Sequence.block_size = 256
    seq = Sequence(list(range(10)), SamplingParams(temperature=1e-5, max_tokens=4, ignore_eos=True))
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = 4
    seq.num_scheduled_tokens = 3
    assert seq.status == SequenceStatus.RUNNING
    assert seq.is_prefill_chunk is True
    assert seq.remaining_prefill_tokens == 6
    seq.num_computed_tokens = 10
    assert seq.is_prefill_chunk is False
