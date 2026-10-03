"""CPU-side checks for mixed-batch token packing (no GPU kernels)."""

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _pack_varlen(seqs: list[Sequence]):
    """Mirror ModelRunner.prepare_prefill packing without CUDA tensors."""
    input_ids = []
    positions = []
    cu_q = [0]
    cu_k = [0]
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        if seq.is_prefill:
            input_ids.extend(seq[start:end])
        else:
            input_ids.append(seq.last_token)
        positions.extend(range(start, end))
        cu_q.append(cu_q[-1] + seqlen_q)
        cu_k.append(cu_k[-1] + end)
    return input_ids, positions, cu_q, cu_k


def test_mixed_decode_and_prefill_packing():
    Sequence.block_size = 256
    decode = Sequence([10, 11, 12, 13], SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    decode.num_cached_tokens = 3  # KV for first 3 tokens
    decode.num_scheduled_tokens = 1
    decode.is_prefill = False

    prefill = Sequence(list(range(20, 30)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    prefill.num_cached_tokens = 0
    prefill.num_scheduled_tokens = 4
    prefill.is_prefill = True

    ids, pos, cu_q, cu_k = _pack_varlen([decode, prefill])
    assert ids == [13, 20, 21, 22, 23]
    assert pos == [3, 0, 1, 2, 3]
    assert cu_q == [0, 1, 5]
    assert cu_k == [0, 4, 8]
    assert cu_k[-1] > cu_q[-1]  # requires block tables / paged KV
