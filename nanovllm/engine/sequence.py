from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0  # tokens with KV written (= num_computed_tokens)
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # INSTRUMENTATION-ONLY: unused by scheduling decisions
        self.workload_class = "unknown"
        self.arrival_time = 0.0
        self.first_admission_time = 0.0
        self.first_token_time = 0.0
        self.finish_time = 0.0
        self.num_preemptions = 0
        self.num_scheduler_steps = 0
        self.num_recomputed_tokens = 0
        self.requested_output_tokens = sampling_params.max_tokens
        self.client_request_id = -1
        self._instrument_admitted = False

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_computed_tokens(self) -> int:
        """Tokens whose KV is already in the cache (alias of num_cached_tokens)."""
        return self.num_cached_tokens

    @num_computed_tokens.setter
    def num_computed_tokens(self, value: int) -> None:
        self.num_cached_tokens = value

    @property
    def is_prefill_chunk(self) -> bool:
        """True while prefill/recompute KV is incomplete for currently known tokens.

        Matches original Nano-vLLM recompute semantics: after preemption, all
        tokens present in `token_ids` (prompt + generated) must regain KV before
        decode resumes. `num_computed_tokens` counts only tokens with valid KV.
        """
        return self.is_prefill and self.num_computed_tokens < self.num_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        """Tokens that still need KV written during prefill/recompute."""
        if not self.is_prefill:
            return 0
        return max(0, self.num_tokens - self.num_computed_tokens)

    @property
    def remaining_compute_tokens(self) -> int:
        """Alias used by the scheduler for prefill/recompute admission."""
        return self.remaining_prefill_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
