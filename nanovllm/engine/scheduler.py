from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_metrics import SchedulerMetrics


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # INSTRUMENTATION-ONLY
        self.metrics = SchedulerMetrics(num_kv_blocks=max(config.num_kvcache_blocks, 1))

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _instrument_on_schedule(self, seq: Sequence) -> None:
        # INSTRUMENTATION-ONLY: counters/timestamps; no control-flow impact
        now = perf_counter()
        if not seq._instrument_admitted:
            seq.first_admission_time = now
            seq._instrument_admitted = True
        seq.num_scheduler_steps += 1

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # INSTRUMENTATION-ONLY
        self.metrics.begin_iteration()
        self.metrics.observe_queues(
            len(self.waiting),
            len(self.running),
            len(self.block_manager.used_block_ids),
        )

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # INSTRUMENTATION-ONLY
            self.metrics.examine_waiting_candidate()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # INSTRUMENTATION-ONLY (real: HOL stop, not skip)
                    self.metrics.record_allocation_failure(hol_skipped=False)
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            self._instrument_on_schedule(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # INSTRUMENTATION-ONLY
            self.metrics.scheduler_iterations += 1
            self.metrics.prefill_iterations += 1
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                self._instrument_on_schedule(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        # INSTRUMENTATION-ONLY
        self.metrics.scheduler_iterations += 1
        self.metrics.decode_iterations += 1
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # INSTRUMENTATION-ONLY
        self.metrics.preemption_count += 1
        self.metrics.preempted_request_ids.add(seq.seq_id)
        recomputed = seq.num_cached_tokens
        self.metrics.recomputed_tokens += recomputed
        seq.num_preemptions += 1
        seq.num_recomputed_tokens += recomputed

        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        now = perf_counter()  # INSTRUMENTATION-ONLY timestamp
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # INSTRUMENTATION-ONLY
            if seq.first_token_time == 0.0:
                seq.first_token_time = now
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # INSTRUMENTATION-ONLY
                seq.finish_time = now
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
