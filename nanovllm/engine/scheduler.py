from __future__ import annotations

import os
from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.scheduler_output import SchedulerOutput

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    """vLLM-V1-style shared token-budget scheduler (parity baseline)."""

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

    def _record_iteration_metrics(self, output: SchedulerOutput) -> None:
        self.metrics.scheduler_iterations += 1
        self.metrics.scheduled_prefill_tokens += output.num_prefill_tokens
        self.metrics.scheduled_decode_tokens += output.num_decode_tokens
        self.metrics.chunked_prefill_count += sum(
            1
            for seq in output.seqs
            if seq.is_prefill
            and seq.num_computed_tokens + seq.num_scheduled_tokens < seq.num_tokens
        )
        if output.is_mixed:
            self.metrics.mixed_iterations += 1
        elif output.is_prefill_only:
            self.metrics.prefill_iterations += 1
        elif output.is_decode_only:
            self.metrics.decode_iterations += 1

    def _debug_enabled(self) -> bool:
        return os.environ.get("NANOVLLM_SCHED_DEBUG", "").strip() not in {"", "0", "false", "False"}

    def _waiting_state_dump(self, limit: int = 10) -> str:
        lines = ["waiting request diagnostic:"]
        rem_pos = rem_zero = no_table = computed_no_kv = prefilled_waiting = 0
        for seq in self.waiting:
            rem = seq.remaining_compute_tokens
            if rem > 0:
                rem_pos += 1
            else:
                rem_zero += 1
            if not seq.block_table:
                no_table += 1
            if seq.num_computed_tokens > 0 and not seq.block_table:
                computed_no_kv += 1
            if seq.num_computed_tokens >= seq.num_prompt_tokens and seq.status == SequenceStatus.WAITING:
                prefilled_waiting += 1

        lines.append(
            "  aggregates: "
            f"remaining>0={rem_pos} remaining==0={rem_zero} "
            f"no_block_table={no_table} "
            f"computed>0_without_kv={computed_no_kv} "
            f"computed>=prompt_but_waiting={prefilled_waiting}"
        )
        for seq in list(self.waiting)[:limit]:
            required_blocks = seq.num_blocks
            lines.append(
                "  "
                f"seq_id={seq.seq_id} status={seq.status.name} is_prefill={seq.is_prefill} "
                f"prompt={seq.num_prompt_tokens} output={seq.num_completion_tokens} "
                f"len={seq.num_tokens} computed={seq.num_computed_tokens} "
                f"cached={seq.num_cached_tokens} block_table_len={len(seq.block_table)} "
                f"remaining_compute={seq.remaining_compute_tokens} "
                f"required_blocks={required_blocks} max_tokens={seq.max_tokens}"
            )
        return "\n".join(lines)

    def _format_schedule_debug(
        self,
        *,
        token_budget: int,
        scheduled_seqs: list[Sequence],
        preempted_this_step: bool,
        num_preempted: int,
        num_skipped_waiting: int,
    ) -> str:
        base = (
            "scheduler empty-batch debug: "
            f"token_budget={token_budget} "
            f"len(running)={len(self.running)} "
            f"len(waiting)={len(self.waiting)} "
            f"free_kv_blocks={len(self.block_manager.free_block_ids)} "
            f"used_kv_blocks={len(self.block_manager.used_block_ids)} "
            f"scheduled_seqs={len(scheduled_seqs)} "
            f"preempted_this_step={preempted_this_step} "
            f"num_preempted={num_preempted} "
            f"num_skipped_waiting={num_skipped_waiting} "
            f"allocation_failures={self.metrics.allocation_failures} "
            f"allocation_failure_steps={self.metrics.allocation_failure_steps} "
            f"allocation_failed_candidates={self.metrics.allocation_failed_candidates} "
            f"hol_skipped_requests={self.metrics.hol_skipped_requests}"
        )
        return base + "\n" + self._waiting_state_dump()

    def _admit_from_waiting(
        self,
        scheduled_seqs: list[Sequence],
        token_budget: int,
    ) -> tuple[list[Sequence], int, int]:
        """FCFS waiting admission with HOL skip. Returns skipped count."""
        skipped: deque[Sequence] = deque()
        num_skipped = 0
        while (
            self.waiting
            and token_budget > 0
            and len(scheduled_seqs) < self.max_num_seqs
            and len(self.running) < self.max_num_seqs
        ):
            seq = self.waiting[0]
            self.metrics.examine_waiting_candidate()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.metrics.record_allocation_failure(hol_skipped=True)
                    skipped.append(self.waiting.popleft())
                    num_skipped += 1
                    continue
                self.block_manager.allocate(seq, num_cached_blocks)

            # Recompute/prefill work is over ALL currently known tokens (prompt +
            # generated), matching origin/real: num_tokens - num_cached_tokens.
            need = seq.remaining_compute_tokens
            if need <= 0:
                # Prefix cache already covers every known token: decode-ready.
                # Do not break the waiting queue (that caused empty-batch stalls).
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                self.running.append(seq)
                continue

            take = min(need, token_budget)
            seq.num_scheduled_tokens = take
            seq.is_prefill = True
            token_budget -= take
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
            self._instrument_on_schedule(seq)

        for seq in reversed(skipped):
            self.waiting.appendleft(seq)
        return scheduled_seqs, token_budget, num_skipped

    def _schedule_running(
        self,
        scheduled_seqs: list[Sequence],
        token_budget: int,
    ) -> tuple[list[Sequence], int, bool, int]:
        """Schedule from RUNNING. Returns preempted flag and preempt count."""
        preempted_this_step = False
        num_preempted = 0
        pending_running = self.running
        self.running = deque()

        while pending_running and len(scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = pending_running.popleft()

            if seq.is_prefill_chunk:
                take = min(seq.remaining_compute_tokens, token_budget)
                if take <= 0:
                    # Decode-ready while still flagged prefill: normalize and retry decode path.
                    seq.is_prefill = False
                    pending_running.appendleft(seq)
                    continue
                seq.num_scheduled_tokens = take
                seq.is_prefill = True
                token_budget -= take
                scheduled_seqs.append(seq)
                self.running.append(seq)
                self._instrument_on_schedule(seq)
                continue

            while not self.block_manager.can_append(seq):
                if pending_running:
                    self.preempt(pending_running.pop())
                    preempted_this_step = True
                    num_preempted += 1
                else:
                    self.preempt(seq)
                    preempted_this_step = True
                    num_preempted += 1
                    seq = None
                    break
            if seq is None:
                continue
            if token_budget < 1:
                self.running.append(seq)
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            token_budget -= 1
            scheduled_seqs.append(seq)
            self.running.append(seq)
            self._instrument_on_schedule(seq)

        self.running.extend(pending_running)
        return scheduled_seqs, token_budget, preempted_this_step, num_preempted

    def schedule(self) -> SchedulerOutput:
        scheduled_seqs: list[Sequence] = []
        token_budget = self.max_num_batched_tokens
        num_skipped_waiting = 0

        # INSTRUMENTATION-ONLY
        self.metrics.begin_iteration()
        self.metrics.observe_queues(
            len(self.waiting),
            len(self.running),
            len(self.block_manager.used_block_ids),
        )

        scheduled_seqs, token_budget, preempted_this_step, num_preempted = self._schedule_running(
            scheduled_seqs, token_budget
        )

        # Anti-thrash unless the batch is still empty (liveness for recompute).
        admit_waiting = (not preempted_this_step) or (not scheduled_seqs)
        if admit_waiting:
            scheduled_seqs, token_budget, num_skipped_waiting = self._admit_from_waiting(
                scheduled_seqs, token_budget
            )

        # Waiting may have promoted decode-ready sequences into running without
        # scheduling tokens. Give RUNNING a second chance in that case.
        if not scheduled_seqs and self.running and token_budget > 0:
            scheduled_seqs, token_budget, preempted2, num_preempted2 = self._schedule_running(
                scheduled_seqs, token_budget
            )
            preempted_this_step = preempted_this_step or preempted2
            num_preempted += num_preempted2

        if not scheduled_seqs:
            msg = self._format_schedule_debug(
                token_budget=token_budget,
                scheduled_seqs=scheduled_seqs,
                preempted_this_step=preempted_this_step,
                num_preempted=num_preempted,
                num_skipped_waiting=num_skipped_waiting,
            )
            if self._debug_enabled():
                print(msg)
            raise RuntimeError(
                "scheduler produced an empty batch with no schedulable tokens; " + msg
            )

        assert sum(s.num_scheduled_tokens for s in scheduled_seqs) <= self.max_num_batched_tokens

        output = SchedulerOutput(scheduled_seqs)
        self._record_iteration_metrics(output)
        return output

    def preempt(self, seq: Sequence):
        # INSTRUMENTATION-ONLY
        self.metrics.preemption_count += 1
        self.metrics.preempted_request_ids.add(seq.seq_id)
        recomputed = seq.num_cached_tokens
        self.metrics.recomputed_tokens += recomputed
        seq.num_preemptions += 1
        seq.num_recomputed_tokens += recomputed

        # Destructive recompute: discard KV and invalidate compute progress.
        # deallocate() zeros num_cached_tokens (= num_computed_tokens) and clears
        # the block table. Prefix-cache hits are rediscovered on next allocate().
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        assert seq.num_computed_tokens == 0
        assert not seq.block_table
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        now = perf_counter()  # INSTRUMENTATION-ONLY timestamp
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Real semantics: while recomputing/prefilling existing tokens, do not
            # append a new sampled token until computed catches up to num_tokens.
            if seq.is_prefill and seq.num_computed_tokens < seq.num_tokens:
                continue
            seq.is_prefill = False
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
