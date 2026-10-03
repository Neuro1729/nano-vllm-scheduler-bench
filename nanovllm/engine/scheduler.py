from __future__ import annotations

import os
from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.scheduler_output import SchedulerOutput
from nanovllm.engine.scheduler_policy import (
    normalize_scheduler_policy,
    pop_waiting_at,
    select_waiting_choice,
)
from nanovllm.engine import mlfq as mlfq_lib

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    """vLLM-V1-style shared token-budget scheduler with pluggable policies."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.scheduler_policy = normalize_scheduler_policy(
            getattr(config, "scheduler_policy", "fcfs")
        )
        self.scheduler_aging_threshold = int(getattr(config, "scheduler_aging_threshold", 128))
        self.mlfq_q0_quantum = int(getattr(config, "mlfq_q0_quantum", 256))
        self.mlfq_q1_quantum = int(getattr(config, "mlfq_q1_quantum", 1024))
        self.mlfq_boost_interval = int(getattr(config, "mlfq_boost_interval", 256))
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # MLFQ RR queues (seq objects). Only used when scheduler_policy == "mlfq".
        self._mlfq_queues: list[deque[Sequence]] = [deque(), deque(), deque()]
        self._mlfq_epoch = 0
        # INSTRUMENTATION-ONLY
        self.metrics = SchedulerMetrics(num_kv_blocks=max(config.num_kvcache_blocks, 1))
        self.metrics.scheduler_policy = self.scheduler_policy
        self.metrics.scheduler_aging_threshold = self.scheduler_aging_threshold
        self.metrics.mlfq_q0_quantum = self.mlfq_q0_quantum
        self.metrics.mlfq_q1_quantum = self.mlfq_q1_quantum
        self.metrics.mlfq_boost_interval = self.mlfq_boost_interval

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.waiting_age_steps = 0
        if self.scheduler_policy == "mlfq":
            seq.mlfq_level = 0
            seq.mlfq_service_in_level = 0
            seq.mlfq_wait_steps = 0
            self._mlfq_queues[0].append(seq)
            self.metrics.mlfq_q0_admissions += 1
        self.waiting.append(seq)

    def _bump_waiting_ages(self) -> None:
        """+1 waiting age for every request currently in WAITING (deterministic)."""
        for seq in self.waiting:
            seq.waiting_age_steps += 1
            self.metrics.observe_waiting_age(seq.waiting_age_steps)

    def _on_leave_waiting(self, seq: Sequence, *, aging_promoted: bool) -> None:
        self.metrics.record_admission_age(seq.waiting_age_steps, aging_promoted=aging_promoted)
        seq.waiting_age_steps = 0

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
                f"required_blocks={required_blocks} max_tokens={seq.max_tokens} "
                f"mlfq_level={seq.mlfq_level}"
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

    # ------------------------------------------------------------------
    # Parity / SJF / SJF-aging path (unchanged behavior)
    # ------------------------------------------------------------------

    def _admit_from_waiting(
        self,
        scheduled_seqs: list[Sequence],
        token_budget: int,
    ) -> tuple[list[Sequence], int, int]:
        """WAITING admission with HOL skip; order from ``scheduler_policy``."""
        policy = getattr(self, "scheduler_policy", "fcfs")
        aging_threshold = getattr(self, "scheduler_aging_threshold", 128)
        skipped: deque[Sequence] = deque()
        num_skipped = 0
        while (
            self.waiting
            and token_budget > 0
            and len(scheduled_seqs) < self.max_num_seqs
            and len(self.running) < self.max_num_seqs
        ):
            choice = select_waiting_choice(
                policy, self.waiting, aging_threshold=aging_threshold
            )
            if choice.index != 0:
                self.metrics.waiting_reorders += 1
            seq = pop_waiting_at(self.waiting, choice.index)
            self.metrics.examine_waiting_candidate()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.metrics.record_allocation_failure(hol_skipped=True)
                    skipped.append(seq)
                    num_skipped += 1
                    continue
                self.block_manager.allocate(seq, num_cached_blocks)

            need = seq.remaining_compute_tokens
            if need <= 0:
                self._on_leave_waiting(seq, aging_promoted=choice.aging_promoted)
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                self.running.append(seq)
                continue

            take = min(need, token_budget)
            seq.num_scheduled_tokens = take
            seq.is_prefill = True
            token_budget -= take
            self._on_leave_waiting(seq, aging_promoted=choice.aging_promoted)
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

    def _schedule_baseline(self) -> SchedulerOutput:
        scheduled_seqs: list[Sequence] = []
        token_budget = self.max_num_batched_tokens
        num_skipped_waiting = 0

        self.metrics.begin_iteration()
        self.metrics.observe_queues(
            len(self.waiting),
            len(self.running),
            len(self.block_manager.used_block_ids),
        )

        scheduled_seqs, token_budget, preempted_this_step, num_preempted = self._schedule_running(
            scheduled_seqs, token_budget
        )

        self._bump_waiting_ages()

        admit_waiting = (not preempted_this_step) or (not scheduled_seqs)
        if admit_waiting:
            scheduled_seqs, token_budget, num_skipped_waiting = self._admit_from_waiting(
                scheduled_seqs, token_budget
            )

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
        self._assert_model_runner_contract(output)
        return output

    # ------------------------------------------------------------------
    # MLFQ path: priority levels + service-token demotion + boost
    # ------------------------------------------------------------------

    def _mlfq_maybe_boost(self) -> None:
        interval = self.mlfq_boost_interval
        if interval <= 0:
            return
        # After every `interval` schedule() calls (epoch already incremented).
        if self._mlfq_epoch % interval != 0:
            return
        active = list(self.waiting) + list(self.running)
        boosted = mlfq_lib.boost_all(self._mlfq_queues, active)
        self.metrics.mlfq_priority_boosts += 1
        self.metrics.mlfq_requests_boosted += boosted

    def _mlfq_charge(self, seq: Sequence, tokens: int) -> None:
        level_before = seq.mlfq_level
        self.metrics.record_mlfq_service(level_before, tokens)
        demoted = mlfq_lib.charge_service(
            seq,
            tokens,
            self._mlfq_queues,
            self.mlfq_q0_quantum,
            self.mlfq_q1_quantum,
        )
        if demoted:
            self.metrics.record_mlfq_demotion(seq.mlfq_level)

    def _mlfq_ensure_resident(self, seq: Sequence) -> bool:
        """Allocate KV and move WAITING -> RUNNING if needed. False if cannot."""
        if seq.status == SequenceStatus.RUNNING and seq.block_table:
            return True
        if seq in self.running and seq.block_table:
            return True
        if len(self.running) >= self.max_num_seqs and seq not in self.running:
            return False
        self.metrics.examine_waiting_candidate()
        if not seq.block_table:
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                self.metrics.record_allocation_failure(hol_skipped=True)
                return False
            self.block_manager.allocate(seq, num_cached_blocks)
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.status = SequenceStatus.RUNNING
        if seq not in self.running:
            self.running.append(seq)
        return True

    def _mlfq_preempt_for_append(
        self,
        seq: Sequence,
        scheduled_seqs: list[Sequence],
    ) -> tuple[bool, int]:
        """Free KV for decode append. Returns (ok, num_preempted).

        Never preempt a sequence already placed in ``scheduled_seqs`` for this
        iteration: that would deallocate its block_table while ModelRunner still
        expects matching slot_mapping entries (parity baseline only preempts
        unscheduled RUNNING work, or self).
        """
        scheduled_ids = {s.seq_id for s in scheduled_seqs}
        num_preempted = 0
        while not self.block_manager.can_append(seq):
            victim = None
            for cand in reversed(self.running):
                if cand is seq or cand.is_finished:
                    continue
                if cand.seq_id in scheduled_ids:
                    continue
                victim = cand
                break
            if victim is None:
                self.preempt(seq)
                num_preempted += 1
                return False, num_preempted
            self.preempt(victim)
            num_preempted += 1
        return True, num_preempted

    def _mlfq_try_serve(
        self,
        seq: Sequence,
        token_budget: int,
        scheduled_seqs: list[Sequence],
    ) -> tuple[int, int, bool]:
        """Attempt to schedule ``seq``. Returns (tokens_used, preempted, scheduled)."""
        if seq.is_finished:
            mlfq_lib.detach_from_queues(self._mlfq_queues, seq)
            return 0, 0, False

        if not self._mlfq_ensure_resident(seq):
            # Cannot allocate/admit: keep at end of its current level for later.
            mlfq_lib.place_on_level(self._mlfq_queues, seq)
            return 0, 0, False

        rem_q = mlfq_lib.remaining_quantum(seq, self.mlfq_q0_quantum, self.mlfq_q1_quantum)
        if rem_q <= 0:
            demoted = mlfq_lib.demote(
                seq, self._mlfq_queues, self.mlfq_q0_quantum, self.mlfq_q1_quantum
            )
            if demoted:
                self.metrics.record_mlfq_demotion(seq.mlfq_level)
            else:
                mlfq_lib.place_on_level(self._mlfq_queues, seq)
            return 0, 0, False

        # Prefill / recompute chunk limited by MLFQ remaining quantum.
        if seq.is_prefill and seq.num_computed_tokens < seq.num_tokens:
            need = seq.remaining_compute_tokens
            take = min(need, token_budget, rem_q)
            if take <= 0:
                # No prefill work under current budget; keep prefill flag and retry later.
                mlfq_lib.place_on_level(self._mlfq_queues, seq)
                return 0, 0, False
            seq.num_scheduled_tokens = take
            seq.is_prefill = True
            scheduled_seqs.append(seq)
            self._instrument_on_schedule(seq)
            self._mlfq_charge(seq, take)
            if not seq.is_finished:
                mlfq_lib.place_on_level(self._mlfq_queues, seq)
            return take, 0, True

        # Decode (1 token, also limited by remaining quantum).
        ok, num_preempted = self._mlfq_preempt_for_append(seq, scheduled_seqs)
        if not ok:
            # seq was self-preempted; still on MLFQ queue via preempt()->waiting
            mlfq_lib.place_on_level(self._mlfq_queues, seq)
            return 0, num_preempted, False
        if token_budget < 1:
            mlfq_lib.place_on_level(self._mlfq_queues, seq)
            return 0, num_preempted, False

        take = min(1, rem_q, token_budget)
        if take <= 0:
            mlfq_lib.place_on_level(self._mlfq_queues, seq)
            return 0, num_preempted, False
        seq.num_scheduled_tokens = take
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
        self._instrument_on_schedule(seq)
        self._mlfq_charge(seq, take)
        if not seq.is_finished:
            mlfq_lib.place_on_level(self._mlfq_queues, seq)
        return take, num_preempted, True

    def _assert_model_runner_contract(self, output: SchedulerOutput) -> None:
        """Cheap invariants: scheduled metadata must match ModelRunner packing."""
        seen: set[int] = set()
        sum_scheduled = 0
        for seq in output.seqs:
            if seq.seq_id in seen:
                raise RuntimeError(
                    f"scheduler contract: duplicate seq_id={seq.seq_id} in one schedule()"
                )
            seen.add(seq.seq_id)
            take = seq.num_scheduled_tokens
            sum_scheduled += take
            if take <= 0:
                raise RuntimeError(
                    "scheduler contract: non-positive scheduled tokens "
                    f"seq_id={seq.seq_id} take={take}"
                )
            if not seq.block_table:
                raise RuntimeError(
                    "scheduler contract: scheduled sequence has empty block_table "
                    f"(seq_id={seq.seq_id} status={seq.status.name} "
                    f"is_prefill={seq.is_prefill} mlfq_level={seq.mlfq_level} "
                    f"mlfq_service_in_level={seq.mlfq_service_in_level} "
                    f"num_tokens={seq.num_tokens} num_prompt_tokens={seq.num_prompt_tokens} "
                    f"num_computed_tokens={seq.num_computed_tokens} "
                    f"scheduled_tokens={take} "
                    f"iteration={self.metrics.scheduler_iterations})"
                )
            start = seq.num_cached_tokens
            end = start + take
            if seq.is_prefill:
                if end > seq.num_tokens:
                    raise RuntimeError(
                        "scheduler contract: prefill scheduled past sequence end "
                        f"seq_id={seq.seq_id} start={start} take={take} "
                        f"num_tokens={seq.num_tokens}"
                    )
            elif take != 1:
                raise RuntimeError(
                    f"scheduler contract: decode must schedule 1 token, got {take} "
                    f"seq_id={seq.seq_id}"
                )
            end_block = (end + self.block_size - 1) // self.block_size
            if end_block > len(seq.block_table):
                raise RuntimeError(
                    "scheduler contract: block_table too short for scheduled span "
                    f"seq_id={seq.seq_id} end={end} end_block={end_block} "
                    f"block_table_len={len(seq.block_table)} is_prefill={seq.is_prefill} "
                    f"mlfq_level={seq.mlfq_level} scheduled_tokens={take}"
                )
        if sum_scheduled != output.total_scheduled_tokens:
            raise RuntimeError(
                "scheduler contract: sum scheduled mismatch "
                f"sum={sum_scheduled} output={output.total_scheduled_tokens}"
            )

    def _schedule_mlfq(self) -> SchedulerOutput:
        scheduled_seqs: list[Sequence] = []
        token_budget = self.max_num_batched_tokens
        preempted_this_step = False
        num_preempted = 0

        self.metrics.begin_iteration()
        self.metrics.observe_queues(
            len(self.waiting),
            len(self.running),
            len(self.block_manager.used_block_ids),
        )
        self.metrics.observe_mlfq_queue_sizes(
            len(self._mlfq_queues[0]),
            len(self._mlfq_queues[1]),
            len(self._mlfq_queues[2]),
        )

        self._mlfq_epoch += 1
        self._mlfq_maybe_boost()

        served_ids: set[int] = set()

        for level in (0, 1, 2):
            if token_budget <= 0 or len(scheduled_seqs) >= self.max_num_seqs:
                break
            q = self._mlfq_queues[level]
            # Keep serving this level while eligible work remains and budget allows.
            # Bound iterations to avoid infinite loops on persistent alloc failure.
            safety = max(1, len(q)) * 3 + 8
            while (
                q
                and token_budget > 0
                and len(scheduled_seqs) < self.max_num_seqs
                and safety > 0
            ):
                safety -= 1
                # One RR pass: pop each current member once.
                n = len(q)
                progressed = False
                for _ in range(n):
                    if token_budget <= 0 or len(scheduled_seqs) >= self.max_num_seqs:
                        break
                    if not q:
                        break
                    seq = q.popleft()
                    if seq.is_finished:
                        continue
                    if seq.seq_id in served_ids:
                        # At most one service slice per request per schedule().
                        mlfq_lib.place_on_level(self._mlfq_queues, seq)
                        continue
                    if seq.mlfq_level != level:
                        mlfq_lib.place_on_level(self._mlfq_queues, seq)
                        continue
                    level_before = seq.mlfq_level
                    tokens, preempted, scheduled = self._mlfq_try_serve(
                        seq, token_budget, scheduled_seqs
                    )
                    if preempted:
                        preempted_this_step = True
                        num_preempted += preempted
                    if scheduled or tokens > 0:
                        token_budget -= tokens
                        served_ids.add(seq.seq_id)
                        progressed = True
                    elif seq.mlfq_level != level_before:
                        # Demoted without a token slice; do not re-serve this iteration.
                        served_ids.add(seq.seq_id)
                        progressed = True
                    # If not scheduled due to alloc failure, seq was re-queued at level.
                if not progressed:
                    break

        # Wait-step accounting for residents that received no service.
        for seq in list(self.waiting) + list(self.running):
            if seq.is_finished:
                continue
            if seq.seq_id not in served_ids:
                seq.mlfq_wait_steps += 1

        if not scheduled_seqs:
            msg = self._format_schedule_debug(
                token_budget=token_budget,
                scheduled_seqs=scheduled_seqs,
                preempted_this_step=preempted_this_step,
                num_preempted=num_preempted,
                num_skipped_waiting=0,
            )
            if self._debug_enabled():
                print(msg)
            raise RuntimeError(
                "scheduler produced an empty batch with no schedulable tokens; " + msg
            )

        assert sum(s.num_scheduled_tokens for s in scheduled_seqs) <= self.max_num_batched_tokens
        output = SchedulerOutput(scheduled_seqs)
        self._record_iteration_metrics(output)
        self._assert_model_runner_contract(output)
        return output

    def schedule(self) -> SchedulerOutput:
        if self.scheduler_policy == "mlfq":
            return self._schedule_mlfq()
        return self._schedule_baseline()

    def preempt(self, seq: Sequence):
        # INSTRUMENTATION-ONLY
        self.metrics.preemption_count += 1
        self.metrics.preempted_request_ids.add(seq.seq_id)
        recomputed = seq.num_cached_tokens
        self.metrics.recomputed_tokens += recomputed
        seq.num_preemptions += 1
        seq.num_recomputed_tokens += recomputed

        # Destructive recompute: discard KV and invalidate compute progress.
        # MLFQ: retain mlfq_level and mlfq_service_in_level (engine pressure must
        # not make a long job look like a fresh short job).
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.waiting_age_steps = 0
        self.block_manager.deallocate(seq)
        assert seq.num_computed_tokens == 0
        assert not seq.block_table
        if seq in self.running:
            self.running.remove(seq)
        self.waiting.appendleft(seq)
        if self.scheduler_policy == "mlfq":
            mlfq_lib.place_on_level(self._mlfq_queues, seq)

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
                if self.scheduler_policy == "mlfq":
                    self.metrics.record_completion_level(seq.mlfq_level)
                    mlfq_lib.detach_from_queues(self._mlfq_queues, seq)
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
