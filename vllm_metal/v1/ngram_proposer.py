# SPDX-License-Identifier: Apache-2.0
"""N-gram (prompt-lookup) speculative decoding proposer for the Metal paged path.

An :class:`NgramProposer` drafts by matching the longest suffix n-gram of each
request's committed token history against an earlier occurrence and copying the
tokens that followed it (vLLM ``method="ngram"``). Unlike
:class:`vllm_metal.v1.draft_model_proposer.DraftModelProposer` it loads no model
and keeps no KV cache: the matching is the pure-Python + Numba KMP kernel that
vLLM ships in :mod:`vllm.v1.spec_decode.ngram_proposer`, which this class wraps.

The wrapper translates the per-step :class:`ProposeContext` into the runtime
draft count and array arguments that upstream's stateless ``propose`` expects
(``num_speculative_tokens``, ``sampled_token_ids``, ``num_tokens_no_spec``,
``token_ids_cpu``) and hands the result back as :class:`DraftTokenIds`. The
committed history lives in ``state.token_ids`` (already updated with this
step's accepted/sampled tokens by the time the runner builds the context).

Like upstream V1's persistent ``InputBatch.token_ids_cpu``, each resident
request owns a fixed row of the preallocated ``token_ids_cpu`` buffer, and each
step copies only the tokens committed since that row was last synced, instead
of an O(batch x history) host copy every step. A row is rebuilt from scratch
whenever it may no longer mirror the request (new request, preempt/resume, V2
readmission, or a rewritten history list) and is handed back by
``release_requests`` and on finish.

The wrapper also keeps a consecutive-miss streak per request, so a request
with no exploitable repetition (free-form prose, for instance) stops paying
the match kernel's per-step scan cost after a few misses in a row, with
periodic retries in case the content turns repetitive later. See
``_record_miss``/``_on_cooldown``.

A lookup match is not itself evidence the request benefits: the target may
reject every proposed token. ``_resolve_pending`` checks a step's proposal
against the *next* step's committed history (which already reflects what the
target actually accepted) to score it as a real hit or a miss, rather than
trusting the lookup kernel's own optimism.

The verify half is unchanged: drafts are handed back via ``take_draft_token_ids``
and verified next step by ``SpeculativeDecodeController.verify_greedy``.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer as VllmNgramProposer

from vllm_metal.v1.proposer import NoOwnKVProposer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig

    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)

# The match kernel scans a request's whole committed history every decode
# step whether or not it finds anything -- an O(history length) Numba scan,
# paid regardless of outcome.
# A request whose content has no exploitable repetition (free-form prose,
# for instance) pays that tax every step for nothing. After this many
# consecutive misses, stop attempting a request for _COOLDOWN_STEPS steps
# rather than giving up on it forever -- generation can turn repetitive
# partway through (e.g. a response that starts as prose and then quotes
# earlier context), so a permanent cutoff would miss that.
_MAX_CONSECUTIVE_MISSES = 8
_COOLDOWN_STEPS = 8


class NgramProposer(NoOwnKVProposer):
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by vLLM's n-gram kernel."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        # Per-request consecutive-miss count and, once throttled, remaining
        # cooldown steps before the next retry. Pruned each step against the
        # scheduler's own finished-id set, not against absence from
        # request_states: vLLM can hand a finished id straight back out to a
        # new request in the same step, and that new request repopulates
        # request_states under the same id before this wrapper ever sees it.
        self._miss_streak: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        # A lookup match doesn't mean the target accepted any of it. Each
        # non-empty draft is stashed here (position it starts at, tokens
        # proposed) and scored against the *next* step's actual committed
        # history in _resolve_pending, before this step's proposal is
        # scored as a hit or a miss.
        self._pending: dict[str, tuple[int, tuple[int, ...]]] = {}
        # Upstream reads only scalar config (prompt_lookup_min/max,
        # num_speculative_tokens, max_model_len, max_num_seqs) and runs a one-time
        # Numba JIT warmup in its constructor — keep that off the hot path.
        self._ngram = VllmNgramProposer(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None
        # Below this many committed tokens, the kernel structurally cannot
        # match (not enough history to form even the shortest n-gram window)
        # -- that's not evidence against repetition, just not enough data
        # yet, so it must not count as a miss. Normalized to a concrete int
        # by SpeculativeConfig.__post_init__ for method="ngram".
        assert spec.prompt_lookup_min is not None
        self._prompt_lookup_min = spec.prompt_lookup_min

        # Pre-allocate the int32 token-id buffer once: one row per request
        # that can be scheduled at the same time, each long enough for the
        # longest committed history. Upstream only reads
        # ``token_ids_cpu[i, :num_tokens_no_spec[i]]``, so a row's tail past
        # its synced length is never read and never needs clearing.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        self._max_model_len = max_model_len
        self._token_ids_cpu = np.zeros((max_num_seqs, max_model_len), dtype=np.int32)
        # Row ownership. ``_row_source[row]`` is the exact ``state.token_ids``
        # list the row mirrors; ``_row_len[row]`` is how many of its leading
        # tokens are already in the buffer (capped at max_model_len). The
        # runner replaces the list object on V1 resume and builds a new
        # RequestState on V2 readmission, so an identity mismatch alone
        # forces a full resync even if release_requests was skipped.
        self._row_of: dict[str, int] = {}
        self._row_source: list[list[int] | None] = [None] * max_num_seqs
        self._row_len = np.zeros(max_num_seqs, dtype=np.int32)
        # Prefer low rows to keep the span handed to the kernel small.
        self._free_rows: list[int] = list(range(max_num_seqs))
        logger.info(
            "N-gram speculative decoding enabled "
            "(prompt_lookup=[%d, %d], num_speculative_tokens=%d, "
            "token_ids_cpu=(%d, %d) (%.2f MiB))",
            spec.prompt_lookup_min,
            spec.prompt_lookup_max,
            spec.num_speculative_tokens,
            max_num_seqs,
            max_model_len,
            self._token_ids_cpu.nbytes / (1024 * 1024),
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> NgramProposer:
        return cls(vllm_config=vllm_config, controller=controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # N-gram matches token ids only; it never reads the target's hidden states.
        return False

    def release_requests(self, req_ids: set[str]) -> None:
        # Called for evicted, preempted and resumed requests. Their history
        # row is a bounded resource and its contents may no longer match
        # what the request resumes with, so hand it back; the next draft
        # rebuilds it from the request's current history. The miss-streak
        # throttle is per-request bookkeeping, not a bounded resource: a
        # preempted request keeps its counters across resume, and they are
        # dropped when its id finishes.
        for req_id in req_ids:
            self._release_row(req_id)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        # Bookkeeping runs unconditionally, before the num_speculative_tokens
        # check: a step with drafting disabled still needs finished ids
        # pruned and pending drafts resolved, or a request id reused in a
        # disabled step would carry stale throttle state into the next step
        # where drafting is enabled again.
        self._prune_finished(ctx.finished_req_ids)
        self._resolve_pending(ctx)

        if ctx.num_speculative_tokens <= 0:
            return None

        drafting = list(
            self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_token_ids,
                ctx.prefill_result_modes,
                ctx.request_states,
            )
        )
        if not drafting:
            return None

        drafting = [
            (req_id, state)
            for req_id, state in drafting
            if not self._on_cooldown(req_id)
        ]
        if not drafting:
            return None

        if len(drafting) > len(self._row_source):
            raise RuntimeError(
                "n-gram proposer: more drafting requests than "
                f"max_num_seqs={len(self._row_source)} history rows"
            )

        # Protect all drafting owners before allocating: a new request may
        # precede an existing owner in drafting order.
        in_use = {
            self._row_of[req_id] for req_id, _ in drafting if req_id in self._row_of
        }
        rows = [
            self._sync_row(req_id, state.token_ids, in_use)
            for req_id, state in drafting
        ]
        # Hand upstream a view spanning the active rows, without gathering
        # (and copying) their histories. In particular, a single drafting
        # request always gets a one-row view and a one-element result.
        # Upstream marks a row "active" by a non-empty sampled-ids entry and
        # reads only token_ids_cpu[i, :num_tokens_no_spec[i]], so rows in the
        # span owned by requests not drafting this step (or free) are skipped
        # and return [].
        first_row, last_row = min(rows), max(rows) + 1
        relative_rows = [row - first_row for row in rows]
        num_rows = last_row - first_row
        num_tokens_no_spec = np.zeros(num_rows, dtype=np.int32)
        num_tokens_no_spec[relative_rows] = self._row_len[rows]
        sampled_token_ids: list[list[int]] = [[]] * num_rows
        for row in relative_rows:
            sampled_token_ids[row] = [0]

        row_drafts = self._ngram.propose(
            ctx.num_speculative_tokens,
            sampled_token_ids,
            num_tokens_no_spec,
            self._token_ids_cpu[first_row:last_row],
        )
        drafts = [row_drafts[row] for row in relative_rows]

        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        for (req_id, state), draft in zip(drafting, drafts, strict=True):
            cap = ctx.draft_caps.get(req_id) if ctx.draft_caps is not None else None
            max_tokens = state.sampling_params.max_tokens
            if max_tokens is not None:
                # Verification also emits a correction or bonus token (#1078).
                remaining = max(0, max_tokens - state.generated_tokens - 1)
                cap = remaining if cap is None else min(cap, remaining)
            if cap is not None and len(draft) > cap:
                draft = draft[:cap]
                if not draft:
                    # Trimmed by a positional or length cap, not by missing
                    # history: not miss evidence for the cooldown.
                    continue
            if not draft:
                # A too-short history isn't evidence against repetition --
                # there just isn't enough of it yet to check.
                if len(state.token_ids) >= self._prompt_lookup_min:
                    self._record_miss(req_id)
                continue
            # Don't score this as a hit yet -- the target hasn't verified it.
            # _resolve_pending scores it next step against what was actually
            # accepted.
            self._pending[req_id] = (len(state.token_ids), tuple(draft))
            req_ids.append(req_id)
            # Upstream already yields Python ints via ndarray.tolist() — the
            # old ``[int(t) for t in draft]`` was redundant.
            draft_token_ids.append(list(draft))

        if not req_ids:
            return None

        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)

    # -- per-request history rows --------------------------------------------

    def _sync_row(self, req_id: str, token_ids: list[int], in_use: set[int]) -> int:
        """Return ``req_id``'s row with ``token_ids`` mirrored into it.

        Resume/readmission replaces the list or releases the row. Otherwise
        committed history grows only by append/extend: speculative decoding
        disables pipeline placeholder backfill and never commits rejected
        drafts. Also resync on shrink or a changed boundary token; arbitrary
        in-place prefix rewrites must invalidate the row via release_requests.
        """
        row = self._row_of.get(req_id)
        if row is None:
            row = self._acquire_row(req_id, in_use)
            synced = 0
        elif self._row_source[row] is not token_ids:
            synced = 0
        else:
            synced = int(self._row_len[row])
            if synced and (
                len(token_ids) < synced
                or self._token_ids_cpu[row, synced - 1] != token_ids[synced - 1]
            ):
                synced = 0
        # Upstream never drafts for a history already at max_model_len, so
        # the buffer needs at most that many tokens.
        num_tokens = min(len(token_ids), self._max_model_len)
        if num_tokens > synced:
            self._token_ids_cpu[row, synced:num_tokens] = token_ids[synced:num_tokens]
        self._row_source[row] = token_ids
        self._row_len[row] = num_tokens
        in_use.add(row)
        return row

    def _acquire_row(self, req_id: str, in_use: set[int]) -> int:
        if not self._free_rows:
            # Every row is owned. Rows are released on finish, preempt and
            # resume, so this means more live requests than max_num_seqs;
            # reclaim one that is not drafting this step -- its owner gets a
            # full resync if it drafts again.
            victim = next(
                (rid for rid, r in self._row_of.items() if r not in in_use), None
            )
            if victim is None:
                raise RuntimeError(
                    "n-gram proposer: more drafting requests than "
                    f"max_num_seqs={len(self._row_source)} history rows"
                )
            self._release_row(victim)
        row = heapq.heappop(self._free_rows)
        self._row_of[req_id] = row
        return row

    def _release_row(self, req_id: str) -> None:
        row = self._row_of.pop(req_id, None)
        if row is None:
            return
        self._row_source[row] = None
        self._row_len[row] = 0
        heapq.heappush(self._free_rows, row)

    # -- miss-streak throttling ----------------------------------------------

    def _on_cooldown(self, req_id: str) -> bool:
        remaining = self._cooldown.get(req_id, 0)
        if remaining <= 0:
            return False
        if remaining == 1:
            del self._cooldown[req_id]
        else:
            self._cooldown[req_id] = remaining - 1
        return True

    def _record_miss(self, req_id: str) -> None:
        # Cap rather than clear: once a request has earned a cooldown, a
        # miss on the very next retry should send it right back into
        # cooldown, not grant another full _MAX_CONSECUTIVE_MISSES-long
        # grace period. Only a genuine hit (_resolve_pending finding real
        # acceptance) fully clears the streak.
        streak = min(self._miss_streak.get(req_id, 0) + 1, _MAX_CONSECUTIVE_MISSES)
        self._miss_streak[req_id] = streak
        if streak >= _MAX_CONSECUTIVE_MISSES:
            self._cooldown[req_id] = _COOLDOWN_STEPS

    def _resolve_pending(self, ctx: ProposeContext) -> None:
        """Score the previous step's proposals against what was actually
        accepted, but only for requests whose draft was actually verified
        this step: present in ``decode_reqs`` *and* the decode_segment for
        that request carried a non-empty draft. Being in ``decode_reqs``
        alone isn't enough -- a request can sit there with an empty
        ``draft_token_ids`` either because no draft was scheduled for it at
        all (a plain decode row) or because the scheduler padded/dropped a
        proposed draft on batch admission (see
        ``SpeculativeDecodeController.active_spec_decode_tokens``). Either
        way, this step's plain-decode token isn't the verification outcome
        of the older pending draft, so it must not be scored against it."""
        if not self._pending:
            return
        verified_req_ids = {
            segment.req_id for segment in ctx.decode_segments if segment.draft_token_ids
        }
        decode_states = dict(ctx.decode_reqs)
        for req_id in list(self._pending):
            if req_id not in verified_req_ids:
                continue  # not verified this step; leave pending for later
            state = decode_states.get(req_id)
            if state is None:
                continue
            position, proposed = self._pending.pop(req_id)
            actual = state.token_ids[position : position + len(proposed)]
            accepted = 0
            # actual may be shorter than proposed if the engine hasn't
            # committed that many new tokens yet -- strict=False by design.
            for proposed_id, actual_id in zip(proposed, actual, strict=False):
                if proposed_id != actual_id:
                    break
                accepted += 1
            if accepted > 0:
                self._miss_streak.pop(req_id, None)
            else:
                self._record_miss(req_id)

    def _prune_finished(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self._miss_streak.pop(req_id, None)
            self._cooldown.pop(req_id, None)
            self._pending.pop(req_id, None)
            self._release_row(req_id)
