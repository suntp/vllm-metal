# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal n-gram speculative-decode proposer.

The proposer wraps vLLM's pure-Python/Numba n-gram kernel, so these tests need no
model or engine: a ``SimpleNamespace`` ``vllm_config`` exercises the upstream
constructor (which reads only scalar config) and a hand-built ``ProposeContext``
drives ``propose``. They lock in the request filtering (greedy-only, skip
intermediate prefills) and the array marshalling into the upstream kernel.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.spec_decode import ngram_proposer as upstream_ngram
from vllm.v1.spec_decode.ngram_proposer import NgramProposer as VllmNgramProposer

from vllm_metal.v1 import ngram_proposer as ngram_mod
from vllm_metal.v1.ngram_proposer import NgramProposer
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


def _vllm_config(
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    num_speculative_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            num_speculative_tokens=num_speculative_tokens,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


def _proposer(
    *,
    prompt_lookup_min: int = 2,
    prompt_lookup_max: int = 3,
    num_speculative_tokens: int = 3,
    max_model_len: int = 512,
    max_num_seqs: int = 4,
) -> NgramProposer:
    return NgramProposer(
        vllm_config=_vllm_config(
            prompt_lookup_min,
            prompt_lookup_max,
            num_speculative_tokens,
            max_model_len,
            max_num_seqs,
        ),
        controller=SpeculativeDecodeController(),
    )


def _request_state(
    token_ids: list[int], *, temperature: float = 0.0
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=temperature),
        generated_tokens=1,
    )


def _context(
    *,
    decode_reqs: list[tuple[str, SimpleNamespace]] | None = None,
    decode_token_ids: list[list[int]] | None = None,
    decode_segments: list[PagedDecodeSegment] | None = None,
    prefill_reqs: list[SimpleNamespace] | None = None,
    prefill_token_ids: list[int] | None = None,
    prefill_result_modes: list[str] | None = None,
    request_states: dict[str, SimpleNamespace] | None = None,
    num_speculative_tokens: int = 3,
    finished_req_ids: set[str] | None = None,
    draft_caps: dict[str, int] | None = None,
) -> ProposeContext:
    decode_reqs = decode_reqs or []
    prefill_reqs = prefill_reqs or []
    if decode_token_ids is None:
        decode_token_ids = [[state.token_ids[-1]] for _, state in decode_reqs]
    if prefill_result_modes is None:
        prefill_result_modes = ["new_final"] * len(prefill_reqs)
    if request_states is None:
        request_states = dict(decode_reqs)
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments or [],
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=(
            [0] * len(prefill_reqs) if prefill_token_ids is None else prefill_token_ids
        ),
        prefill_result_modes=prefill_result_modes,
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=len(decode_reqs),
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=finished_req_ids or set(),
        draft_caps=draft_caps,
    )


def _verified_segment(req_id: str, *, has_draft: bool = True) -> PagedDecodeSegment:
    """A decode_segment for req_id: a real (if trivial) verify row if
    has_draft, a plain no-draft row otherwise."""
    draft_token_ids = (0,) if has_draft else ()
    num_query_tokens = len(draft_token_ids) + 1
    return PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=tuple(range(num_query_tokens)),
        start_row=0,
        num_query_tokens=num_query_tokens,
        draft_token_ids=draft_token_ids,
        cache_start_pos=0,
        block_ids=(0,),
    )


class TestNgramProposerProtocol:
    def test_never_needs_target_hidden_states(self) -> None:
        proposer = _proposer()
        assert proposer.needs_target_hidden_states([], has_final_prefill=False) is False
        assert proposer.needs_target_hidden_states([], has_final_prefill=True) is False


class TestNgramProposePropose:
    def test_matches_repetitive_suffix_and_drafts_continuation(self) -> None:
        # Suffix [1, 2] recurs earlier; the tokens that followed it were 3, 1, 2.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(decode_reqs=[("r0", state)])

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3, 1, 2]]

    def test_uses_scheduler_selected_num_speculative_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            num_speculative_tokens=1,
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3]]

    def test_scheduler_selected_zero_tokens_returns_none(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            num_speculative_tokens=0,
        )

        assert proposer.propose(ctx) is None

    def test_no_match_returns_none(self) -> None:
        # Non-repeating context shorter than the n-gram window: no draft.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    @pytest.mark.parametrize("positional_cap,max_tokens", [(1, None), (1, 5), (3, 3)])
    def test_draft_caps_take_the_tighter_bound(
        self, positional_cap, max_tokens
    ) -> None:
        # The match yields three tokens. With one output already committed,
        # max_tokens=5/3 leaves a draft budget of 3/1, respectively.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        state.sampling_params.max_tokens = max_tokens
        ctx = _context(decode_reqs=[("r0", state)], draft_caps={"r0": positional_cap})

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3]]

    def test_draft_cap_to_empty_is_not_miss_evidence(self) -> None:
        # A draft trimmed away by a positional cap says nothing about
        # repetition exploitability: no miss streak, no pending record, and
        # the request drafts again once the cap lifts.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        capped = _context(decode_reqs=[("r0", state)], draft_caps={"r0": 0})

        for _ in range(2 * ngram_mod._MAX_CONSECUTIVE_MISSES):
            assert proposer.propose(capped) is None

        assert "r0" not in proposer._cooldown
        assert proposer._miss_streak.get("r0", 0) == 0
        assert "r0" not in proposer._pending

        uncapped = _context(decode_reqs=[("r0", state)])
        drafts = proposer.propose(uncapped)
        assert drafts is not None
        assert drafts.draft_token_ids == [[3, 1, 2]]

    @pytest.mark.parametrize("generated_tokens,expected_draft", [(2, [3]), (3, [])])
    def test_remaining_output_tokens_bound_the_draft(
        self, generated_tokens, expected_draft
    ) -> None:
        # max_tokens=4 with 2 generated leaves room for one draft plus the
        # bonus token; with 3 generated there is no room. Each case starts
        # fresh so a prior unverified proposal cannot leave pending state.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        state.sampling_params = SamplingParams(temperature=0.0, max_tokens=4)
        state.generated_tokens = generated_tokens

        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        if expected_draft:
            assert drafts is not None
            assert drafts.req_ids == ["r0"]
            assert drafts.draft_token_ids == [expected_draft]
        else:
            assert drafts is None
            assert "r0" not in proposer._pending
        assert proposer._miss_streak.get("r0", 0) == 0

    def test_empty_context_returns_none(self) -> None:
        assert _proposer().propose(_context()) is None

    def test_skips_request_without_sampled_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        # An empty sampled-ids entry marks a row that did not decode this step.
        ctx = _context(decode_reqs=[("r0", state)], decode_token_ids=[[]])

        assert proposer.propose(ctx) is None

    @pytest.mark.parametrize("temperature,max_tokens", [(0.8, 16), (0.0, 1)])
    def test_skips_ineligible_request(self, temperature, max_tokens) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2], temperature=temperature)
        state.sampling_params.max_tokens = max_tokens
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    @pytest.mark.parametrize("max_tokens,expected_draft", [(16, [6, 4, 5]), (2, [])])
    def test_finalized_prefill_respects_output_budget(
        self, max_tokens, expected_draft
    ) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        # The first output is already committed. max_tokens=2 still passes
        # eligibility but leaves no room for a draft plus a verifier token.
        state.sampling_params.max_tokens = max_tokens
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"p0": state},
        )

        drafts = proposer.propose(ctx)

        if expected_draft:
            assert drafts is not None
            assert drafts.req_ids == ["p0"]
            assert drafts.draft_token_ids == [expected_draft]
        else:
            assert drafts is None

    def test_skips_intermediate_prefill(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["intermediate"],
            request_states={"p0": state},
        )

        assert proposer.propose(ctx) is None

    def test_mixed_batch_drops_unmatched_keeps_matched(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        matched = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        unmatched = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", matched), ("r1", unmatched)])

        drafts = proposer.propose(ctx)

        # Only the matched request appears; the kernel returned [] for r1.
        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3, 1, 2]]

    def test_mixed_output_budgets_preserve_request_alignment(self) -> None:
        proposer = _proposer()
        short = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        last = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        full = _request_state([7, 8, 9, 7, 8, 9, 7, 8])
        short.sampling_params.max_tokens = 3  # Two output slots remain.
        last.sampling_params.max_tokens = 2  # Only one output slot remains.
        # Drop the middle row; distinct histories expose any shifted IDs.
        ctx = _context(decode_reqs=[("r0", short), ("r1", last), ("r2", full)])

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0", "r2"]
        assert drafts.draft_token_ids == [[3], [9, 7, 8]]

    def test_drops_prefill_already_seen_as_decode(self) -> None:
        # A request present in both decode and prefill lists must draft once.
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        prefill = SimpleNamespace(req_id="r0")
        ctx = _context(
            decode_reqs=[("r0", state)],
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"r0": state},
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]


class TestNgramMissThrottle:
    """The kernel scans a request's whole history every step whether or not
    it finds anything, so a request that never matches should stop being
    handed to the kernel after enough consecutive misses -- these tests
    mock the wrapped upstream call directly so they exercise only the
    throttle bookkeeping, independent of the installed vLLM's kernel
    signature."""

    def test_throttles_after_max_consecutive_misses(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                assert proposer.propose(ctx) is None
            calls_before_throttle = mock_propose.call_count

            # One more miss would be the (N+1)th in a row: by now the request
            # should be on cooldown and skipped before ever reaching the kernel.
            assert proposer.propose(ctx) is None
            assert mock_propose.call_count == calls_before_throttle

    def test_capped_streak_returns_to_cooldown_after_one_more_miss(self) -> None:
        """A retry that misses again right after cooldown must go straight
        back into cooldown, not get another full _MAX_CONSECUTIVE_MISSES-long
        grace period -- the streak caps at the threshold instead of clearing
        when cooldown is set."""
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
            assert "r0" in proposer._cooldown

            for _ in range(ngram_mod._COOLDOWN_STEPS):
                proposer.propose(ctx)
            calls_at_retry = mock_propose.call_count

            # The retry itself misses again: this single miss must be enough
            # to re-cooldown, not the first of another eight.
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_retry + 1
            assert "r0" in proposer._cooldown

            calls_after_recooldown = mock_propose.call_count
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_after_recooldown

    def test_real_acceptance_resets_the_streak(self) -> None:
        """A lookup match only counts once the *next* step's committed
        history shows the target actually accepted it -- not the moment the
        kernel proposes it."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([7, 8, 9])
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_verified_segment("r0")],
        )

        with patch.object(proposer._ngram, "propose") as mock_propose:
            mock_propose.return_value = [[]]
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES - 1):
                proposer.propose(ctx)
            assert "r0" not in proposer._cooldown

            # The kernel finds something...
            mock_propose.return_value = [[99]]
            drafts = proposer.propose(ctx)
            assert drafts is not None
            # ...and the engine genuinely accepts it (appends exactly what
            # was proposed to committed history).
            state.token_ids.append(99)

            # Resolving that acceptance must fully clear the streak, not
            # just discount it by one.
            mock_propose.return_value = [[]]
            proposer.propose(ctx)
            assert proposer._miss_streak.get("r0", 0) <= 1

    def test_lookup_match_without_acceptance_still_throttles(self) -> None:
        """The kernel finding *something* every step isn't evidence the
        request benefits -- if the target never actually accepts any of it,
        this must throttle exactly like true misses would."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_verified_segment("r0")],
        )

        # One resolved miss per call from the second call on (the first has
        # nothing pending yet to resolve), so the streak needs one extra
        # call beyond _MAX_CONSECUTIVE_MISSES to actually cross the
        # threshold and land in cooldown.
        with patch.object(proposer._ngram, "propose", return_value=[[99]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES + 1):
                proposer.propose(ctx)
                # The target never agrees: the engine appends some other
                # token instead of the proposed 99. (Once throttled, propose
                # stops even calling the kernel, so this is a no-op then.)
                state.token_ids.append(123)

        assert "r0" in proposer._cooldown

    def test_short_history_is_not_counted_as_a_miss(self) -> None:
        """Below prompt_lookup_min, the kernel structurally cannot match --
        that's not evidence against repetition, just not enough history yet,
        and must never accumulate toward throttling."""
        proposer = _proposer(prompt_lookup_min=4, prompt_lookup_max=5)
        state = _request_state([1, 2])  # shorter than prompt_lookup_min
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(2 * ngram_mod._MAX_CONSECUTIVE_MISSES):
                assert proposer.propose(ctx) is None

        assert "r0" not in proposer._cooldown
        assert proposer._miss_streak.get("r0", 0) == 0

    def test_cooldown_expires_and_retries(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
            calls_at_throttle = mock_propose.call_count

            # Every call during cooldown must be skipped before the kernel.
            for _ in range(ngram_mod._COOLDOWN_STEPS - 1):
                proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle

            # The cooldown-th call is the last skipped one; the call after
            # that must reach the kernel again.
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle + 1

    def test_prune_finished_clears_throttle_state(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # r0 finishes: its bookkeeping must not survive to be misread by a
        # later, unrelated request that happens to reuse the same id.
        proposer._prune_finished({"r0"})
        assert "r0" not in proposer._cooldown
        assert "r0" not in proposer._miss_streak

        with patch.object(
            proposer._ngram, "propose", return_value=[[1]]
        ) as mock_propose:
            drafts = proposer.propose(ctx)
        assert drafts is not None
        mock_propose.assert_called_once()

    def test_reused_request_id_does_not_inherit_old_throttle_state(self) -> None:
        """vLLM can hand a finished request's id straight to a brand-new
        request in the same scheduler step. The new request is present in
        request_states under that id from the moment it appears, so pruning
        on absence from request_states would never catch this -- pruning
        must key off the scheduler's own finished_req_ids instead."""
        proposer = _proposer()
        old_state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", old_state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # Same step: r0 finishes and a brand-new request reuses the id, so
        # it is simultaneously in finished_req_ids and in request_states.
        new_state = _request_state([1, 2, 3])
        new_ctx = _context(
            decode_reqs=[("r0", new_state)],
            finished_req_ids={"r0"},
        )

        with patch.object(
            proposer._ngram, "propose", return_value=[[42]]
        ) as mock_propose:
            drafts = proposer.propose(new_ctx)

        # The new request must be evaluated fresh, not skipped as if it
        # were still the old, throttled one.
        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        mock_propose.assert_called_once()

    def test_prune_and_resolve_run_even_when_speculative_tokens_disabled(
        self,
    ) -> None:
        """A step with drafting disabled (num_speculative_tokens <= 0) must
        still prune finished ids and resolve pending drafts -- that
        bookkeeping can't be gated behind the same early return that skips
        drafting, or a request id reused during a disabled step would carry
        stale throttle state into a later step where drafting resumes."""
        proposer = _proposer()
        old_state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", old_state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # Same id reused by a new request, but this step has speculative
        # decoding disabled.
        new_state = _request_state([1, 2, 3])
        disabled_ctx = _context(
            decode_reqs=[("r0", new_state)],
            finished_req_ids={"r0"},
            num_speculative_tokens=0,
        )
        assert proposer.propose(disabled_ctx) is None  # K=0, nothing drafted
        assert "r0" not in proposer._cooldown  # cleanup must still happen
        assert "r0" not in proposer._miss_streak
        assert "r0" not in proposer._row_of

        # A later, enabled step must evaluate the new request fresh.
        enabled_ctx = _context(decode_reqs=[("r0", new_state)])
        with patch.object(
            proposer._ngram, "propose", return_value=[[42]]
        ) as mock_propose:
            drafts = proposer.propose(enabled_ctx)
        assert drafts is not None
        mock_propose.assert_called_once()

    def test_pending_not_resolved_for_unscheduled_live_request(self) -> None:
        """A request can be alive in request_states without being scheduled
        for decode this step (still mid-prefill, paused, preempted). A
        pending draft for it must not be scored as a miss just because it
        wasn't verified this particular step."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[99]]):
            drafts = proposer.propose(ctx)
        assert drafts is not None
        assert "r0" in proposer._pending

        # Next step: r0 is still alive but not scheduled for decode.
        unscheduled_ctx = _context(
            decode_reqs=[],
            request_states={"r0": state},
        )
        proposer.propose(unscheduled_ctx)

        # The pending draft must survive untouched, not be scored as a miss.
        assert "r0" in proposer._pending
        assert proposer._miss_streak.get("r0", 0) == 0

    def test_pending_not_resolved_when_segment_carries_no_draft(self) -> None:
        """A request can be in decode_reqs with an empty decode_segment
        draft_token_ids -- no draft was scheduled for it at all, or the
        scheduler padded/dropped a proposed draft on batch admission (see
        SpeculativeDecodeController.active_spec_decode_tokens). Either way,
        an older pending draft for it must not be scored just because the
        request itself shows up in decode_reqs again."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3])
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_verified_segment("r0")],
        )

        with patch.object(proposer._ngram, "propose", side_effect=[[[99]], [[]]]):
            drafts = proposer.propose(ctx)
            assert drafts is not None
            assert proposer._pending["r0"] == (3, (99,))

            # Next step: r0 is still in decode_reqs, but this step's
            # segment carries no draft at all.
            state.token_ids.append(123)
            plain_ctx = _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_verified_segment("r0", has_draft=False)],
            )
            proposer.propose(plain_ctx)

        # The pending draft from the verified step must survive untouched.
        assert proposer._pending["r0"] == (3, (99,))

    def test_release_frees_rows_but_preserves_throttle_and_pending(self) -> None:
        proposer = _proposer(max_num_seqs=2)
        ctx = _context(decode_reqs=[("miss", _request_state([7, 8, 9]))])
        for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
            proposer.propose(ctx)
        matched = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        proposer.propose(_context(decode_reqs=[("match", matched)]))
        before = (
            dict(proposer._miss_streak),
            dict(proposer._cooldown),
            dict(proposer._pending),
        )
        assert all(before)

        # Reconciliation can release an ID at both preemption and resume.
        for _ in range(2):
            proposer.release_requests({"miss", "match", "unknown"})
        assert (proposer._miss_streak, proposer._cooldown, proposer._pending) == before
        assert not proposer._row_of
        assert sorted(proposer._free_rows) == [0, 1]
        assert all(source is None for source in proposer._row_source)

        with patch.object(proposer._ngram, "propose") as mock_propose:
            assert proposer.propose(ctx) is None
        mock_propose.assert_not_called()  # Resume must keep the cooldown.


def _full_copy_drafts(
    kernel: VllmNgramProposer,
    k: int,
    histories: dict[str, list[int]],
    max_model_len: int,
) -> dict[str, list[int]]:
    """The pre-incremental wrapper: zero a fresh buffer, copy every drafting
    request's whole history into row i, and call upstream with every row
    active."""
    req_ids = list(histories)
    token_ids_cpu = np.zeros((len(req_ids), max_model_len), dtype=np.int32)
    for i, req_id in enumerate(req_ids):
        token_ids_cpu[i, : len(histories[req_id])] = histories[req_id]
    num_tokens_no_spec = np.array(
        [len(histories[req_id]) for req_id in req_ids], dtype=np.int32
    )
    drafts = kernel.propose(k, [[0]] * len(req_ids), num_tokens_no_spec, token_ids_cpu)
    return {req_id: d for req_id, d in zip(req_ids, drafts, strict=True) if d}


def _v1_kernel_drafts(
    histories: dict[str, list[int]],
    *,
    min_n: int,
    max_n: int,
    k: int,
    max_model_len: int,
) -> dict[str, list[int]]:
    """Upstream V1's single-request kernel on each history, one at a time."""
    out: dict[str, list[int]] = {}
    for req_id, history in histories.items():
        if len(history) >= max_model_len:
            continue
        draft = upstream_ngram._find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=np.array(history, dtype=np.int32),
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )
        if len(draft):
            out[req_id] = draft.tolist()
    return out


def _proposed(drafts) -> dict[str, list[int]]:
    if drafts is None:
        return {}
    return dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))


class TestNgramIncrementalHistory:
    """Each request keeps a persistent row of token_ids_cpu that is appended
    to, not rewritten, every step. Drafts must stay token-identical to the
    old whole-history copy and to upstream's V1 kernel through everything
    that can invalidate a row: preemption with V2 readmission or V1 resume,
    finish and row reuse, multi-token acceptance, and paused requests."""

    @pytest.fixture(autouse=True)
    def _no_throttle(self, monkeypatch) -> None:
        # Compare every proposal, not just the ones the miss throttle lets
        # through; throttling has its own tests above.
        monkeypatch.setattr(ngram_mod, "_MAX_CONSECUTIVE_MISSES", 10**9)

    # Cross both k values with release modes; each case must cover every event.
    @pytest.mark.parametrize("seed,k", [(0, 3), (1, 5)])
    @pytest.mark.parametrize("release_on_invalidate", [True, False])
    def test_randomized_steps_match_full_copy_and_v1_kernel(
        self, k: int, release_on_invalidate: bool, seed: int
    ) -> None:
        min_n, max_n, max_model_len, max_num_seqs = 2, 3, 96, 6
        rng = random.Random(seed)
        proposer = _proposer(
            prompt_lookup_min=min_n,
            prompt_lookup_max=max_n,
            num_speculative_tokens=k,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
        reference = upstream_ngram.NgramProposer(
            _vllm_config(min_n, max_n, k, max_model_len, max_num_seqs)
        )
        vocab = 6  # small, so histories repeat and drafts are common
        running: dict[str, SimpleNamespace] = {}
        preempted: dict[str, SimpleNamespace] = {}
        last_drafts: dict[str, list[int]] = {}
        next_id = 0
        free_ids: list[str] = []
        num_compared = 0
        events: set[str] = set()

        def invalidate(req_ids: set[str]) -> None:
            if release_on_invalidate and req_ids:
                proposer.release_requests(req_ids)

        for _ in range(120):
            finished: set[str] = set()
            # Arrivals; finished ids are sometimes handed straight back out.
            while (
                len(running) + len(preempted) < 2 * max_num_seqs and rng.random() < 0.7
            ):
                if free_ids and rng.random() < 0.5:
                    req_id = free_ids.pop(rng.randrange(len(free_ids)))
                    events.add("reuse")
                else:
                    req_id, next_id = f"r{next_id}", next_id + 1
                prompt = [rng.randrange(vocab) for _ in range(rng.randint(1, 40))]
                running[req_id] = _request_state(prompt)
            # Commit this step's tokens: an accepted draft prefix plus one
            # verifier token (rejected draft tokens are never committed),
            # or a plain decode token.
            for req_id, state in running.items():
                draft = last_drafts.get(req_id, [])
                accepted = draft[: rng.randint(0, len(draft))] if draft else []
                if draft:
                    events.add("accept" if accepted else "reject")
                    if len(accepted) > 1:
                        events.add("multi_accept")
                    if accepted and len(accepted) < len(draft):
                        events.add("partial_accept")
                new = [*accepted, rng.randrange(vocab)]
                room = max_model_len - len(state.token_ids)
                state.token_ids.extend(new[: max(room, 0)])
            # Lifecycle events.
            for req_id in list(running):
                state = running[req_id]
                roll = rng.random()
                if roll < 0.05 or len(state.token_ids) >= max_model_len:
                    del running[req_id]
                    finished.add(req_id)
                    free_ids.append(req_id)
                    events.add("finish")  # Exercise _prune_finished itself.
                elif roll < 0.12:
                    preempted[req_id] = running.pop(req_id)
                    invalidate({req_id})
                    events.add("preempt")
                elif roll < 0.16 and len(state.token_ids) > 1:
                    del state.token_ids[-rng.randint(1, len(state.token_ids) - 1) :]
                    events.add("shrink")
                elif roll < 0.20:
                    state.token_ids = [rng.randrange(vocab) for _ in state.token_ids]
                    events.add("rewrite")
            for req_id in list(preempted):
                if rng.random() < 0.3:
                    old = preempted.pop(req_id)
                    if rng.random() < 0.5:
                        # V2: a fresh RequestState built from the prefill ids.
                        state = _request_state(list(old.token_ids))
                        events.add("readmit")
                    else:
                        # V1 resume: same state, list replaced by the
                        # scheduler's copy, possibly without an in-flight
                        # token the runner had already appended.
                        state = old
                        drop = rng.randint(0, min(1, len(old.token_ids) - 1))
                        state.token_ids = list(
                            old.token_ids[: len(old.token_ids) - drop]
                        )
                        events.add("resume")
                    running[req_id] = state
                    invalidate({req_id})
            # Some running requests sit this step out (paused/unscheduled).
            scheduled = [
                (req_id, state)
                for req_id, state in running.items()
                if rng.random() < 0.85
            ]
            rng.shuffle(scheduled)
            scheduled = scheduled[:max_num_seqs]
            if len(scheduled) < len(running):
                events.add("pause")
            num_tokens = rng.randint(1, k)
            ctx = _context(
                decode_reqs=scheduled,
                request_states=dict(running),
                num_speculative_tokens=num_tokens,
                finished_req_ids=finished,
            )

            histories = {req_id: list(state.token_ids) for req_id, state in scheduled}
            resident = set(proposer._row_of) - finished
            got = _proposed(proposer.propose(ctx))
            # These owners cannot all fit, even after pruning finished ids.
            if len(resident | histories.keys()) > max_num_seqs:
                assert resident - proposer._row_of.keys()
                events.add("capacity")
            want = _full_copy_drafts(reference, num_tokens, histories, max_model_len)
            v1 = _v1_kernel_drafts(
                histories,
                min_n=min_n,
                max_n=max_n,
                k=num_tokens,
                max_model_len=max_model_len,
            )
            assert got == want == v1
            assert list(got) == list(want)
            assert len(proposer._row_of) + len(proposer._free_rows) == max_num_seqs
            assert len(set(proposer._row_of.values())) == len(proposer._row_of)
            assert set(proposer._row_of.values()).isdisjoint(proposer._free_rows)
            for req_id, state in scheduled:
                assert state.token_ids == histories[req_id]
                row = proposer._row_of[req_id]
                assert proposer._row_len[row] == len(state.token_ids)
                assert (
                    proposer._token_ids_cpu[row, : len(state.token_ids)].tolist()
                    == state.token_ids
                )
            num_compared += len(histories)
            last_drafts = got

        assert num_compared > 200
        assert events == {
            "reuse",
            "accept",
            "reject",
            "finish",
            "preempt",
            "shrink",
            "rewrite",
            "readmit",
            "resume",
            "pause",
            "multi_accept",
            "partial_accept",
            "capacity",
        }

    def test_only_new_tokens_are_copied(self) -> None:
        proposer = _proposer()
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        row = proposer._row_of["r0"]
        # Poison an already-synced position that is not the last one: an
        # incremental append must leave it alone.
        proposer._token_ids_cpu[row, 0] = 777
        proposer._token_ids_cpu[row, 10:] = 888
        state.token_ids.extend([3, 1])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert proposer._token_ids_cpu[row, 0] == 777
        assert proposer._token_ids_cpu[row, 8:10].tolist() == [3, 1]
        assert proposer._row_len[row] == 10
        assert np.all(proposer._token_ids_cpu[row, 10:] == 888)

    def test_single_request_mock_after_batch_uses_one_row_view(self) -> None:
        proposer = _proposer()
        states = [(f"r{i}", _request_state([i, 1, 2, i, 1, 2, i, 1])) for i in range(3)]
        proposer.propose(_context(decode_reqs=states))
        with patch.object(
            proposer._ngram, "propose", return_value=[[99]]
        ) as mock_propose:
            drafts = proposer.propose(_context(decode_reqs=states[-1:]))
        assert _proposed(drafts) == {"r2": [99]}
        k, sampled, lengths, tokens = mock_propose.call_args.args
        assert k == 3 and sampled == [[0]] and lengths.tolist() == [8]
        assert tokens.shape == (1, 512)
        assert np.shares_memory(tokens, proposer._token_ids_cpu)
        assert tokens[0, :8].tolist() == states[-1][1].token_ids

    def test_reordered_sparse_batch_skips_inactive_rows(self) -> None:
        proposer = _proposer()
        states = [(f"r{i}", _request_state([i, 1, 2, i, 1, 2, i, 1])) for i in range(4)]
        proposer.propose(_context(decode_reqs=states))
        scheduled = [states[3], states[1]]
        with patch.object(
            proposer._ngram, "propose", wraps=proposer._ngram.propose
        ) as mock_propose:
            drafts = proposer.propose(_context(decode_reqs=scheduled))
        assert drafts is not None
        assert drafts.req_ids == ["r3", "r1"]
        assert drafts.draft_token_ids == [[2, 3, 1], [2, 1, 1]]
        _, sampled, lengths, tokens = mock_propose.call_args.args
        assert sampled == [[0], [], [0]]
        assert lengths.tolist() == [8, 0, 8]
        assert np.shares_memory(tokens, proposer._token_ids_cpu)

    @pytest.mark.parametrize("mode", ["decode_stop_before_bonus", "prefill_eos"])
    def test_terminal_tokens_skip_resident_rows(self, mode: str) -> None:
        proposer = _proposer()
        states = [(f"r{i}", _request_state([1, 2, 3, 1, 2, 3, 1, 2])) for i in range(3)]
        terminal = states[1][1]
        terminal.sampling_params = SamplingParams(temperature=0, stop_token_ids=[7])
        terminal.sampling_params.update_from_generation_config({}, eos_token_id=9)
        assert set(_proposed(proposer.propose(_context(decode_reqs=states)))) == {
            "r0",
            "r1",
            "r2",
        }

        if mode == "decode_stop_before_bonus":
            terminal.token_ids.extend([7, 2])
            ctx = _context(decode_reqs=states, decode_token_ids=[[2], [7, 2], [2]])
        else:
            # Eligibility uses the current prefill token, even before history updates.
            ctx = _context(
                decode_reqs=[states[0], states[2]],
                prefill_reqs=[SimpleNamespace(req_id="r1")],
                prefill_token_ids=[9],
                request_states=dict(states),
            )

        assert _proposed(proposer.propose(ctx)) == {"r0": [3, 1, 2], "r2": [3, 1, 2]}

    @pytest.mark.parametrize("change", ["replace_list", "shrink", "rewrite"])
    def test_invalidated_history_resyncs(self, change: str) -> None:
        proposer = _proposer()
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        assert _proposed(proposer.propose(_context(decode_reqs=[("r0", state)]))) == {
            "r0": [3, 1, 2]
        }
        # Same boundary token: checking only the last token misses the rewrite.
        replacement = [4, 2, 6, 4, 2, 6, 4, 2]
        if change == "replace_list":
            # V1 resume and V2 readmission both change token-list identity.
            state.token_ids = replacement
        elif change == "shrink":
            state.token_ids[:] = replacement[:5]
        else:
            # In-place rewrites require the runner's lifecycle invalidation;
            # ordinary committed histories only grow by append/extend.
            proposer.release_requests({"r0"})
            state.token_ids[:] = replacement
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        histories = {"r0": state.token_ids}
        expected = _full_copy_drafts(proposer._ngram, 3, histories, 512)
        assert expected == _v1_kernel_drafts(
            histories, min_n=2, max_n=3, k=3, max_model_len=512
        )
        assert expected and _proposed(drafts) == expected
        row = proposer._row_of["r0"]
        assert (
            proposer._token_ids_cpu[row, : len(state.token_ids)].tolist()
            == state.token_ids
        )

    def test_rewritten_last_token_resyncs(self) -> None:
        proposer = _proposer()
        state = _request_state([2, 3, 1, 2, 3, 1, 2, 9])
        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
        # Same list object, same length, last committed token rewritten.
        state.token_ids[-1] = 3
        drafts = _proposed(proposer.propose(_context(decode_reqs=[("r0", state)])))
        expected = _v1_kernel_drafts(
            {"r0": state.token_ids}, min_n=2, max_n=3, k=3, max_model_len=512
        )
        assert expected and drafts == expected

    @pytest.mark.parametrize("new_id", ["r0", "r1"])
    def test_finished_row_is_reused_without_stale_tail(self, new_id: str) -> None:
        proposer = _proposer()
        long_state = _request_state([9, 8, 9, 8, 9, 8, 9, 8, 7, 7, 7, 7])
        proposer.propose(_context(decode_reqs=[("r0", long_state)]))
        row = proposer._row_of["r0"]
        # r0 finishes; a new request takes its row with a shorter history.
        short_state = _request_state([1, 2, 1, 2])
        drafts = proposer.propose(
            _context(decode_reqs=[(new_id, short_state)], finished_req_ids={"r0"})
        )
        assert proposer._row_of == {new_id: row}
        expected = _v1_kernel_drafts(
            {new_id: short_state.token_ids}, min_n=2, max_n=3, k=3, max_model_len=512
        )
        assert expected and _proposed(drafts) == expected

    def test_rows_reclaimed_when_live_requests_exceed_capacity(self) -> None:
        proposer = _proposer(max_num_seqs=2)
        states = {f"r{i}": _request_state([i, 1, 2, i, 1, 2, i, 1]) for i in range(3)}
        # Three live requests, never more than two drafting in one step.
        for pair in (("r0", "r1"), ("r2", "r0"), ("r1", "r2"), ("r0", "r1")):
            retained = {r: proposer._row_of[r] for r in pair if r in proposer._row_of}
            ctx = _context(decode_reqs=[(r, states[r]) for r in pair])
            assert _proposed(proposer.propose(ctx)) == {
                r: [2, int(r[1]), 1] for r in pair
            }
            # A newcomer must not evict an owner later in drafting order.
            assert all(proposer._row_of[r] == row for r, row in retained.items())
        assert len(proposer._row_of) == 2

    def test_over_capacity_batch_fails_before_reassigning_rows(self) -> None:
        proposer = _proposer(max_num_seqs=1)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        with pytest.raises(
            RuntimeError, match="more drafting requests than max_num_seqs=1"
        ):
            proposer.propose(_context(decode_reqs=[("r1", state), ("r2", state)]))
        assert proposer._row_of == {"r0": 0}
        assert _proposed(proposer.propose(_context(decode_reqs=[("r0", state)]))) == {
            "r0": [3, 1, 2]
        }

    @pytest.mark.parametrize("length", [7, 8, 9])
    def test_history_near_max_model_len_is_bounded(self, length: int) -> None:
        proposer = _proposer(max_model_len=8)
        state = _request_state(([1, 2, 3] * 4)[:length])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        expected = _v1_kernel_drafts(
            {"r0": state.token_ids}, min_n=2, max_n=3, k=3, max_model_len=8
        )
        assert _proposed(drafts) == expected
        assert proposer._row_len[proposer._row_of["r0"]] == min(length, 8)
        # Shrinking below the limit must invalidate a previously capped row.
        state.token_ids[:] = [1, 2, 1, 2]
        assert _proposed(proposer.propose(_context(decode_reqs=[("r0", state)]))) == {
            "r0": [1, 2]
        }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
