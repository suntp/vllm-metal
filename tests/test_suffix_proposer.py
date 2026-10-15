# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal suffix-decoding proposer.

The proposer wraps vLLM's ``SuffixDecodingProposer``, whose engine-side loop
runs entirely off an InputBatch-shaped stub, so most tests drive
``propose`` against a recording fake of that upstream class (no
``arctic_inference`` install needed) and lock in the marshalling: stub
fields, per-request deltas, non-drafting rows kept alive, preemption
rollback, cap trimming, and bookkeeping lifecycle. One integration test
exercises the real arctic suffix tree when the package is importable.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams

from tests.test_ngram_spec_decode import _context
from vllm_metal.v1 import suffix_proposer as suffix_mod
from vllm_metal.v1.spec_decode import SpeculativeDecodeController
from vllm_metal.v1.suffix_proposer import SuffixProposer


class _FakeVllmSuffix:
    """Records upstream-facing calls; drafts mirror the fed deltas."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, object, list[list[int]]]] = []
        self.extra_draft: dict[int, list[int]] = {}

    def propose(self, num_speculative_tokens, input_batch, sampled_token_ids):
        self.calls.append(
            (num_speculative_tokens, input_batch, [list(s) for s in sampled_token_ids])
        )
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            drafts.append(self.extra_draft.get(i, sampled))
        return drafts


def _request_state(
    token_ids: list[int],
    *,
    prompt_len: int,
    temperature: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=list(token_ids),
        prompt_len=prompt_len,
        sampling_params=SamplingParams(temperature=temperature),
        generated_tokens=1,
    )


def _proposer(
    fake: _FakeVllmSuffix, *, num_speculative_tokens: int = 3
) -> SuffixProposer:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="suffix",
            num_speculative_tokens=num_speculative_tokens,
            suffix_decoding_max_tree_depth=24,
            suffix_decoding_max_spec_factor=1.0,
            suffix_decoding_min_token_prob=0.1,
            suffix_decoding_max_cached_requests=10000,
        ),
        model_config=SimpleNamespace(max_model_len=512),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    with patch.object(suffix_mod, "_build_vllm_suffix", return_value=fake):
        return SuffixProposer.build(
            vllm_config=vllm_config,
            controller=SpeculativeDecodeController(),
        )


class TestSuffixMarshalling:
    def test_stub_carries_prompt_split_and_history(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state = _request_state([5, 6, 7, 8, 9, 10, 11], prompt_len=4)

        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert drafts is not None
        num_spec, stub, sampled = fake.calls[0]
        assert num_spec == 3  # the configured width, satisfying upstream's assert
        assert stub.req_ids == ["r0"]
        assert stub.req_id_to_index == {"r0": 0}
        assert stub.num_tokens_no_spec.tolist() == [7]
        assert stub.num_prompt_tokens.tolist() == [4]
        assert stub.token_ids_cpu[0, :7].tolist() == [5, 6, 7, 8, 9, 10, 11]
        # First sighting: the delta is everything past the prompt.
        assert sampled == [[9, 10, 11]]
        # The fake echoes the delta back as the draft.
        assert drafts.draft_token_ids == [[9, 10, 11]]

    def test_second_step_feeds_only_the_delta(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state0 = _request_state([1, 2, 3, 4, 5], prompt_len=3)
        proposer.propose(_context(decode_reqs=[("r0", state0)]))

        state1 = _request_state([1, 2, 3, 4, 5, 6, 7], prompt_len=3)
        proposer.propose(_context(decode_reqs=[("r0", state1)]))

        _, _, sampled = fake.calls[1]
        assert sampled == [[6, 7]]

    def test_nondrafting_rows_stay_in_the_stub_but_get_no_draft(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        greedy = _request_state([1, 2, 3, 4], prompt_len=2)
        sampled_req = _request_state([5, 6, 7, 8], prompt_len=2, temperature=1.0)

        drafts = proposer.propose(
            _context(decode_reqs=[("greedy", greedy), ("sampled", sampled_req)])
        )

        _, stub, sampled = fake.calls[0]
        # Both live requests are rows (keeps the sampled request's tree from
        # being archived), but only the greedy row carries a delta.
        assert stub.req_ids == ["greedy", "sampled"]
        assert sampled == [[3, 4], []]
        assert drafts is not None
        assert drafts.req_ids == ["greedy"]
        assert drafts.draft_token_ids == [[3, 4]]

    def test_preemption_rollback_clamps_the_delta(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state0 = _request_state([1, 2, 3, 4, 5, 6], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state0)]))

        # Preempted and recomputed shorter than what was already fed: the
        # anchor resets and the delta is empty rather than negative.
        state1 = _request_state([1, 2, 3, 4], prompt_len=2)
        drafts = proposer.propose(_context(decode_reqs=[("r0", state1)]))
        _, _, sampled = fake.calls[1]
        assert sampled == [[]]
        assert drafts is None

    def test_draft_caps_trim(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3, 4, 5], prompt_len=2)

        ctx = replace(_context(decode_reqs=[("r0", state)]), draft_caps={"r0": 1})
        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert len(drafts.draft_token_ids[0]) == 1

    def test_finished_ids_and_release_clear_bookkeeping(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state0 = _request_state([1, 2, 3, 4, 5], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state0)]))
        assert proposer._last_len == {"r0": 5}

        proposer.propose(_context(decode_reqs=[], finished_req_ids={"r0"}))
        assert "r0" not in proposer._last_len

        proposer.propose(_context(decode_reqs=[("r1", state0)]))
        proposer.release_requests({"r1"})
        assert "r1" not in proposer._last_len

    def test_disabled_or_empty_drafting_returns_none(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3], prompt_len=1)

        assert (
            proposer.propose(
                _context(decode_reqs=[("r0", state)], num_speculative_tokens=0)
            )
            is None
        )
        # Nothing eligible (non-greedy only): no upstream call at all.
        sampled_req = _request_state([1, 2, 3], prompt_len=1, temperature=1.0)
        assert proposer.propose(_context(decode_reqs=[("s", sampled_req)])) is None
        assert fake.calls == []

    def test_empty_draft_rows_are_dropped(self) -> None:
        fake = _FakeVllmSuffix()
        fake.extra_draft = {0: []}  # the tree has nothing for this suffix
        proposer = _proposer(fake)
        state = _request_state([9, 9, 9], prompt_len=1)

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None


class TestSuffixRealTree:
    def test_repetitive_history_drafts_continuation(self) -> None:
        pytest.importorskip(
            "arctic_inference.suffix_decoding", reason="arctic_inference not installed"
        )
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="suffix",
                num_speculative_tokens=4,
                suffix_decoding_max_tree_depth=24,
                suffix_decoding_max_spec_factor=1.0,
                suffix_decoding_min_token_prob=0.0,
                suffix_decoding_max_cached_requests=10000,
            ),
            model_config=SimpleNamespace(max_model_len=512),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        proposer = SuffixProposer.build(
            vllm_config=vllm_config,
            controller=SpeculativeDecodeController(),
        )

        # The history contains [10, 11, 12] -> [13, 14] (-> 10, the repeat)
        # and the request has just re-emitted [10, 11, 12]: the tree should
        # speculate the earlier continuation, as deep as the matched length
        # allows under max_spec_factor=1.0.
        state = _request_state([1, 2, 10, 11, 12, 13, 14, 10, 11, 12], prompt_len=7)
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert drafts is not None
        assert drafts.draft_token_ids == [[13, 14, 10]]

        # A request whose suffix never appeared drafts nothing.
        fresh = _request_state([31, 32, 33, 34], prompt_len=2)
        assert proposer.propose(_context(decode_reqs=[("r1", fresh)])) is None
