# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5 MTP proposer's request-local drafter cache."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
    Qwen3_5MTPDraftModel,
)
from vllm import SamplingParams

from vllm_metal.config import QWEN_MTP_DRAFTER_KEY
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.qwen_mtp_proposer import (
    QwenMTPProposer,
    drafter_source,
    is_qwen_mtp,
)
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

HIDDEN = 32
VOCAB = 64
K = 3


def _drafter() -> Qwen3_5MTPDraftModel:
    mx.random.seed(0)
    config = Qwen3_5MTPConfig.from_dict(
        {
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": HIDDEN,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "vocab_size": VOCAB,
                "rms_norm_eps": 1e-6,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 512,
                "tie_word_embeddings": False,
                "mtp_num_hidden_layers": 1,
            }
        }
    )
    drafter = Qwen3_5MTPDraftModel(config)
    target = SimpleNamespace(
        embed_tokens=nn.Embedding(VOCAB, HIDDEN),
        lm_head=nn.Linear(HIDDEN, VOCAB, bias=False),
    )
    drafter.bind(target)
    mx.eval(drafter.parameters(), target.embed_tokens.parameters())
    return drafter


def _proposer(drafter):
    return QwenMTPProposer(drafter=drafter, controller=SpeculativeDecodeController())


def _request_state(max_tokens=None, generated_tokens=0, *, min_tokens=0, temperature=0):
    return SimpleNamespace(
        sampling_params=SamplingParams(
            temperature=temperature, min_tokens=min_tokens, max_tokens=max_tokens
        ),
        generated_tokens=generated_tokens,
    )


def _prefill_ctx(
    req_id,
    chunk,
    start_pos,
    rows,
    sampled_token,
    *,
    final=True,
    request_state=None,
    draft_caps=None,
    finished=frozenset(),
):
    request_state = request_state or _request_state()
    return ProposeContext(
        target_hidden_states=rows,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[
            SimpleNamespace(req_id=req_id, token_ids=list(chunk), start_pos=start_pos)
        ],
        prefill_token_ids=[sampled_token],
        prefill_result_modes=["final" if final else "intermediate"],
        request_states={req_id: request_state},
        cu_seqlens=[0, len(chunk)],
        num_decode_segments=0,
        num_speculative_tokens=K,
        finished_req_ids=set(finished),
        draft_caps=draft_caps,
    )


def _verify_ctx(
    req_id, last_token, drafts, sampled, start_pos, rows, *, request_state=None
):
    segment = PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=(last_token, *drafts),
        start_row=0,
        num_query_tokens=len(drafts) + 1,
        draft_token_ids=tuple(drafts),
        cache_start_pos=start_pos,
        block_ids=((0,),),
    )
    request_state = request_state or _request_state()
    return ProposeContext(
        target_hidden_states=rows,
        decode_reqs=[(req_id, request_state)],
        decode_segments=[segment],
        decode_token_ids=[list(sampled)],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        request_states={req_id: request_state},
        cu_seqlens=[0, len(drafts) + 1],
        num_decode_segments=1,
        num_speculative_tokens=K,
        finished_req_ids=set(),
    )


def _drafts(out) -> dict[str, list[int]]:
    return dict(zip(out.req_ids, out.draft_token_ids, strict=True)) if out else {}


@pytest.fixture(scope="module")
def drafter():
    return _drafter()


@pytest.mark.parametrize("accepted", [0, 1, K])
def test_incremental_cache_matches_single_pass(drafter, accepted) -> None:
    """Chunked prefill + a verify step leave the same cache and drafts as one
    prefill over the committed tokens, whatever the target accepted."""
    mx.random.seed(1)
    prompt = [5, 9, 12, 7, 3, 30, 41, 2]
    hidden = mx.random.normal((len(prompt) + K + 1, HIDDEN))
    first = 17

    incremental = _proposer(drafter)
    split = 5
    assert (
        incremental.propose(
            _prefill_ctx("r", prompt[:split], 0, hidden[:split], 0, final=False)
        )
        is None
    )
    out = incremental.propose(
        _prefill_ctx("r", prompt[split:], split, hidden[split : len(prompt)], first)
    )
    drafts = _drafts(out)["r"]
    assert len(drafts) == K

    # The target accepts `accepted` drafts, then emits its own token.
    correction = (drafts[accepted] + 1) % VOCAB if accepted < K else 11
    sampled = [*drafts[:accepted], correction]
    verify_rows = hidden[len(prompt) : len(prompt) + K + 1]
    after_verify = _drafts(
        incremental.propose(
            _verify_ctx("r", first, drafts, sampled, len(prompt), verify_rows)
        )
    )["r"]

    committed = [*prompt, first, *drafts[:accepted]]
    single = _proposer(drafter)
    expected = _drafts(
        single.propose(
            _prefill_ctx("r", committed, 0, hidden[: len(committed)], correction)
        )
    )["r"]

    assert after_verify == expected
    _assert_same_cache(incremental, single)


def _assert_same_cache(actual, expected):
    for got, want in zip(
        actual._states["r"].cache, expected._states["r"].cache, strict=True
    ):
        assert got.offset == want.offset
        for a, b in zip(got.state, want.state, strict=True):
            assert mx.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("miss", ["min_tokens", "missing_hidden"])
def test_decode_admits_from_first_eligible_target_row(drafter, miss):
    hidden = mx.random.normal((4, HIDDEN))
    state = _request_state(
        generated_tokens=1, min_tokens=4 if miss == "min_tokens" else 0
    )
    proposer = _proposer(drafter)
    prompt_rows = None if miss == "missing_hidden" else hidden
    assert (
        proposer.propose(
            _prefill_ctx("r", [1, 2, 3, 4], 0, prompt_rows, 5, request_state=state)
        )
        is None
    )
    position = 4
    while state.generated_tokens < state.sampling_params.min_tokens - 1:
        state.generated_tokens += 1
        assert (
            proposer.propose(
                _verify_ctx("r", 5, [], [5], position, hidden[:1], request_state=state)
            )
            is None
        )
        position += 1

    state.generated_tokens += 1
    actual = proposer.propose(
        _verify_ctx("r", 5, [], [6], position, hidden[:1], request_state=state)
    )
    fresh = _proposer(drafter)
    expected = fresh.propose(_prefill_ctx("r", [5], position, hidden[:1], 6))
    assert len(_drafts(actual)["r"]) == K
    assert _drafts(actual) == _drafts(expected)
    _assert_same_cache(proposer, fresh)


@pytest.mark.parametrize("start_pos", [0, 9])
def test_discontinuous_prefill_restarts_drafter(drafter, start_pos):
    hidden = mx.random.normal((3, HIDDEN))
    proposer = _proposer(drafter)
    proposer.propose(_prefill_ctx("r", [1, 2, 3], 0, hidden, 0, final=False))
    resumed = _prefill_ctx("r", [7, 8], start_pos, hidden[:2], 9)
    fresh = _proposer(drafter)
    assert _drafts(proposer.propose(resumed)) == _drafts(fresh.propose(resumed))
    _assert_same_cache(proposer, fresh)


def test_decode_restarts_after_missing_verify_rows(drafter):
    proposer = _proposer(drafter)
    rows = mx.random.normal((3, HIDDEN))
    drafts = _drafts(proposer.propose(_prefill_ctx("r", [1, 2, 3], 0, rows, 4)))["r"]
    assert (
        proposer.propose(_verify_ctx("r", 4, drafts, [drafts[0], 5], 3, None)) is None
    )

    actual = proposer.propose(_verify_ctx("r", 5, [], [6], 5, rows[:1]))
    fresh = _proposer(drafter)
    assert _drafts(actual) == _drafts(
        fresh.propose(_prefill_ctx("r", [5], 5, rows[:1], 6))
    )
    _assert_same_cache(proposer, fresh)


@pytest.mark.parametrize(
    ("draft_caps", "max_tokens", "generated", "expected"),
    [
        ({"r": 1}, None, 0, 1),
        (None, 10, 8, 1),
        ({"r": 2}, 10, 0, 2),
        ({"r": 0}, None, 0, 0),
    ],
)
def test_draft_length_respects_caps(
    drafter, draft_caps, max_tokens, generated, expected
) -> None:
    proposer = _proposer(drafter)
    out = proposer.propose(
        _prefill_ctx(
            "r",
            [1, 2, 3],
            0,
            mx.zeros((3, HIDDEN)),
            4,
            request_state=_request_state(max_tokens, generated),
            draft_caps=draft_caps,
        )
    )
    assert len(_drafts(out).get("r", [])) == expected


def test_batched_drafts_match_single_request_drafts(drafter) -> None:
    """Several requests drafted together (shared LM head call per step) give
    the same drafts as each drafted alone, whatever their draft caps."""
    mx.random.seed(2)
    prompts = {"a": [3, 8, 1, 22], "b": [40, 5], "c": [9, 9, 14]}
    sampled = {"a": 6, "b": 13, "c": 50}
    caps = {"a": K, "b": 1, "c": 2}
    hidden = {r: mx.random.normal((len(p), HIDDEN)) for r, p in prompts.items()}

    expected = {}
    for req_id, prompt in prompts.items():
        alone = _proposer(drafter)
        expected |= _drafts(
            alone.propose(
                _prefill_ctx(
                    req_id,
                    prompt,
                    0,
                    hidden[req_id],
                    sampled[req_id],
                    draft_caps={req_id: caps[req_id]},
                )
            )
        )

    req_ids = list(prompts)
    cu = [0]
    for req_id in req_ids:
        cu.append(cu[-1] + len(prompts[req_id]))
    together = _proposer(drafter)
    out = together.propose(
        ProposeContext(
            target_hidden_states=mx.concatenate([hidden[r] for r in req_ids]),
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=[
                SimpleNamespace(req_id=r, token_ids=prompts[r], start_pos=0)
                for r in req_ids
            ],
            prefill_token_ids=[sampled[r] for r in req_ids],
            prefill_result_modes=["final"] * len(req_ids),
            request_states={r: _request_state() for r in req_ids},
            cu_seqlens=cu,
            num_decode_segments=0,
            num_speculative_tokens=K,
            finished_req_ids=set(),
            draft_caps=caps,
        )
    )
    assert _drafts(out) == expected
    assert {r: len(d) for r, d in expected.items()} == caps


def test_ineligible_request_keeps_no_drafter_cache(drafter) -> None:
    proposer = _proposer(drafter)
    out = proposer.propose(
        _prefill_ctx(
            "r",
            [1, 2, 3],
            0,
            mx.zeros((3, HIDDEN)),
            4,
            request_state=_request_state(temperature=1),
        )
    )
    assert out is None
    assert "r" not in proposer._states


def test_finished_request_id_starts_fresh(drafter) -> None:
    """vLLM can reuse a finished id in the same step; the new request must
    not inherit the old drafter cache."""
    proposer = _proposer(drafter)
    fresh = _proposer(drafter)
    proposer.propose(_prefill_ctx("r", [1, 2, 3], 0, mx.zeros((3, HIDDEN)), 4))
    reused = proposer.propose(
        _prefill_ctx("r", [7, 8], 0, mx.zeros((2, HIDDEN)), 9, finished={"r"})
    )
    expected = fresh.propose(_prefill_ctx("r", [7, 8], 0, mx.zeros((2, HIDDEN)), 9))
    assert _drafts(reused) == _drafts(expected)
    assert proposer._states["r"].cache[0].offset == fresh._states["r"].cache[0].offset


def test_is_qwen_mtp_matches_draft_model_type() -> None:
    def config(method, model_type):
        return SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type)
            ),
        )

    assert is_qwen_mtp(config("mtp", "qwen3_5_mtp"))
    assert not is_qwen_mtp(config("mtp", "gemma4_assistant"))
    assert not is_qwen_mtp(config("ngram", "qwen3_5_mtp"))
    assert not is_qwen_mtp(None)


def test_drafter_source_prefers_split_drafter() -> None:
    def vllm_config(additional):
        return SimpleNamespace(
            additional_config=additional,
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(model="org/target")
            ),
        )

    assert drafter_source(vllm_config({})) == "org/target"
    assert (
        drafter_source(vllm_config({QWEN_MTP_DRAFTER_KEY: "org/target-MTP-4bit"}))
        == "org/target-MTP-4bit"
    )
