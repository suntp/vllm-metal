# SPDX-License-Identifier: Apache-2.0
"""Host-side NAX split/unsplit pipeline choice (vllm_metal.attention.nax_split)."""

import mlx.core as mx
import pytest

from tests.gqa_test_utils import _primitive
from tests.test_gqa_decode_routing import _mixed_call, _mixed_inputs
from vllm_metal.attention.nax_split import (
    nax_split_hint,
    nax_split_kwargs,
    nax_tile_work,
)


def _work(seqs):
    """seqs: (query tokens, cached context tokens) per sequence."""
    cu, kv = [0], []
    for q, context in seqs:
        cu.append(cu[-1] + q)
        kv.append(q + context)
    return nax_tile_work(cu, kv)


# (batch, expected hint for head_size 128, 256), from M5 Max measurements.
_CASES = [
    ([(63, 8129)], 0, 0),  # short prefill
    ([(64, 8128)], 0, 0),  # one full block
    ([(65, 8127)], 0, 0),
    ([(193, 7999)], 0, 0),  # three full blocks + residual
    ([(257, 7935)], 0, 1),
    ([(513, 7679)], 0, 1),
    ([(1025, 7167)], 1, 1),
    ([(2048, 6144)], 1, 1),
    ([(63, 8129), (1, 8191), (1, 8191), (1, 8191)], 0, 0),  # 63 + decode
    ([(2048, 6144), (1, 8191), (1, 8191), (1, 8191)], 1, 1),  # 2048 + decode
    ([(193, 1855), (63, 8129)], 0, 0),  # long + short on cached prefix
    ([(193, 1855), (63, 32705)], 0, 0),
    ([(1024, 7168), (63, 32705)], 0, 1),
    ([(63, 32705)], 0, 0),  # cached 32K + 63
    ([(2047, 129024)], 1, 1),  # last chunk of a 128K prefill
]


@pytest.mark.parametrize(("seqs", "d128", "d256"), _CASES)
def test_routing_table(seqs, d128, d256):
    work = _work(seqs)
    assert nax_split_hint(work, 128) == d128
    assert nax_split_hint(work, 64) == d128
    assert nax_split_hint(work, 256) == d256
    assert nax_split_hint(work, 512) == 0


def test_tile_counts_follow_the_causal_mask():
    work = _work([(129, 64)])
    # Two full blocks end at KV 128 and 192 (4 and 6 tiles); the residual
    # row sees all 193 tokens (7 tiles).
    assert (work.full_blocks, work.full_tiles, work.partial_tiles) == (2, 10, 7)


def test_split_kwargs_defaults_to_the_original_loop():
    caps = {"nax_split_hint": True}
    work = _work([(2048, 6144)])
    assert nax_split_kwargs({}, work, 128) == {}
    assert nax_split_kwargs({"nax_split_hint": False}, work, 128) == {}
    assert nax_split_kwargs(caps, work, 128) == {"nax_split_hint": 1}
    assert nax_split_kwargs(caps, None, 128) == {"nax_split_hint": 0}


def test_separated_decode_work_changes_the_prefill_hint():
    cu, kv = [0, 1, 257], [131072, 8192]
    whole = nax_tile_work(cu, kv)
    prefill = nax_tile_work(cu[1:], kv[1:])
    assert (whole.full_blocks, whole.full_tiles, whole.partial_tiles) == (4, 1012, 4096)
    assert (prefill.full_blocks, prefill.full_tiles, prefill.partial_tiles) == (
        4,
        1012,
        0,
    )
    assert nax_split_hint(whole, 256) == 0
    assert nax_split_hint(prefill, 256) == 1


@pytest.mark.parametrize(
    "caps,expected",
    [
        ({}, {}),
        ({"nax_prefill_split_hint": True}, {}),
        ({"nax_split_hint": True}, {"nax_split_hint": 0}),
        (
            {"nax_split_hint": True, "nax_prefill_split_hint": False},
            {"nax_split_hint": 0},
        ),
        (
            {"nax_split_hint": True, "nax_prefill_split_hint": True},
            {"nax_split_hint": 0, "nax_prefill_split_hint": 1},
        ),
    ],
)
def test_prefill_hint_keeps_whole_batch_and_old_build_compatibility(caps, expected):
    whole = nax_tile_work([0, 1, 257], [131072, 8192])
    prefill = nax_tile_work([1, 257], [8192])
    assert nax_split_kwargs(caps, whole, 256, prefill_work=prefill) == expected
    if caps.get("nax_split_hint"):
        assert nax_split_kwargs(caps, whole, 256) == {"nax_split_hint": 0}


def test_sliding_window_counts_only_tiles_inside_the_window():
    # The kernel skips KV tiles left of each block's window, so a 128-token
    # window leaves the 256-row prefill only ~6 tiles per full block: the
    # causal-prefix count would split, the windowed count must not.
    seqs = [(256, 131072)] + [(63, 0)] * 9
    cu, kv = [0], []
    for q, context in seqs:
        cu.append(cu[-1] + q)
        kv.append(q + context)
    causal = nax_tile_work(cu, kv)
    windowed = nax_tile_work(cu, kv, sliding_window=128)
    assert causal.full_blocks == windowed.full_blocks == 4
    assert windowed.full_tiles == 4 * 6 and windowed.partial_tiles == 9 * 2
    assert nax_split_hint(causal, 128) == 1
    assert nax_split_hint(windowed, 128) == 0
    # A window wider than the context changes nothing.
    assert nax_tile_work(cu, kv, sliding_window=1 << 20) == causal


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        nax_tile_work([0, 4], [4, 5])


def test_native_hint_does_not_change_outputs():
    from vllm_metal.metal import get_ops

    ops = get_ops()
    mx.random.seed(0)
    block_size, heads, kv_heads, dim = 16, 8, 2, 128
    seqs = [(300, 40), (70, 0)]
    blocks = [(q + c + block_size - 1) // block_size for q, c in seqs]
    k = mx.random.normal((sum(blocks), block_size, kv_heads, dim)).astype(mx.float16)
    v = mx.random.normal(k.shape).astype(mx.float16)
    q = mx.random.normal((sum(s[0] for s in seqs), heads, dim)).astype(mx.float16)
    starts = [0, blocks[0]]
    width = max(blocks)
    tables = mx.array(
        [
            [start + i if i < n else 0 for i in range(width)]
            for start, n in zip(starts, blocks, strict=True)
        ],
        mx.int32,
    )
    seq_lens = mx.array([q_ + c for q_, c in seqs], mx.int32)
    cu = mx.array([0, 300, 370], mx.int32)
    nax = ops.nax_supported() and ops.nax_ready()
    previous = ops._set_paged_dispatch_diagnostics(True)
    outs, pipelines = [], []
    for hint in (-1, 0, 1):
        out = mx.array(0)
        ops.paged_attention_primitive(
            q,
            k,
            v,
            kv_heads,
            dim**-0.5,
            0.0,
            tables,
            seq_lens,
            cu,
            block_size,
            340,
            -1,
            out,
            nax_split_hint=hint,
        )
        mx.eval(out)
        outs.append(out)
        pipelines.append(ops.last_nax_split_pipeline())
    ops._set_paged_dispatch_diagnostics(previous)
    assert mx.array_equal(outs[0], outs[1]) and mx.array_equal(outs[0], outs[2])
    # -1 falls back to the batch average (185 query tokens per sequence).
    assert pipelines == ([1, 0, 1] if nax else [-1, -1, -1])
    with pytest.raises(ValueError):
        ops.paged_attention_primitive(
            q,
            k,
            v,
            kv_heads,
            dim**-0.5,
            0.0,
            tables,
            seq_lens,
            cu,
            block_size,
            340,
            -1,
            mx.array(0),
            nax_split_hint=2,
        )


@pytest.mark.parametrize(
    "decode_kv,decode_rows,metadata,separated",
    [
        (131072, 1, True, True),
        (128, 1, True, False),
        (8192, 1, False, False),
        (8192, 3, True, False),
    ],
    ids=["ordinary-split", "short-decode", "missing-metadata", "expanded-verify"],
)
def test_native_prefill_hint_only_applies_after_decode_separation(
    nax_prefill, decode_kv, decode_rows, metadata, separated
):
    """Conflicting hints expose the native subset choice and preserve row ownership."""
    ops = nax_prefill
    inputs = _mixed_inputs(
        [decode_kv], prefill_kv=8192, geometry=(8, 2, 256, 16), dtype=mx.float16
    )
    inputs.update(
        query=mx.random.normal((decode_rows + 256, 8, 256)).astype(mx.float16),
        cu_seqlens_q=mx.array([0, decode_rows, decode_rows + 256], mx.int32),
        num_decode_tokens=decode_rows,
        nax_split_hint=0,
    )
    if decode_rows > 1:
        inputs.update(
            cu_seqlens_q=mx.array([0, 1, 2, 3, 259], mx.int32),
            block_tables=inputs["block_tables"][mx.array([0, 0, 0, 1])],
            seq_lens=mx.array([8190, 8191, 8192, 8192], mx.int32),
        )
    if not metadata:
        for name in (
            "num_decode_requests",
            "num_decode_tokens",
            "max_decode_context_len",
        ):
            inputs.pop(name)
    whole = _mixed_call(
        inputs, num_decode_requests=0, num_decode_tokens=0, max_decode_context_len=0
    )
    assert ops.last_paged_dispatch() == "nax_prefill"
    assert ops.last_nax_split_pipeline() == 0
    decode = None
    if separated:
        decode = _primitive(
            inputs["query"][:1],
            inputs["key_cache"],
            inputs["value_cache"],
            inputs["block_tables"][:1],
            inputs["seq_lens"][:1],
            inputs["block_size"],
            decode_kv,
            [decode_kv],
            preplanned=False,
            gqa_disabled=True,
        )
        mx.eval(decode)
    # Evaluate each node before observing diagnostics; hints must not share a
    # lazy result. Missing/-1 subset hints retain the old whole-batch choice.
    for subset in ({"nax_prefill_split_hint": 1}, {"nax_prefill_split_hint": -1}, {}):
        actual = _mixed_call(inputs, **subset)
        assert ops.last_paged_dispatch() == (
            "mixed_nax_prefill_decode" if separated else "nax_prefill"
        )
        assert ops.last_nax_split_pipeline() == int(
            separated and subset.get("nax_prefill_split_hint") == 1
        )
        if separated:
            assert mx.array_equal(actual[:1], decode).item()
            assert mx.array_equal(actual[1:], whole[1:]).item()
        else:
            assert mx.array_equal(actual, whole).item()
    for invalid in (-2, 2):
        with pytest.raises(ValueError, match="nax_prefill_split_hint"):
            _mixed_call(inputs, nax_prefill_split_hint=invalid)
