# SPDX-License-Identifier: Apache-2.0
"""Host-side NAX split/unsplit pipeline choice (vllm_metal.attention.nax_split)."""

import mlx.core as mx
import pytest

from vllm_metal.attention.nax_split import nax_split_hint, nax_tile_work


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
    outs = []
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
        outs.append(out)
    mx.eval(outs)
    assert mx.array_equal(outs[0], outs[1]) and mx.array_equal(outs[0], outs[2])
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
