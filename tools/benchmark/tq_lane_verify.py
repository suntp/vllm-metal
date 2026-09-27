# SPDX-License-Identifier: Apache-2.0
"""TurboQuant prefill lane: correctness + interleaved A/B (old vs new).

Replicates the sdpa.py lane at the primitive level with production cache
layouts ((blocks, bs, kvh, packed) + per-32-head-dim scale/zero), then:

  correctness — new lane (batch dequant -> unquantized dispatch) vs old
  lane (per-token quantized kernel) and vs an fp32 reference computed on
  the dequantized caches;
  performance — interleaved arms, medians; the new arm's time INCLUDES
  the batch dequant (the honest lane cost).

    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py
"""

import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from vllm_metal.attention.caches.turboquant import (
    turbo_quant_decode,
    turbo_quant_encode,
)
from vllm_metal.metal import get_ops

NQ, NKV, HS, BS = 24, 4, 256, 16
SCALE = HS**-0.5


def build_pool(seq, seed=715):
    """Encode a dense (tokens, kvh, dim) KV into production pool layouts."""
    mx.random.seed(seed)
    blocks = (seq + BS - 1) // BS
    tokens = blocks * BS
    k = mx.random.normal((tokens, NKV, HS)).astype(mx.bfloat16)
    v = mx.random.normal((tokens, NKV, HS)).astype(mx.bfloat16)
    (ki, ks, kz), (vi, vs) = turbo_quant_encode(k, v, "q8_0", 3)

    def pool(x):
        return x.reshape(blocks, BS, NKV, -1)

    return {
        "blocks": blocks,
        "kq": pool(ki),
        "vq": pool(vi),
        "ks": pool(ks),
        "vs": pool(vs),
        "kz": pool(kz),
        "k_ref": k,
        "v_ref": v,
    }


def reference(q, k, v, seq, qlen):
    qf = q.astype(mx.float32).transpose(1, 0, 2).reshape(NKV, -1, qlen, HS)
    kf = k[:seq].astype(mx.float32).transpose(1, 0, 2)
    vf = v[:seq].astype(mx.float32).transpose(1, 0, 2)
    s = qf @ kf[:, None].transpose(0, 1, 3, 2) * SCALE
    qpos = mx.arange(qlen) + (seq - qlen)
    kvpos = mx.arange(seq)
    s = mx.where(kvpos[None, None, None, :] <= qpos[None, None, :, None], s, -1e30)
    p = mx.softmax(s, axis=-1)
    o = (p @ vf[:, None]).reshape(NKV * (NQ // NKV), qlen, HS).transpose(1, 0, 2)
    return o.astype(q.dtype)


def run_old(ops, q, pool, cu, table, lens, seq, out):
    from vllm_metal.attention.caches.turboquant import get_v_centroids

    ops.paged_attention_primitive(
        q,
        pool["kq"],
        pool["vq"],
        NKV,
        SCALE,
        0.0,
        table,
        lens,
        cu,
        BS,
        seq,
        -1,
        out,
        window_seqlen_q=1,
        sinks=None,
        key_scale_cache=pool["ks"],
        value_scale_cache=pool["vs"],
        key_zero_cache=pool["kz"],
        v_centroids=get_v_centroids(3),
        use_turboquant=True,
        quant_type="q8_0",
        v_bits=3,
    )


def run_new(ops, q, pool, cu, table, lens, seq, out, dtype=mx.bfloat16):
    tokens = pool["kq"].shape[0] * pool["kq"].shape[1]
    k16, v16 = turbo_quant_decode(
        (
            pool["kq"].reshape(tokens, NKV, -1),
            pool["ks"].reshape(tokens, NKV, -1),
            pool["kz"].reshape(tokens, NKV, -1),
        ),
        (pool["vq"].reshape(tokens, NKV, -1), pool["vs"].reshape(tokens, NKV, -1)),
        output_dtype=dtype,
    )
    blocks = pool["blocks"]
    ops.paged_attention_primitive(
        q,
        k16.reshape(blocks, BS, NKV, HS),
        v16.reshape(blocks, BS, NKV, HS),
        NKV,
        SCALE,
        0.0,
        table,
        lens,
        cu,
        BS,
        seq,
        -1,
        out,
        window_seqlen_q=1,
        sinks=None,
    )


def correctness(ops, qlen, seq):
    pool = build_pool(seq)
    table = mx.arange(pool["blocks"], dtype=mx.int32)[None]
    lens = mx.array([seq], mx.int32)
    cu = mx.array([0, qlen], mx.int32)
    q = mx.random.normal((qlen, NQ, HS)).astype(mx.bfloat16)
    mx.eval(
        q, table, lens, cu, pool["kq"], pool["vq"], pool["ks"], pool["vs"], pool["kz"]
    )

    out_old = mx.array(0)
    run_old(ops, q, pool, cu, table, lens, seq, out_old)
    mx.eval(out_old)
    out_new = mx.array(0)
    run_new(ops, q, pool, cu, table, lens, seq, out_new)
    mx.eval(out_new)

    ref = reference(q, pool["k_ref"], pool["v_ref"], seq, qlen)
    d_new_ref = float(
        np.abs(
            np.array(out_new.astype(mx.float32)) - np.array(ref.astype(mx.float32))
        ).max()
    )
    d_old_ref = float(
        np.abs(
            np.array(out_old.astype(mx.float32)) - np.array(ref.astype(mx.float32))
        ).max()
    )
    d_new_old = float(
        np.abs(
            np.array(out_new.astype(mx.float32)) - np.array(out_old.astype(mx.float32))
        ).max()
    )
    scale_o = float(np.abs(np.array(ref.astype(mx.float32))).max())
    print(
        f"correctness qlen={qlen} seq={seq}: new-vs-ref {d_new_ref:.2e} "
        f"old-vs-ref {d_old_ref:.2e} new-vs-old {d_new_old:.2e} "
        f"(out scale {scale_o:.2f})",
        flush=True,
    )
    return d_new_old


def bench(ops, qlen, seq, reps=7):
    pool = build_pool(seq)
    table = mx.arange(pool["blocks"], dtype=mx.int32)[None]
    lens = mx.array([seq], mx.int32)
    cu = mx.array([0, qlen], mx.int32)
    q = mx.random.normal((qlen, NQ, HS)).astype(mx.bfloat16)
    mx.eval(
        q, table, lens, cu, pool["kq"], pool["vq"], pool["ks"], pool["vs"], pool["kz"]
    )
    out = mx.array(0)
    samples = {"old": [], "new": []}
    for rep in range(reps):
        for arm in ("old", "new") if rep % 2 == 0 else ("new", "old"):
            t0 = time.perf_counter()
            if arm == "old":
                run_old(ops, q, pool, cu, table, lens, seq, out)
            else:
                run_new(ops, q, pool, cu, table, lens, seq, out)
            mx.eval(out)
            samples[arm].append((time.perf_counter() - t0) * 1e3)
    m_old = statistics.median(samples["old"])
    m_new = statistics.median(samples["new"])
    print(
        f"bench qlen={qlen} seq={seq}: old {m_old:8.1f}ms  "
        f"new {m_new:8.1f}ms  speedup {m_old / m_new:5.2f}x",
        flush=True,
    )
    return m_old / m_new


def main():
    ops = get_ops()
    check = correctness(ops, 130, 640)
    assert check < 5e-2, check
    sp1 = bench(ops, 1568, 31360)
    sp2 = bench(ops, 2048, 100352)
    print(f"SUMMARY speedups: {sp1:.2f}x / {sp2:.2f}x", flush=True)


if __name__ == "__main__":
    sys.exit(main())
