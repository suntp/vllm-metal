# SPDX-License-Identifier: Apache-2.0
"""Production TurboQuant prefill correctness, interleaved timing and memory.

Both arms execute sdpa_forward, including projections, fused cache writes,
block translation and dispatch. The reference disables only the new planner.
The fixture binds upstream storage and uses nonidentity tables, optional page
padding, shared prefixes and extra unused pool capacity. Reported error is
against the compressed native attention path on the same quantized cache.

    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite crossover
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite memory
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite geometry --tiled
"""

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import time
from contextlib import nullcontext
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from tools.benchmark.tq_prefill_case import build_case
from vllm_metal.attention.impls import sdpa
from vllm_metal.metal import get_ops


def run(case, reference):
    # A fresh forward's metadata; normal multi-layer serving reuses it after
    # the first layer. Both arms pay the same projection and cache-write work.
    case.ctx.kernel_metadata_cache.clear()
    context = (
        patch.object(sdpa, "_turboquant_prefill_plan", return_value=None)
        if reference
        else nullcontext()
    )
    with context:
        output = case.forward()
        mx.eval(output)
    return output


def measure(label, kwargs, reps, warmup):
    case = build_case(**kwargs)
    reference = run(case, True)
    lane = run(case, False)
    delta = mx.abs(lane.astype(mx.float32) - reference.astype(mx.float32)).max().item()
    np.testing.assert_allclose(
        np.array(lane.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        atol=0.02,
        rtol=0.03,
    )
    del lane, reference
    samples = {"compressed": [], "prefill": []}
    for rep in range(warmup + reps):
        arms = [True, False] if rep % 2 == 0 else [False, True]
        for reference in arms:
            start = time.perf_counter()
            output = run(case, reference)
            elapsed = (time.perf_counter() - start) * 1000
            if rep >= warmup:
                samples["compressed" if reference else "prefill"].append(elapsed)
            del output
    memory = {}
    for reference in (True, False):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        output = run(case, reference)
        mx.synchronize()
        memory["compressed" if reference else "prefill"] = (
            mx.get_peak_memory() - before
        ) / 2**20
        del output
    median = {k: statistics.median(v) for k, v in samples.items()}
    plans = [
        plan
        for meta in case.ctx.kernel_metadata_cache.values()
        for plan in meta.tq_prefill_plans.values()
        if plan is not None
    ]
    print(
        json.dumps(
            {
                "case": label,
                "config": kwargs,
                "max_abs_error": delta,
                "median_ms": median,
                "speedup": median["compressed"] / median["prefill"],
                "peak_extra_mib": memory,
                "samples_ms": samples,
                "lane_selected": bool(plans),
                "lane_requests": sum(plan.prefill.seq_lens.shape[0] for plan in plans),
                "workspace_estimate_bytes": sum(plan.workspace_bytes for plan in plans),
            },
            default=str,
        ),
        flush=True,
    )
    del case
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def cases(suite):
    common = {"head_dim": 256, "n_heads": 24, "n_kv_heads": 4}
    if suite == "crossover":
        for qlen, context in [
            (2, 4096),
            (2, 32768),
            (8, 32768),
            (32, 32768),
            (128, 8192),
            (512, 8192),
            (128, 32768),
            (512, 32768),
            (1153, 1153),
        ]:
            yield (
                f"q{qlen}-kv{context}",
                dict(common, qlens=(qlen,), context_lens=(context,)),
            )
    elif suite == "geometry":
        for hd, nq, nkv in [(64, 8, 8), (128, 8, 2), (256, 24, 4), (512, 8, 1)]:
            for dtype in [mx.float16, mx.bfloat16]:
                for qlen in [128, 256, 512]:
                    yield (
                        f"d{hd}-q{nq}-kv{nkv}-{dtype}-{qlen}",
                        {
                            "head_dim": hd,
                            "n_heads": nq,
                            "n_kv_heads": nkv,
                            "dtype": dtype,
                            "qlens": (qlen,),
                            "context_lens": (8192,),
                        },
                    )
    elif suite == "limits":
        for context in [8192, 10880, 10912, 10928, 16384, 131072]:
            yield (
                f"cap-kv{context}",
                dict(common, qlens=(128,), context_lens=(context,)),
            )
        for count in [2, 4]:
            yield (
                f"independent4k-{count}",
                dict(common, qlens=(128,) * count, context_lens=(4096,) * count),
            )
        for kq, vq in [
            ("q4_0", "q4_0"),
            ("q5_0", "q5_0"),
            ("int2", "q2_0"),
            ("uint8", "q8_0"),
        ]:
            yield (
                f"cap-formats-{kq}-{vq}",
                dict(
                    common, qlens=(128,), context_lens=(10880,), k_quant=kq, v_quant=vq
                ),
            )
    else:
        for padding in [0, 512]:
            for pool in [1024, 16384]:
                yield (
                    f"padding{padding}-pool{pool}",
                    dict(
                        common,
                        page_padding=padding,
                        pool_blocks=pool,
                        qlens=(128,),
                        context_lens=(513,),
                    ),
                )
        for batch in [1, 4, 8]:
            yield (
                f"mixed{batch}",
                dict(
                    common,
                    qlens=(128,) + (1,) * (batch - 1),
                    context_lens=(8192,) + (16,) * (batch - 1),
                ),
            )
        yield (
            "shared-prefix",
            dict(
                common,
                qlens=(128,) * 8,
                context_lens=(8192,) * 8,
                shared_prefix=True,
            ),
        )

        yield (
            "independent-histories",
            dict(common, qlens=(128,) * 8, context_lens=(8192,) * 8),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        choices=["crossover", "geometry", "memory", "limits"],
        default="crossover",
    )
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tiled", action="store_true")
    args = ap.parse_args()
    if args.reps < 1 or args.warmup < 0:
        ap.error("reps must be positive and warmup must be nonnegative")
    ops = get_ops()
    if args.tiled:
        os.environ["VLLM_METAL_TQ_PREFILL"] = "1"
    ops.set_nax_enabled(not args.tiled)
    print(
        json.dumps(
            {
                "device": mx.device_info()["device_name"],
                "nax_ready": ops.nax_ready(),
                "workspace_limit_bytes": sdpa.prefill_workspace_bytes(),
                "versions": {
                    n: importlib.metadata.version(n) for n in ["vllm", "mlx", "mlx-lm"]
                },
            }
        ),
        flush=True,
    )
    try:
        for label, kwargs in cases(args.suite):
            measure(label, kwargs, args.reps, args.warmup)
    finally:
        ops.set_nax_enabled(True)


if __name__ == "__main__":
    main()
