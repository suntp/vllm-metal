# SPDX-License-Identifier: Apache-2.0
"""Op-level A/B benchmark for the paged-attention decode primitive.

Times kernel paths of one source tree in a single process with
interleaved repeats: the GQA paged-decode kernel at a forced partition
size against the production split-KV path (``--baseline gqa-disabled``),
or a single production shape with no candidate arm (``--candidate none``,
the shape-bench mode).  Without ``--kv-len``, every partition is measured
at the auto-route threshold ``P * ceil(simd_groups * cores / q_heads)``
-- the KV length at which the router would first select that tier -- so
each tier is sampled exactly where it matters.

Runs against a source checkout whose native extension is already built::

    python -m tools.benchmark.macos.primitive_ab \
        --source /path/to/checkout --label run1 --out runs/run1.json

Numerics are checked against the reference implementation shipped in the
tree's test suite, and the dispatch family actually taken is read back
after every measured block, so an arm that silently routed elsewhere
fails loudly instead of producing a plausible number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# simd groups per GPU core the partition router packs against; mirrors
# kSimdGroupsPerCore in the paged-attention dispatch.
SIMD_GROUPS_PER_CORE = 33

# Attention geometries of small single-GPU checkpoints, for matrix runs.
# Any other shape can be given explicitly with --heads/--kv-heads/...
GEOMETRY_PRESETS = {
    "minicpm5-2b": {"q": 16, "kv": 2, "head": 128, "block": 16},
    "qwen3-4b": {"q": 32, "kv": 8, "head": 128, "block": 16},
    "hs256-24-4": {"q": 24, "kv": 4, "head": 256, "block": 16},
    "hs256-16-2": {"q": 16, "kv": 2, "head": 256, "block": 16},
}

BASELINE_FAMILIES = {
    "gqa-disabled": {"per_token_ps0", "per_token_ps512"},
    "prod": None,  # routes wherever the tree routes; recorded, not asserted
}


def threshold(partition: int, query_heads: int, cores: int) -> int:
    """KV length where the router first selects *partition*."""
    groups = (SIMD_GROUPS_PER_CORE * cores + query_heads - 1) // query_heads
    return partition * groups


def source_git(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_geometries(args: argparse.Namespace) -> list[dict]:
    if args.heads is not None:
        if args.geometry:
            raise SystemExit(
                "pass either --geometry presets or explicit "
                "--heads/--kv-heads/..., not both"
            )
        return [
            {
                "id": f"custom-{args.heads}-{args.kv_heads}-{args.head_size}",
                "q": args.heads,
                "kv": args.kv_heads,
                "head": args.head_size,
                "block": args.block_size,
            }
        ]
    ids = args.geometry or list(GEOMETRY_PRESETS)
    unknown = [i for i in ids if i not in GEOMETRY_PRESETS]
    if unknown:
        raise SystemExit(
            f"unknown geometry {unknown}; choose from {list(GEOMETRY_PRESETS)} "
            "or pass --heads/--kv-heads/--head-size/--block-size"
        )
    return [{"id": i, **GEOMETRY_PRESETS[i]} for i in ids]


def make_problem(ops, geo: dict, length: int):
    """Allocate the KV cache/query/block-table problem for one shape."""
    import mlx.core as mx
    import numpy as np

    nq, nkv, hs, block = geo["q"], geo["kv"], geo["head"], geo["block"]
    mx.random.seed(715)
    n_blocks = (length + block - 1) // block
    raw = mx.random.normal((n_blocks, block, nkv, 2 * hs)).astype(mx.bfloat16)
    strides = (block * nkv * 2 * hs, nkv * 2 * hs, 2 * hs, 1)
    key = ops.as_strided(raw, (n_blocks, block, nkv, hs), strides, 0)
    value = ops.as_strided(raw, (n_blocks, block, nkv, hs), strides, hs)
    queries = [mx.random.normal((1, nq, hs)).astype(mx.bfloat16) for _ in range(16)]
    table = mx.array(np.random.default_rng(715).permutation(n_blocks).astype(np.int32))[
        None
    ]
    lens = mx.array([length], mx.int32)
    cu = mx.array([0, 1], mx.int32)
    mx.eval(key, value, *queries, table, lens, cu)
    return key, value, queries, table, lens, cu


def bench_one(ops, reference, geo, partition, length, args, baseline_mode):
    """One (geometry, partition, length) cell; returns the result row."""
    import mlx.core as mx
    import numpy as np

    nq, nkv, hs = geo["q"], geo["kv"], geo["head"]
    key, value, queries, table, lens, cu = make_problem(ops, geo, length)
    ref = reference(
        query=queries[0],
        key_cache=key,
        value_cache=value,
        query_lens=[1],
        kv_lens=[length],
        block_tables=np.array(table),
        scale=hs**-0.5,
    )
    mx.eval(ref)

    use_candidate = args.candidate == "gqa"
    arms = ["candidate", "baseline"] if use_candidate else ["baseline"]

    def evaluate(arm: str, query_index: int = 0):
        out = mx.array(0)
        query = queries[query_index % len(queries)]
        if arm == "candidate":
            ops._gqa_paged_attention_for_test(
                query,
                key,
                value,
                hs**-0.5,
                table,
                lens,
                geo["block"],
                length,
                partition,
                out,
            )
        elif baseline_mode == "gqa-disabled":
            ops.paged_attention_primitive(
                query,
                key,
                value,
                nkv,
                hs**-0.5,
                0.0,
                table,
                lens,
                cu,
                geo["block"],
                length,
                -1,
                out,
                window_seqlen_q=1,
                num_decode_requests=1,
                gqa_disabled=True,
            )
        else:
            ops.paged_attention_primitive(
                query,
                key,
                value,
                nkv,
                hs**-0.5,
                0.0,
                table,
                lens,
                cu,
                geo["block"],
                length,
                -1,
                out,
            )
        mx.eval(out)
        return out

    # Numerics gate on the candidate arm: a fast wrong kernel is worthless.
    if use_candidate:
        out = evaluate("candidate")
        family = ops.last_paged_dispatch()
        if family != "gqa_decode":
            raise SystemExit(f"candidate did not route to gqa_decode: {family}")
        if ops.last_gqa_partition_size() != partition:
            raise SystemExit(
                f"candidate partition {ops.last_gqa_partition_size()} != {partition}"
            )
        err = float(
            np.abs(
                np.array(out.astype(mx.float32)) - np.array(ref.astype(mx.float32))
            ).max()
        )
        if err >= 3e-2:
            raise SystemExit(f"numerics regression: max abs err {err}")
        rel_l2 = float(
            np.linalg.norm(
                np.array(out.astype(mx.float32)) - np.array(ref.astype(mx.float32))
            )
            / max(np.linalg.norm(np.array(ref.astype(mx.float32))), 1e-10)
        )
    else:
        err, rel_l2 = None, None

    deadline = time.monotonic() + args.warm_seconds
    warm_i = 0
    while time.monotonic() < deadline:
        evaluate(arms[warm_i % len(arms)], warm_i)
        warm_i += 1

    samples = {arm: [] for arm in arms}
    families = {}
    for rep in range(args.reps):
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for arm in order:
            for i in range(8):
                evaluate(arm, i)
            start = time.perf_counter()
            for i in range(args.inner):
                evaluate(arm, i)
            samples[arm].append((time.perf_counter() - start) / args.inner)
            evaluate(arm)
            families[arm] = {
                "family": ops.last_paged_dispatch(),
                "partition": ops.last_gqa_partition_size(),
            }

    if use_candidate:
        if families["candidate"]["family"] != "gqa_decode":
            raise SystemExit(f"candidate routed to {families['candidate']['family']}")
        if families["candidate"]["partition"] != partition:
            raise SystemExit("candidate partition drifted mid-run")
    expected = BASELINE_FAMILIES[baseline_mode]
    if expected and families["baseline"]["family"] not in expected:
        raise SystemExit(
            f"baseline routed to {families['baseline']['family']}, "
            f"expected one of {sorted(expected)}"
        )

    def stats(times):
        med = statistics.median(times)
        return {
            "us": med * 1e6,
            "logical_kv_gb_s": length * nkv * hs * 2 * 2 / med / 1e9,
            "cv": statistics.pstdev(times) / statistics.mean(times),
            "samples_us": [t * 1e6 for t in times],
        }

    row = {
        "model": geo["id"],
        "q": nq,
        "kv": nkv,
        "head": hs,
        "block": geo["block"],
        "partition": partition if use_candidate else None,
        "baseline_mode": baseline_mode,
        "kv_tokens": length,
        "max_abs_err": err,
        "relative_l2": rel_l2,
        "families": families,
    }
    if use_candidate:
        row["gqa"] = stats(samples["candidate"])
        row["disabled"] = stats(samples["baseline"])
        row["speedup_vs_disabled"] = row["disabled"]["us"] / row["gqa"]["us"]
        row["gain_vs_disabled"] = row["speedup_vs_disabled"] - 1
    else:
        row["prod"] = stats(samples["baseline"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="source checkout with a built native extension",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True, help="result JSON path")
    parser.add_argument(
        "--geometry",
        nargs="+",
        choices=sorted(GEOMETRY_PRESETS),
        help="geometry preset ids (default: all)",
    )
    parser.add_argument("--heads", type=int, help="explicit query heads")
    parser.add_argument("--kv-heads", type=int, help="explicit KV heads")
    parser.add_argument("--head-size", type=int, help="explicit head size")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--partitions",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="candidate partition sizes to force",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        nargs="+",
        help="explicit KV lengths; default is the per-partition auto-route threshold",
    )
    parser.add_argument(
        "--candidate",
        choices=["gqa", "none"],
        default="gqa",
        help="'none' benches only the production path (shape mode)",
    )
    parser.add_argument(
        "--baseline",
        choices=["gqa-disabled", "prod"],
        default="gqa-disabled",
        help="'prod' runs the production path as-is (no forced routing)",
    )
    parser.add_argument("--warm-seconds", type=float, default=8)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--inner", type=int, default=80)
    parser.add_argument(
        "--previous",
        type=Path,
        help="prior result JSON; records arm-vs-same-arm gain across runs",
    )
    args = parser.parse_args()
    if args.candidate == "none" and not args.kv_len:
        raise SystemExit("shape mode (--candidate none) requires --kv-len")

    source = args.source.resolve()
    if not (source / "vllm_metal").is_dir():
        raise SystemExit(f"{source} does not look like a vllm-metal checkout")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "tests"))

    try:
        from test_gqa_paged_decode import _grouped_paged_reference
    except ImportError as exc:
        raise SystemExit(
            f"reference helper not importable from {source / 'tests'} "
            f"({exc}); the tree must ship its paged-attention tests"
        ) from None
    from tools.benchmark.macos import evidence
    from vllm_metal.metal import get_ops

    ops = get_ops()
    cores = ops.detected_gpu_core_count()
    geometries = build_geometries(args)
    baseline_mode = args.baseline
    if args.candidate == "gqa" and not hasattr(ops, "_gqa_paged_attention_for_test"):
        raise SystemExit(
            "--candidate gqa requires a tree that ships the GQA test entry "
            "(ops._gqa_paged_attention_for_test)"
        )

    result = {
        "schema": "vllm-metal-macos-primitive-ab/1",
        "label": args.label,
        "head": source_git(source),
        "cores": cores,
        "platform": evidence.platform_record(),
        "argv": sys.argv[1:],
        "started_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "warm_seconds": args.warm_seconds,
            "reps": args.reps,
            "inner": args.inner,
            "dtype": "bfloat16",
            "pages": "shuffled",
            "kv_layout": "strided_packed",
            "candidate": args.candidate,
            "baseline": baseline_mode,
        },
        "rows": [],
    }
    print(
        json.dumps(
            {
                "label": args.label,
                "head": result["head"],
                "cores": cores,
            }
        ),
        flush=True,
    )

    previous = None
    if args.previous is not None:
        previous_doc = json.loads(args.previous.read_text())
        previous = {(r["model"], r["partition"]): r for r in previous_doc["rows"]}

    for geo in geometries:
        if args.candidate == "gqa":
            cells = [
                (part, args.kv_len or [threshold(part, geo["q"], cores)])
                for part in args.partitions
            ]
        else:
            cells = [(None, args.kv_len)]
        for part, lengths in cells:
            for length in lengths:
                row = bench_one(
                    ops,
                    _grouped_paged_reference,
                    geo,
                    part,
                    length,
                    args,
                    baseline_mode,
                )
                if previous is not None:
                    prior = previous.get((row["model"], row["partition"]))
                    if prior is not None:
                        row["previous_label"] = previous_doc["label"]
                        for arm in ("gqa", "prod"):
                            if arm in prior and arm in row:
                                row[f"gain_vs_previous_{arm}"] = (
                                    prior[arm]["us"] / row[arm]["us"] - 1
                                )
                result["rows"].append(row)
                print(json.dumps(row), flush=True)

    result["finished_utc"] = datetime.now(UTC).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print("SUMMARY", flush=True)
    for row in result["rows"]:
        gain = row.get("gain_vs_disabled")
        if gain is not None:
            print(
                f"{row['model']:12} P{row['partition']:<3} L={row['kv_tokens']:<6} "
                f"disabled {row['disabled']['us']:8.1f}us  "
                f"gqa {row['gqa']['us']:8.1f}us  "
                f"vs split-KV {gain * 100:+6.2f}%",
                flush=True,
            )
        else:
            print(
                f"{row['model']:12} {'--':<4} L={row['kv_tokens']:<6} "
                f"prod {row['prod']['us']:8.1f}us",
                flush=True,
            )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
