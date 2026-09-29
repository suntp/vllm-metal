# SPDX-License-Identifier: Apache-2.0
"""Ensure crossover measurements compare distinct, valid production paths."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tools.benchmark import tq_lane_verify as bench
from vllm_metal.attention.impls import turboquant_prefill as tq_prefill


@pytest.mark.parametrize("kv_heads", [2, 8])
def test_crossover_materializes_below_policy_threshold(
    monkeypatch, capsys, force_tiled_prefill, kv_heads
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "64")
    threshold = tq_prefill.min_prefill_tokens(8, kv_heads, 128)
    bench.measure(
        "below-policy",
        {"qlens": (32,), "context_lens": (257,), "n_kv_heads": kv_heads},
        reps=1,
        warmup=0,
        force_materialization=True,
    )
    row = json.loads(capsys.readouterr().out.strip())
    assert row["production_lane_selected"] is False
    assert row["lane_selected"] is True
    assert row["forced_materialization"] is True
    assert row["workspace_estimate_bytes"] > 0
    assert all(value > 0 for value in row["median_ms"].values())
    assert tq_prefill.min_prefill_tokens(8, kv_heads, 128) == threshold


def test_crossover_does_not_bypass_workspace_limit(monkeypatch, force_tiled_prefill):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "1")
    with pytest.raises(RuntimeError, match="materialization rejected"):
        bench.measure(
            "over-budget",
            {"qlens": (32,), "context_lens": (8192,)},
            reps=1,
            warmup=0,
            force_materialization=True,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_benchmark_rejects_matching_nonfinite_outputs(monkeypatch, value):
    """Agreement between invalid arms must not produce a timing result."""
    case = SimpleNamespace(
        ctx=SimpleNamespace(kernel_metadata_cache={}),
        forward=lambda: mx.array([value]),
    )
    monkeypatch.setattr(bench, "build_case", lambda **_: case)
    with pytest.raises(RuntimeError, match="nonfinite attention output"):
        bench.measure("nonfinite", {}, reps=1, warmup=0)
