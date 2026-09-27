# SPDX-License-Identifier: Apache-2.0
"""Exercise production routing, fused cache writes and native TQ attention."""

import mlx.core as mx
import numpy as np
import pytest

from tools.benchmark.tq_prefill_case import build_case
from vllm_metal.attention.impls import sdpa
from vllm_metal.metal import get_ops


@pytest.fixture
def recorded_ops(monkeypatch):
    native = get_ops()
    calls = []

    class RecordingOps:
        def __getattr__(self, name):
            return getattr(native, name)

        def paged_attention_primitive(self, *args, **kwargs):
            calls.append((args, kwargs))
            return native.paged_attention_primitive(*args, **kwargs)

    monkeypatch.setattr(sdpa, "get_ops", lambda: RecordingOps())
    return calls


@pytest.fixture(params=[False, True], ids=["native-default", "tiled"])
def prefill_backend(request):
    ops = get_ops()
    ops.set_nax_enabled(not request.param)
    yield
    ops.set_nax_enabled(True)


def assert_parity(case):
    # No eval between sdpa_forward's tq_encode and attention: in particular,
    # current-token cache cells were NOT populated by the fixture.
    output = case.forward()
    mx.eval(output)
    reference = case.reference()
    mx.eval(reference)
    assert mx.all(mx.isfinite(output)).item()
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        atol=0.02,
        rtol=0.03,
    )
    return output, reference


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    ("k_quant", "v_quant"),
    [
        ("q8_0", "q3_0"),
        ("q4_0", "q4_0"),
        ("q5_0", "q5_0"),
        ("int2", "q2_0"),
        ("uint8", "q8_0"),
    ],
)
def test_prefill_quantization_formats(
    recorded_ops, prefill_backend, dtype, k_quant, v_quant
):
    case = build_case(dtype=dtype, k_quant=k_quant, v_quant=v_quant)
    assert_parity(case)
    assert len(recorded_ops) == 1
    args, kwargs = recorded_ops[0]
    assert not kwargs.get("use_turboquant", False)
    assert args[1].dtype == dtype
    assert args[1].shape == (17, 16, 2, 128)


@pytest.mark.parametrize(
    ("head_dim", "block_size"), [(64, 8), (128, 32), (256, 544), (512, 64)]
)
def test_prefill_strided_translated_pages(
    recorded_ops, prefill_backend, head_dim, block_size
):
    case = build_case(
        head_dim=head_dim,
        block_size=block_size,
        page_padding=512,
        qlens=(256,),
        context_lens=(1153,),
        k_quant="q4_0",
        v_quant="q4_0",
        extra_table_pages=3,
    )
    assert_parity(case)
    args, kwargs = recorded_ops[0]
    assert not kwargs.get("use_turboquant", False)
    kb = args[9]
    # Only ceil(valid KV / kernel block) pages, even if the scheduler page
    # contains many unused sub-blocks and the input table has spare capacity.
    assert args[1].shape[0] == (1153 + kb - 1) // kb


def test_mixed_batch_routes_and_restores_rows(recorded_ops, prefill_backend):
    case = build_case(
        qlens=(1, 128, 3, 129),
        context_lens=(4097, 384, 515, 641),
        page_padding=512,
        shared_prefix=True,
        softcap=2.0,
    )
    assert_parity(case)
    assert len(recorded_ops) == 2
    prefill, fallback = recorded_ops
    assert not prefill[1].get("use_turboquant", False)
    assert fallback[1]["use_turboquant"]
    assert prefill[0][0].shape[0] == 257
    assert fallback[0][0].shape[0] == 4
    assert prefill[0][7].tolist() == [384, 641]
    assert fallback[0][7].tolist() == [4097, 515]
    # The two selected requests share 16 physical prefix pages. Neither the
    # long decode's 4097 tokens nor its rectangular padding is materialized.
    assert prefill[0][1].shape[0] == 24 + 41 - 16
    assert prefill[0][6][0, 0].item() == prefill[0][6][1, 0].item()


@pytest.mark.parametrize("qlen", [1, 2, 32, 127, 128])
def test_short_suffix_crossover(recorded_ops, qlen):
    case = build_case(qlens=(qlen,), context_lens=(2049,))
    output, reference = assert_parity(case)
    quantized = recorded_ops[0][1].get("use_turboquant", False)
    assert quantized == (qlen < 128)
    if quantized:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("past", [0, 2048])
def test_cold_prefills_batch_but_unrelated_histories_keep_compressed(
    recorded_ops, prefill_backend, past
):
    case = build_case(
        qlens=(128,) * 4, context_lens=(past + 128,) * 4, page_padding=512
    )
    assert_parity(case)
    assert len(recorded_ops) == (1 if past == 0 else 2)
    assert not recorded_ops[0][1].get("use_turboquant", False)
    assert recorded_ops[0][0][1].shape[0] == (32 if past == 0 else 136)
    if past:
        assert recorded_ops[1][1]["use_turboquant"]
        assert recorded_ops[1][0][0].shape[0] == 384


def test_capacity_fallback_restores_interleaved_rows(recorded_ops):
    case = build_case(
        qlens=(128, 1, 129, 5, 128), context_lens=(2049, 4097, 2063, 515, 2177)
    )
    assert_parity(case)
    assert len(recorded_ops) == 2
    assert not recorded_ops[0][1].get("use_turboquant", False)
    assert recorded_ops[-1][1]["use_turboquant"]
    assert recorded_ops[-1][0][0].shape[0] == 263


def test_prefill_scratch_does_not_scale_with_unrelated_histories():
    import gc

    peaks = []
    for count in [1, 4]:
        case = build_case(
            qlens=(128,) * count,
            context_lens=(8192,) * count,
            head_dim=256,
            n_heads=24,
            n_kv_heads=4,
        )
        mx.eval(case.forward())
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        output = case.forward()
        mx.eval(output)
        mx.synchronize()
        peaks.append(mx.get_peak_memory() - before)
        del case, output
        gc.collect()
        mx.clear_cache()
    # Output/query rows grow, but four independent histories must not retain
    # four full dequantizations. Generous margin avoids allocator-size noise.
    assert peaks[1] < 2 * peaks[0] + 32 * 2**20, peaks


@pytest.mark.parametrize("qlen", [128, 255, 256])
def test_mha_requires_more_rows_to_amortize_dequant(recorded_ops, qlen):
    case = build_case(
        qlens=(qlen,), context_lens=(1025,), n_heads=8, n_kv_heads=8, head_dim=64
    )
    output, reference = assert_parity(case)
    quantized = recorded_ops[0][1].get("use_turboquant", False)
    assert quantized == (qlen < 256)
    if quantized:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("qlen", [128, 255, 256])
def test_wide_heads_keep_short_chunks_compressed(recorded_ops, qlen):
    case = build_case(qlens=(qlen,), context_lens=(1025,), head_dim=512)
    output, reference = assert_parity(case)
    quantized = recorded_ops[0][1].get("use_turboquant", False)
    assert quantized == (qlen < 256)
    if quantized:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("mode", ["verify", "fp32", "sliding"])
def test_preserves_compressed_fallbacks(recorded_ops, mode):
    kwargs = {}
    if mode == "verify":
        kwargs["qlens"] = (5,)
    elif mode == "fp32":
        kwargs["dtype"] = mx.float32
    else:
        kwargs["sliding_window"] = 32
    case = build_case(**kwargs)
    if mode == "verify":
        case.ctx.verify_window_q = 5
    output, reference = assert_parity(case)
    assert recorded_ops[0][1]["use_turboquant"]
    assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("qlen", [1, 128])
def test_rejects_sinks_during_prefill_and_decode(recorded_ops, qlen):
    case = build_case(qlens=(qlen,))
    case.inner.sinks = mx.zeros((case.inner.n_heads,), mx.float32)
    with pytest.raises(ValueError, match="sinks are not supported with TurboQuant"):
        case.forward()
    assert recorded_ops[0][1]["use_turboquant"]


def test_preserves_mm_prefix_rejection(recorded_ops):
    case = build_case()
    case.ctx.segment_bidi_ranges = [[(129, 257)]]
    case.ctx.bidi_layer_kinds = frozenset({"full"})
    with pytest.raises(
        ValueError, match="mm_prefix ranges are not supported with TurboQuant"
    ):
        case.forward()
    assert recorded_ops[0][1]["use_turboquant"]


def test_prefill_metadata_reused_only_within_forward(recorded_ops):
    case = build_case()
    first = case.forward()
    mx.eval(first)
    meta = next(iter(case.ctx.kernel_metadata_cache.values()))
    plan = meta.tq_prefill_plans[(16, 128)]
    second = case.forward()
    mx.eval(second)
    assert next(iter(case.ctx.kernel_metadata_cache.values())) is meta
    assert meta.tq_prefill_plans[(16, 128)] is plan
    assert mx.array_equal(first, second).item()
    case.ctx.kernel_metadata_cache.clear()
    third = case.forward()
    mx.eval(third)
    fresh = next(iter(case.ctx.kernel_metadata_cache.values()))
    assert fresh is not meta
    assert fresh.tq_prefill_plans[(16, 128)] is not plan


def test_read_existing_cache_stays_read_only(recorded_ops):
    case = build_case(k_quant="q4_0", v_quant="q4_0")
    output = case.forward()
    mx.eval(output)
    before = np.array(case.cache._storage.buffers[0])
    read_output, _ = sdpa.sdpa_forward(
        case.inner,
        case.x,
        case.ctx,
        case.cache,
        0,
        read_existing_kv=True,
    )
    mx.eval(read_output)
    assert mx.array_equal(output, read_output).item()
    np.testing.assert_array_equal(before, np.array(case.cache._storage.buffers[0]))


def test_prefill_addressing_preserves_large_physical_page_ids():
    case = build_case()
    # Planner-only test: legal int32 page IDs whose absolute token indices
    # exceed int32, without allocating a multi-terabyte cache for the test.
    case.ctx.block_tables = [[2**27 + i for i in range(17)]]
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    plan = sdpa._turboquant_prefill_plan(
        case.ctx, meta, case.ctx.block_tables, 16, 8, 2, 128
    )
    assert plan is not None
    assert plan.pool_pages[::16].tolist() == case.ctx.block_tables[0]
    assert plan.pool_offsets[:16].tolist() == list(range(16))
