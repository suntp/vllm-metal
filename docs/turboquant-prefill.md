# TurboQuant prefill

TurboQuant normally reads compressed KV directly in the per-token attention
kernel. Substantial prefill chunks can instead gather their referenced cache
pages, dequantize them, and use the existing NAX (M5) or tiled attention kernel.
The same configured key format and value bit width are used for cache writes
and dequantization.

## Routing and memory

A segment qualifies when its number of query tokens is at least
`max(128, head_dim / 2, ceil(256 * num_kv_heads / num_query_heads))`. This is a
conservative crossover rule from the included MHA/GQA/MQA sweep, not an online
calibration or a guarantee on every device. Short suffixes, decode, speculative
verification, FP32, sliding-window layers and image-block attention retain the
compressed path. Attention sinks still raise the existing unsupported-combination
error during prefill as well as decode.

The materialized batch contains only valid kernel pages. Unused table columns,
unused sub-blocks of large scheduler pages, and decode passengers are excluded;
shared physical prefix pages are gathered once. Page/token indexing preserves
the upstream storage strides, including padded pages. The temporary block table
addresses this gathered copy; scheduler-owned block IDs and the cache allocation
remain authoritative.

Per-forward routing/gather indices share the lifetime of the existing kernel
metadata memo. Dequantized K/V are temporary MLX graph inputs, not persistent
per-layer caches. Only one batch is materialized, limited to the largest cached
prefix plus the current prefill query tokens (rounded to kernel blocks). Prefills
that would exceed that bound use the compressed path, along with short requests;
their outputs are restored to the original packed order. Cold prefills can still
batch together, and shared prefixes can fit several continuation requests within
the bound. This avoids multiplying full-context scratch by concurrency without
adding a separate allocator or a host synchronization barrier. A single long
context still requires temporary dequantization memory.

## Validation and reproduction

`tests/attention/test_turboquant_prefill.py` exercises `sdpa_forward` with real
Q/K/V projections, fused cache writes and native attention on upstream-allocated
storage. It covers K/V bit widths, FP16/BF16, NAX/tiled selection, nonidentity
tables, shared prefixes, translated and padded pages, mixed-output ordering,
fallbacks, cache read-only calls and scratch growth. Current-token cache cells
start empty so the first forward also checks lazy writer/reader dependencies.

The benchmark uses the same production path and storage fixture. Its compressed
reference disables only the prefill planner inside the process. Both arms include
projections, cache writes and metadata construction, run in interleaved order
after warmup, and report medians, numerical error and peak additional MLX memory:

```bash
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite geometry --tiled
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite memory
```

For model-level comparison, run each arm in a fresh process with the same model
and arguments:

```bash
PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model --arm tq-reference
PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model --arm tq
```

This tool is an in-process offline benchmark. Its JSON includes actual token
counts, model/quantization choices, dependency versions, generation wall time
and token IDs. TTFT is null when vLLM does not return it. Use a separate serving
benchmark to measure concurrent request throughput.

Dequantization rounds K/V to the query dtype before attention, and NAX/tiled
accumulation differs from the compressed kernel's deferred value transform.
Numerical tolerance is therefore required; greedy token sequences need not be
identical. Kernel parity and a short model smoke do not establish model-quality
equivalence. Future fused dequantization or using uncompressed current-chunk K/V
should be evaluated separately against these numerical and memory contracts.
