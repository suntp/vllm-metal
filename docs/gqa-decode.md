# GQA decode routing

The default paged attention path selects among 64-, 128-, 256-, and
512-token GQA partitions for the measured single-request geometries below.
Each partition has FP16/BF16 specializations for head dimensions 128/256
with kernel block16, plus head256 with kernel block32. These form 24 active
shader specializations.

## Automatic eligibility and partition selection

The supported `(query heads, KV heads, head dimension)` geometries are
`(32,8,128)`, `(24,4,256)`, `(16,2,128)`, and `(16,2,256)`.
Kernel block16 is supported for each; block32 is additionally supported for
`(16,2,256)`. The existing scope is not inferred from a model-name list.

Let `C` be the detected GPU core count and `Q` the query-head count. All four
partitions use the same empirical budget of 33 SIMD groups per core.
Select the largest P satisfying `floor(KV_length / P) * Q >= 33 * C`;
otherwise use the established path. Equivalently, each partition starts at
`P * ceil(33 * C / Q)`. Admission and promotion share one rule, with no
head-ratio correction or separate admission budget. This is an empirical
policy, not a hardware identity or a prediction of the fastest partition at
every position.

On a 40-core GPU this gives:

| Q/KV/head dimension | Start P64 | Start P128 | Start P256 | Start P512 |
|---|---:|---:|---:|---:|
| 32/8/128 | 2,688 | 5,376 | 10,752 | 21,504 |
| 24/4/256 | 3,520 | 7,040 | 14,080 | 28,160 |
| 16/2/128 or 256 | 5,312 | 10,624 | 21,248 | 42,496 |

These values scale with the detected core count; they are not per-device or
per-model context tables in the implementation. Core-count scaling does not
establish cross-device performance or guarantee a speedup at every boundary.
The budget is shared by all four partitions; it has no per-model exceptions.

Only complete partitions count toward selection. The producer, temporary
buffers and reducer still use `ceil(KV_length / P)` so the final partial
partition is processed. Selection is stateless, uses at most four integer
comparisons, and adds no startup benchmark or per-request performance probe.
`gqa_decode_partition_size` exposes the default decision for tests;
`gqa_decode_shape_eligible` is true when it selects a nonzero partition.

Kernel block size is the view after hybrid-cache translation: a 1056-token
scheduler page selects block32, while a 784- or 528-token page selects
block16. Translated page IDs address their kernel-sized subpages even when
upstream K/V storage remains unreshaped.

Every eligible call additionally requires:

- One pure-decode request, with `num_decode_requests` equal to 1 or omitted.
- A verification window of at most 1.
- Matching FP16/BF16 query, key-cache and value-cache types.
- A kernel page size allowed above and sufficient reducer shared memory.
- No TurboQuant, attention sinks, logit soft-capping or sliding window.
- A known, positive GPU core count.

Other calls, including multi-request batches and unknown core counts, use the
established attention family. Model context limits, cache capacity and
primitive resource limits still apply; there is no extra 128K ceiling.
The 16/4/256 geometry remains excluded because its previous full-model
numerical validation did not support expanding the default scope.

## Disable switch

Set `VLLM_METAL_DISABLE_GQA_DECODE=1` before starting the server to keep
eligible requests on the established path. This switch provides an A/B and
operational fallback; it cannot enable an otherwise ineligible call.

There is no startup calibration or disk-cached performance threshold for
this gate. `VLLM_METAL_GQA_AUTOTUNE` and the former mutable gate-parameter
APIs are no longer used.

## Validation

`tests/test_gqa_paged_decode.py` evaluates the primitive and checks
`last_paged_dispatch()` alongside the numerical reference. Positive cases
must actually report `gqa_decode`; partition tests additionally check
the executed 64/128/256/512 specialization through
`last_gqa_partition_size()`. Boundary, multi-request, verification, feature,
and disabled cases must report the appropriate established family, including
upstream mixed prefill/decode.
References include independent grouped CPU FP32 attention and native MLX SDPA.
All four geometries and both cache dtypes are checked across the former
128K boundary, at 192K, and at 256K including a partial final partition.
Shared-storage tests use upstream-allocated
K/V views, non-contiguous page tables, native writes, prefix-page copying,
source-page clearing and a subsequent decode write. Dominant attention rows
make missing writes observable even in a long context.
The 1056-token upstream-page case also exercises the real block-table
translation and unreshaped K/V storage: the block32 kernel addresses each
translated page with its 32-token stride.
`tests/test_attention_sdpa.py` checks that the environment switch and scheduler
decode count reach the primitive.

Positive route tests need a reported GPU core count and sufficient partition
grid. They skip on hosts that cannot enable GQA, including some virtual CI
GPUs; the unknown-core fallback test runs there instead. Unconditional
feature/dtype/page-size fallback tests do not require core-count detection.
The library-availability check loads all 24 active GQA specializations and
their matching reducers even when the reported core count is unavailable.
Validate the positive path on a capable GPU before reporting GQA coverage.

`last_paged_dispatch()` records the last family selected in the process. It
is suitable for serial tests and isolated worker checks; it is not
per-request telemetry for concurrent serving. Evaluate the operation before
reading it, because MLX builds graphs lazily.

For performance validation, rebuild native artifacts from the tested
revision and use the real `vllm serve` process topology. Record the model,
runtime versions, hardware, warmup, KV lengths, repetitions, and background
GPU activity; compare enabled and disabled arms with actual worker-side
dispatch checks. Separate primitive timing from HTTP decode throughput,
and keep noisy measurements visible. The
[macOS benchmarking discussion](https://github.com/vllm-project/vllm-metal/issues/713)
explains why in-process engine measurements and short probes are not
interchangeable with serving results.

## Why the broader gate was removed

An earlier design combined a partition-occupancy floor with a scalar
potential-KV-reread proxy and fitted a threshold at startup. Additional
geometry, submission-mode, and load tests did not justify extending that
single threshold to all shader-supported cases. The proxy is not measured
DRAM traffic: each simdgroup still issues its own K/V loads, and the actual
traffic depends on cache residency. Grouping related heads can improve
locality without an explicit cross-simdgroup KV broadcast.

A separate experiment added paired timing, confirmation runs, numerical
checks, telemetry, and invocation-bound permits with expiry, revocation,
and native validation before encoding. It rejected known invalidation but
did not establish performance for later requests after resource conditions
changed. Tests in which requests always fell back did not validate positive
GQA performance. That experimental machinery is not part of this routing
implementation.

These limitations motivate the explicit scope above; they do not establish
that a more general optimization is impossible. Expanding automatic routing
requires its own dispatch, numerical, and real-serving benchmark evidence.
