# GQA decode routing

The default paged attention path selects among 64-, 128-, 256-, and
512-token GQA partitions for the measured single-request geometries below.
Each partition has FP16/BF16 specializations for head dimensions 128/256
with kernel block16, plus head256 with kernel block32. These form 24 active
shader specializations.

This is a Metal-backend split-KV occupancy heuristic over precompiled
partitions. It uses a fixed work budget, counts only complete partitions,
and greedily tries 512, 256, 128, then 64: among eligible tiers it prefers
fewer splits. This is the same class of backend-internal scheduling choice
as [FA3 split selection](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h),
[FlashInfer planning](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh),
and [ROCm partition choices](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu),
not an equivalent cost model. FA3 also favors fewer splits near its estimated
best efficiency; this implementation does not adopt its wave-efficiency model.
A P1024 tier was evaluated but did not consistently beat P512 across the
tested workloads, so P512 is the largest shipped tier. Some very-long-context
P1024 measurements improved; this is not a claim that P1024 always loses.

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

The GQA producer walks pages rather than tokens. Lane 0 preloads the next
block-table entry and `simd_shuffle` broadcasts it; a 4-token inner step
overlaps independent QK dots with K/V latency. Every tier reads K/V
straight from device memory: at the short-context lengths where the small
tiers are selected the working set is L2-resident, and threadgroup
staging of those tiers measured as a net cost at 32/8 and noise-level at
16/2 on this device, so it is omitted here (see follow-ups). Online
softmax stays in registers.

The producer writes the same log2-space `(max, exp-sum)` plus
epsilon-normalized `tmp_out` contract as split-KV. The shared
`paged_attention_v2_reduce` therefore merges GQA partials unchanged:
this path never modifies the reduce, so the established split-KV decode
paths (multi-request, TurboQuant, sinks, every head size) keep their
existing behavior and numerics byte-for-byte.

Every eligible call additionally requires:

- One pure-decode request, with `num_decode_requests` equal to 1 or omitted.
- A verification window of at most 1.
- Matching FP16/BF16 query, key-cache and value-cache types.
- A kernel page size allowed above.
- No TurboQuant, attention sinks, logit soft-capping or sliding window.
- A known, positive GPU core count.

Other calls, including multi-request batches and unknown core counts, use the
established attention family. Model context limits, cache capacity and
primitive resource limits still apply. The 16/4/256 geometry remains
excluded from default routing.

## Disable switch

Set `VLLM_METAL_DISABLE_GQA_DECODE=1` before starting the server to keep
eligible requests on the established path. This switch provides an A/B and
operational fallback; it cannot enable an otherwise ineligible call.

The switch is captured once when each forward's `PagedAttentionContext` is
created and shared by its layers. The `gqa_disabled` keyword is sent only
when disabling GQA on a native build that supports it. Pre-GQA native builds
already use the established path and receive no new keyword. An unrecognized
GQA build without disable support requires a rebuild rather than silently
ignoring the switch.

## Validation

`tests/test_gqa_paged_decode.py` evaluates the primitive and checks
`last_paged_dispatch()` alongside the numerical reference. Positive cases
must actually report `gqa_decode`; partition tests additionally check
the executed 64/128/256/512 specialization through
`last_gqa_partition_size()`. Boundary, multi-request, verification, feature,
and disabled cases must report the appropriate established family, including
upstream mixed prefill/decode.
References include independent grouped CPU FP32 attention and native MLX SDPA.
All four geometries and both cache dtypes are checked at 128K, 192K,
and 256K, including a partial final partition.
Shared-storage tests use upstream-allocated
K/V views, non-contiguous page tables, native writes, prefix-page copying,
source-page clearing and a subsequent decode write. Dominant attention rows
make missing writes observable even in a long context.
The 1056-token upstream-page case also exercises the real block-table
translation and unreshaped K/V storage: the block32 kernel addresses each
translated page with its 32-token stride.
Staged-path tests cover a 32-token kernel page (two 16-token tiles),
partial last pages with `n_tok` in `{1,5,6,7,13,15}`, and the group-6
round-robin used by 24/4/256.
`tests/test_attention_sdpa.py` checks that the environment switch and scheduler
decode count reach the primitive.

Default-policy positive route tests need a GPU core count and a sufficient
partition grid. Hosts whose IORegistry does not report cores inject a
test-only count through `_override_detected_gpu_core_count_for_test` so
the default selector still runs; the unknown-core fallback test forces
that count to zero. Production serving never calls the override.
Kernel correctness is tested separately through the private
`_gqa_paged_attention_for_test` entry, which selects an explicit partition
without changing global state or the public primitive's routing API.
Its partition is captured in the lazy primitive and its equivalence key.
These tests execute all four partitions, both dtypes, every shipped shader
specialization, tails and upstream shared-page writes/copies even when CI
cannot report GPU cores. Numerical parity does not establish performance
eligibility. Feature/dtype/page-size fallback tests remain independent.
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

## Follow-up work (separate PRs)

These are directions for evaluation after this scoped change, not additional
enablement or performance claims in #715:

1. **Multi-request decode:** extend the grid, reduction and selection policy
   for different request lengths and batch sizes. This broadens the useful
   workload range; batches that already fill the GPU may need different
   choices from the single-request policy.
2. **Mixed prefill/decode:** integrate with the decode prefix split by merged
   [#851](https://github.com/vllm-project/vllm-metal/pull/851), after validating
   multi-request decode. Preserve row offsets, page tables and the prefill
   path; benchmark continuous batching rather than inferring its benefit
   from isolated decode. The integration point is clear, but correctness
   and admission need their own tests.
3. **TurboQuant decode:** evaluate consuming packed KV/scales in the GQA
   kernel. This could complement the prefill optimization in
   [#853](https://github.com/vllm-project/vllm-metal/pull/853), which now handles
   bounded TurboQuant prefill. Avoid assuming that materializing the whole
   history on every decode step is cheap; validate format, memory and quality
   effects.
4. **Sliding windows:** bound loads and split selection by the actual visible
   KV window. This is relevant to Gemma/Mistral-style attention, but their
   geometries and other features such as sinks or soft-capping must also
   satisfy the supported kernel contract.
5. **Speculative verification:** evaluate sharing KV across both grouped heads
   and multiple query rows. The existing verification kernel already shares
   KV across rows; compare against it while preserving causal masks and
   controlling register pressure. A larger gain is possible, not established.
6. **Cross-device staging:** threadgroup K/V staging for the P64/P128
   producer was implemented and measured (two controlled ablations on an
   M5 Pro): a net cost at the 32/8 geometry and noise-level at 16/2, so
   it is removed from the producer. The long-context regime, where the
   KV working set exceeds L2 and staging does pay, needs a GQA-owned
   P512 specialization first. Devices with a smaller L2 may land
   elsewhere on the short-tier trade-off; re-implement and re-measure
   there before treating staging as a cross-device win.
