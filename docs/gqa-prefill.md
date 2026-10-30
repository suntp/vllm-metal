# Grouped NAX prefill

The MLX NAX dispatcher can process three or six query heads from one six-head
KV group in a BQ16 threadgroup. `VLLM_METAL_NAX_GQA_PREFILL` defaults to `1`;
set it to `0` and restart workers to retain upstream NAX.

Eligibility is based on tensor geometry and attention features, not a model
name: BF16, head dimension 256, Q/KV head ratio 6, kernel page 16, one ordinary
causal sequence, no mixed suffix offset, sliding window, softcap or sinks.
Kernel pages may be translated from larger physical cache pages. Other shapes
retain their established route. Each dispatch executes one selected kernel.

Both layouts preserve the per-head calculation. h3 uses 96 threads and h6 uses
192; neither is always fastest. Complete query/KV tiles omit bounds checks,
while partial tiles and causal masking remain guarded. Whole-request alignment
is not required.

## Admission and calibration

`nax_prefill_work` counts the KV tiles visited by complete and partial query
blocks. It keeps issued work, useful work, guarded-query work and isolated KV
edges separate. `nax_gqa_work_score` compares each eligible grouped layout with
the actual upstream pipeline, using query/KV heads and detected GPU cores to
limit credit for parallel work. These counts are source-work proxies, not DRAM
traffic, register allocation, measured occupancy or predicted milliseconds.

The coefficients are manual conservative calibration weights. They were
calibrated with paired native-kernel comparisons on M5 (10 GPU cores), M5 Pro
(20) and M5 Max (40), varying Q, prefix length, Q/KV head count, page alignment
and execution order. The retained comparison includes upstream, h3 and h6;
beating upstream alone does not establish the fastest layout.

| Score component | Weights in the implementation | What the measurements constrain |
|---|---|---|
| Optimized upstream: padding, tail critical path, full work and guarded-Q savings | 13, 8, 2.5, 15 | A smaller Q tile can remove wasted/tail work, but grouping still costs extra KV streams and state |
| Optimized upstream: stream/state and guarded-Q cost | 10/h, 3.5, 16 | Do not assume shared KV makes all per-head loads disappear |
| Guarded upstream: partial work, critical path, reuse and non-tail guarded work | 4, 0.875, 3.5, 4 | Long-prefix short queries differ from dense cold-prefill blocks |
| Guarded upstream: lost grid capacity and additional group state | 10.5, 32, 3, 3.5, 0.75 | Wider groups must supply enough independent work for the GPU |
| Fixed/core cost, KV edges and guarded Q tails | 11.5, 2, 5 | Preserve a margin for launch/edge/tail work rather than crediting every complete tile as a win |
| Admission margin | 0.5 plus capacity-scaled tail reserve | A positive score alone is insufficient |

The source-work count was separately checked against an independent loop
oracle. Correct work counts do not make empirical coefficients universal.
Head dimension is an eligibility boundary here; this score is not a
cross-dimension performance model.

The six-head candidate normally requires at least one complete group per core
and enough complete SIMD-group work or KV work. A dense-work allowance requires
at least 32 complete h6 groups per core and at least 95% full-tile work. The
additional guarded h6 allowance is restricted to the measured 10-core range;
scaling that allowance by group/core counts alone produced regressions.

The R4 calibration used base `93562f1` and NAX library `633139ae…`. The following
paired native measurements explain why the wider guarded allowance was not
extended to 20/40 cores. These are kernel timings, not model TTFT:

| GPU cores | Workload | Candidate/reference | Two-round latency change | Decision |
|---:|---|---|---|---|
| 10 | H24/K4, Q144/P31360 | h6 / retained h3 | -12.51%, -11.83% | Retain the 10-core guarded allowance |
| 20 | H48/K8, Q144/P24577 | broader h6 / retained h3 | +36.29%, +36.65% | Reject broad guarded admission |
| 40 | H96/K16, Q144/P24577 | broader h6 / retained h3 | +33.15%, +36.63% | Reject broad guarded admission |
| 80 | No available NAX execution host | Not measured | No performance result | Do not infer performance from CPU policy tests |

These rows have different head geometries and must not be pooled into a
cross-device speedup. Full results, exact measured artifacts, final-source
regressions and serving measurements belong in the accompanying PR evidence.

Known limitations remain: some odd-K selections trail historical experimental
layouts even while beating upstream; D128/D512 and other ratios need their own
compiled layouts and resource validation. The separate GQA8 implementation is
not part of this change. Other models with matching geometry have operator
coverage, not a blanket whole-model performance guarantee.

## Context length

There is no hard 32K total-KV admission limit. Removing the earlier calibration
cap was checked on M5/Pro/Max with 100 paired operator case-runs across the
32K boundary, 64K and approximately 128K KV. Outputs were bitwise equal and
no admitted path showed a repeated significant loss above 2% against actual
upstream or h3 in that matrix. This is operator evidence, not 128K model E2E
or a speedup guarantee at arbitrary lengths. Complete-work density, tail cost
and core capacity still gate admission.
