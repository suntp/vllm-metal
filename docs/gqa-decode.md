# GQA decode routing

The paged attention primitive can use `paged_attention_gqa_decode` for a
limited set of single-request, long-context decode calls. Six shader
specializations cover head dimensions 128/256 with kernel block size 16,
plus head dimension 256 with block size 32, in FP16/BF16. The geometry and
performance checks below further restrict routing.

## Automatic eligibility

The following geometries and inclusive minimum KV lengths are eligible:

| Query heads | KV heads | Head dimension | Kernel block size | Minimum KV tokens |
|---:|---:|---:|---:|---:|
| 32 | 8 | 128 | 16 | 32,768 |
| 24 | 4 | 256 | 16 | 32,768 |
| 16 | 2 | 128 | 16 | 65,536 |
| 16 | 2 | 256 | 16 or 32 | 32,768 |

Kernel block size is the view passed to the primitive after hybrid-cache
translation. For example, a 1056-token scheduler block selects the largest
supported divisor, 32, whereas a 528-token block selects 16. Merely adding a
head geometry would not enable GQA for the former view without its block32
specializations. Other geometries with block32 continue to use the
established path.

There is no additional GQA-specific maximum context length. The model's
context limit, cache capacity and the primitive's resource limits still apply.
An eligible request stays on GQA as its KV length grows beyond 131,072;
crossing that former policy boundary does not change kernel families.

Every row additionally requires:

- One pure-decode request, with `num_decode_requests` equal to 1 or omitted.
- A verification window of at most 1.
- Matching FP16 or BF16 query, key-cache, and value-cache types.
- A kernel block size allowed by the geometry table.
- No TurboQuant, attention sinks, logit soft-capping, or sliding-window
  attention.
- Enough partition threadgroups for the conservative occupancy guard:
  `ceil(KV tokens / 512) * KV heads >= 3 * detected GPU cores`.
  An unknown GPU core count falls back.

Calls outside these conditions use the established attention family. That
includes multi-request batches, even if each request individually matches a
row, and lengths below the corresponding minimum. The 16/2/128 geometry starts
at 64K; small measured gains at 32K were not used to widen automatic routing.

The 16/4/256 geometry remains excluded. A MiMo-V2.6-Distill-Qwen-9B-OptiQ
trial showed operator and serving gains, but a repeatable enabled/disabled
32K generation comparison failed the strict top-5 rule. Continuous decode
logits confirmed the divergence on the same token history. Both paths also
differed from a native MLX continuation on that history, so this does not
establish a GQA kernel correctness defect. Further full-model numerical
validation is needed before widening this default scope.

The gate checks geometry rather than model names and does not contain a
device-name allowlist. Its bounds and occupancy guard are empirical choices,
not a claim that GQA wins for every device or competing GPU workload.

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
must actually report `gqa_decode`; boundary, multi-request, verification,
feature, and disabled cases must report the appropriate established family.
References include independent grouped CPU FP32 attention and native MLX SDPA.
All four geometries and both cache dtypes are checked across the former
128K boundary, at 192K, and at 256K including a partial final partition.
Shared-storage tests cover all six specializations using upstream-allocated
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
The library-availability check loads all six specializations even when the
reported core count is unavailable.
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
