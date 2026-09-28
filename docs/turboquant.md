# TurboQuant KV Cache Compression

vllm-metal supports TurboQuant-based KV cache compression. Keys use per-block
affine quantization; values use a Walsh–Hadamard rotation followed by per-block
Lloyd-Max quantization. Quantize/dequantize runs natively on Apple Silicon via
MLX and Metal kernels. Quantization is lossy; model-quality impact depends on
the model, bit widths, context length and workload.

## Quick Start

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

TurboQuant is controlled via vLLM's `--additional-config` JSON, not a separate environment variable.

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `turboquant` | `false` | Enable TurboQuant KV cache compression |
| `k_quant` | `"q8_0"` | Key quantization type (see table below) |
| `v_quant` | `"q3_0"` | Value quantization type (Lloyd-Max) |

### Supported Key Quant Types

K uses per-block affine quantization without a Walsh–Hadamard rotation.

| `k_quant` | Bits | Notes |
|-----------|------|-------|
| `q8_0`, `int8`, `uint8` | 8 | Higher-precision key option |
| `q5_0` | 5 | Good quality / size trade-off |
| `q4_0`, `int4`, `uint4` | 4 | Lower-memory key option; validate model quality |
| `int2`, `uint2` | 2 | Aggressive; noticeable quality loss |

### Supported Value Quant Types

V uses Lloyd-Max (non-uniform) quantization with a Walsh–Hadamard rotation. Values are mapped to precomputed centroids per bitwidth.

| `v_quant` | Bits |
|-----------|------|
| `q2_0` | 2 |
| `q3_0` | 3 |
| `q4_0` | 4 |
| `q5_0` | 5 |
| `q8_0` | 8 |

## Compression

Measured on a Qwen3-0.6B-shaped KV cache (28 layers, 4 KV heads, head_dim=128, block_size=16) vs fp16:

| Config | Compression | K mse | V mse |
|--------|-------------|-------|-------|
| `k_quant=q8_0`, `v_quant=q3_0` (default) | **2.56x** | 0.00002 | 0.03241 |
| `k_quant=q5_0`, `v_quant=q3_0` | 3.37x | 0.00154 | 0.03241 |
| `k_quant=q4_0`, `v_quant=q3_0` | 3.76x | 0.00658 | 0.03241 |
| `k_quant=uint2`, `v_quant=q3_0` | 4.92x | 0.16639 | 0.03241 |

At `max_model_len=32768` on Llama-3.2-1B, the default `q8_0/q3_0` configuration frees roughly 2.5x more context for the same KV memory budget.

## Requirements and Caveats

- **MHA and hybrid (SDPA + GDN linear attention) models are supported.** In hybrid models, only the SDPA layers are compressed; GDN recurrent state retains its configured state dtypes and is not quantized by TurboQuant.
- **MLA models are not supported.** Enabling `turboquant` on an MLA model raises `NotImplementedError` at startup rather than silently falling back.
- **Head dim must be 64, 128, 256, or 512** — sizes supported by the FWHT Metal kernel. Models outside this set are not supported yet.
- Quality is model-dependent. For production use, spot-check perplexity with your target config before rolling out aggressive settings (`int2`, `q2_0`).

Three-bit V selects among eight non-uniform centroids for each rotated
coordinate, with per-block scales. Rotation and Lloyd-Max centroids reduce
distortion; they do not make compression lossless. K errors affect attention
weights, while V errors affect their weighted sum. Validate both bit widths
against a BF16-cache baseline with the same model weights and workload.

## Known Quality Floors

The historical observations below are workload-specific, not guarantees for
every model. Compression ratios use the geometry and scale metadata from the
table above. In particular, the lowest-bit settings have produced severe output
degradation and should not be treated as interchangeable serving presets.

| Config | Compression | Quality guidance |
|--------|-------------|------------------|
| `q8_0` / `q3_0` | 2.56x | Default bit widths; validate against a BF16 cache on the target workload |
| `q8_0` / `q2_0` | 2.78x | A fluency dip has been observed; validate before using it to save memory |
| `q4_0` / `q3_0` | 3.76x | Lower-memory option with model-dependent quality loss |
| `int2` / `q3_0` | 4.92x | **Degraded**: topic drift and numeric artefacts have been observed; capacity benchmarks only |
| `int2` / `q2_0` | 5.82x | **Broken output observed**: degenerate repetition loops; not for serving |

## Examples

### Normal Compression

Use the default bit widths after validating quality on the target workload:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

### Aggressive Compression

For memory-bound workloads where the measured quality loss is acceptable:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q4_0", "v_quant": "q3_0"}'
```

## Prefill Acceleration

Eligible TurboQuant prefills materialize the referenced KV pages and use the
existing NAX or tiled attention kernel. `VLLM_METAL_TQ_PREFILL=auto` enables this
when NAX is available (M5); `1` opts into tiled prefill on other GPUs, and `0`
disables it. The crossover was calibrated on M5 Pro; broader hardware and shape
calibration is required before enabling tiled prefill by default on M1–M4.

Eligibility uses the **new query tokens in the current scheduler chunk**:
`max(128, head_dim / 2, ceil(256 * num_kv_heads / num_query_heads))` or more.
The lane supports single/multiple requests, mixed prefill/decode batches, and
prefix caching on or off. After a prefix hit, only the uncached suffix contributes
query tokens; the attention still reads the relevant cached history. Decode,
short suffixes, speculative verification, FP32, sliding-window and image-block
attention keep their existing paths. Only full-attention SDPA layers are
accelerated; GDN/linear layers are unchanged. Attention sinks remain unsupported
with TurboQuant.

The fused materializer reads the original strided packed cache and writes K/V
directly in FP16/BF16. Unpacking stays in registers; scale math and inverse FWHT
use FP32 arithmetic. There are no context-sized packed gathers or FP32 arrays.
It uses `mx.fast.metal_kernel`, JIT-compiled on first use for each dtype,
geometry and quantization-format specialization; warmed timings exclude this
initial compilation cost.

Dequantized pages are temporary. Shared physical prefixes are decoded once
per layer **per scheduler step**; each prefill chunk re-materializes its
referenced history. Scheduler-owned storage, block tables and their lifetime
remain authoritative.

`VLLM_METAL_TQ_PREFILL_MAX_MIB=auto` reserves 2% of the device's recommended
working set, rounded up to 64 MiB, with a 256 MiB floor and 2 GiB ceiling. A
number overrides the allowance in MiB; `0` disables materialization. Set these
variables before worker startup. The existing cache planner subtracts the
allowance **once, inside `gpu_memory_utilization`**, before allocating KV blocks.
This is a fixed allowance, not permission to borrow currently free memory.

Admission counts final K/V, page indices, block tables and any mixed-batch
query/output copies. Independent histories add their sizes; shared physical
pages count once. An entire batch that fits avoids the split-copy charge.
Oversized histories fall back before materialization, while smaller requests
can still qualify. Each accelerated layer evaluates its output with `mx.eval`
and drains the GPU stream with `mx.synchronize` before returning. This adds a
host synchronization boundary in each scheduler step, so temporary K/V from
successive layers cannot accumulate. Normal model buffers remain in the
existing profiled execution budget.

The worker logs the reserved allowance, first lane activation and first budget
fallback. Debug logs include selected/fallback request counts, gathered tokens
and estimated bytes. Larger histories still need more scratch space: this is
bounded materialization, not constant-memory streaming attention.

## Validation and Reproduction

`tests/attention/test_turboquant_prefill.py` checks the production wrapper using
upstream-allocated storage, real cache writes and native attention. Coverage
includes formats/dtypes, padded and translated pages, shared/independent
histories, mixed-output ordering, fallbacks, admission and cross-layer memory
bounds. Fused dequantization is compared with the independent Python decoder.

The attention microbenchmark uses the same fixture, with interleaved timings,
numerical error, actual dispatch and peak additional MLX memory. Suites are
`crossover`, `geometry` and `long`; add `--tiled` to test the tiled backend:

```bash
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover
```

For whole-model TTFT, run both TQ paths in one warmed model with identical
quantization. The reference disables only the prefill planner. Prefix caching
is off so repeated prompts execute prefill:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --arm paired --prompt-tokens 8192 16384 \
  --max-tokens 1 --reps 3 --warmup 1 --output latency.json
```

JSON records vLLM's `first_token_latency`, wall time, actual layer dispatch,
workspace, MLX memory and runtime versions. TTFT summaries are medians of the
measured repetitions, excluding warmup. Missing TTFT or an inactive requested
TQ lane fails explicitly. This offline tool excludes HTTP and concurrent serving
queues. `--prefix-probe --prompt-tokens 8192` instead seeds and reuses real cached
prefixes with short/long suffixes, recording cache hits and dispatch.

For teacher-forced perplexity, use a fixed corpus and score the same windows in
both paths. This isolates the prefill implementation:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --quality-text /path/to/wikitext-test.txt \
  --quality-window 1024 --quality-windows 16 --output quality.json
```

Repeat with `--arm bf16` for an uncompressed cache, or `--arm tq --v-quant q4_0`
for another TQ format, writing separate output files. Keep weights, tokenization
and windows fixed; verify corpus/token-ID hashes match. Defaults score 16,368
tokens from the first 16 × 1,024-token windows without extra special tokens.
Paired runs report a window-bootstrap NLL interval. Record corpus source, split
and revision with results. TQ-vs-TQ parity does not measure quantization loss
relative to BF16; short-window perplexity does not establish long-context task
accuracy. Different attention arithmetic also means greedy outputs can differ.

For a whole-model long-context execution and memory check:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --arm tq --prompt-tokens 131072 --max-tokens 32 \
  --reps 1 --warmup 0 --progress-interval 30 --output long-context.json
```

This reports progress, actual dispatch, peak active MLX allocation and peak above
the pre-request allocation. A single run establishes neither paired speedup nor
long-context retrieval quality.
