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

## Known Quality Floors

Not every supported `(k_quant, v_quant)` combination is suitable for a given
workload. K errors change the attention weights produced by softmax; V errors
enter the weighted sum. V can therefore tolerate lower precision in some
workloads, but its errors need not cancel, and later layers can change their
effect on the answer. Keeping scale and inverse-transform arithmetic in FP32
reduces additional rounding; it cannot recover information discarded by
quantization.

Three-bit V stores one of eight non-uniform centroids for each rotated
coordinate, together with per-block scales. It does not round each original
value to one of eight globally fixed numbers. Rotation distributes concentrated
features across coordinates, and the centroids minimize distortion for the
assumed distribution. These mechanisms reduce error; they do not imply
lossless storage or universal task-quality equivalence.

Historical qualitative observations below are workload-specific, not a
substitute for a BF16-cache comparison on the target model. Compression ratios
use the same geometry as the table above and include scales; padding and hybrid
cache groups can change the resulting scheduler capacity:

| Config | Compression | Qualitative quality | Recommended use |
|--------|-------------|---------------------|-----------------|
| `q8_0` / `q3_0` | 2.56x | Requires target-workload validation | **Default bit widths**, configurable |
| `q8_0` / `q2_0` | 2.78x | Usable; minor fluency dip | Tight-memory serving |
| `q4_0` / `q3_0` | 3.76x | Quality impact depends on the model | Memory-bound workloads |
| `int2` / `q3_0` | 4.92x | **Degraded**: semi-coherent, topic-drift, numeric artefacts ("2018 2018") | Capacity benchmarks only |
| `int2` / `q2_0` | 5.82x | **Broken**: degenerate repetition loops ("concept concept concept…") | Not for serving |

For a reproducible small quality probe, see the
[BF16/TQ cache comparison](turboquant-prefill.md#validation-and-reproduction).
Keep model weights fixed when changing cache precision. Comparing two attention
paths that both use K8/V3 only measures the path change, not the loss from K8/V3
relative to BF16. Perplexity on short windows also does not establish long-context
retrieval, reasoning or code-generation accuracy.

## Examples

### Normal Compression

Using the default bit widths, after validating quality on the target workload:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

### Aggressive Compression

For memory-bound workloads where some quality loss is acceptable:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q4_0", "v_quant": "q3_0"}'
```
