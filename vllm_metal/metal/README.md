# Metal Kernel Sources

Native paged-attention and linear-attention Metal shaders for the MLX backend.

All shader sources live in `kernels_v2/`.

**License / provenance:** Portions of `utils.metal` and `pagedattention.metal`
are adapted from Apple's [MLX](https://github.com/ml-explore/mlx) framework
(Apache-2.0, © 2023 Apple Inc.); `pagedattention.metal` also adapts portions of
the [vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0).
`turboquant.metal`, `pagedattention_tiled.metal`, and `mla.metal` are
vLLM-project Apache-2.0 sources. `pagedattention_nax.metal` adapts MLX's
`steel_attention_nax` (MIT, © 2025 Apple Inc.).

## How the shaders are compiled

The source builders concatenate shaders in the order below, stripping local
`#include "…"` directives.

### 1. C++ `_paged_ops` extension: `__init__.py` and `build.py`

By default, `get_ops()` loads the prebuilt nanobind extension and three
required `.metallib` libraries, plus optional NAX support. With
`VLLM_METAL_BUILD_FROM_SOURCE=1`, it builds the extension and compiles the
shader sources in-process through MLX. Packaging uses the same source
builders to precompile the `.metallib` files via `python -m vllm_metal.metal.build`.

| Library | Source builder | Concatenated sources (in order) |
|---------|----------------|----------------------------------|
| **v2 paged attention** | `_build_v2_paged_attention_source` | `#define VLLM_METAL_PARTITION_SIZE` · `#define VLLM_METAL_PA_WINDOW_ROWS` · `float8.metal` · `utils.metal` · `turboquant.metal` · `reshape_and_cache.metal` · `pagedattention.metal` · `pagedattention_tiled.metal` |
| **GDN state operations** | `_build_gdn_source` | `utils.metal` · `gdn_linear_attention.metal` · `gdn_state_scatter.metal` |
| **MLA** | `_build_mla_paged_attention_source` | `utils.metal` · `mla.metal` |
| **NAX prefill** | `_build_nax_source` | `pagedattention_nax.metal` (optional; macOS 26.2 SDK or newer to build) |

### 2. `mx.fast.metal_kernel` snippets: `attention/impls/gdn_lazy.py`

The lazy GDN decode and prefill paths compile four shaders directly through
MLX, via `_read_v2_metal_source`:

| Shader | Compiled kernel name |
|--------|----------------------|
| `gdn_conv1d_silu_decode.metal` | `gdn_conv1d_silu_decode_v2` |
| `gdn_conv1d_silu_prefill.metal` | `gdn_conv1d_silu_prefill_v2` |
| `gdn_recurrent_decode.metal` | `gdn_recurrent_decode_v2` |
| `gdn_recurrent_prefill.metal` | `gdn_recurrent_prefill_v2` |

## File reference

| File | Role |
|------|------|
| `float8.metal` | FP8 E4M3 encode/decode helpers. Concatenated before `utils.metal`. |
| `utils.metal` | Generic vector types and shared helpers; includes `float8.metal`. Adapted from Apple MLX. |
| `reshape_and_cache.metal` | Scatter projected K/V into the paged cache through an MLX Primitive. |
| `pagedattention.metal` | Per-token paged-attention kernel with online softmax and sink support. Adapted from Apple MLX + vLLM. |
| `pagedattention_tiled.metal` | Tiled Flash-Attention-style kernel using simdgroup 8×8 MMA; shares the paged-attention library. |
| `pagedattention_nax.metal` | Optional M5 NAX paged-prefill kernel using MPP tensor operations. |
| `turboquant.metal` | TurboQuant encode/scatter and dequantization helpers: asymmetric uniform K quantization and Lloyd-Max V quantization with FWHT rotation. |
| `mla.metal` | Single-pass paged Multi-head Latent Attention. |
| `gdn_linear_attention.metal` | GDN (gated delta-net) linear-attention kernel for hybrid models. |
| `gdn_state_scatter.metal` | Scatter or zero cache/state rows, including updates to shared upstream storage. |
| `gdn_conv1d_silu_decode.metal` | Lazy GDN decode: causal conv1d + SiLU. |
| `gdn_conv1d_silu_prefill.metal` | Lazy GDN prefill: causal conv1d + SiLU. |
| `gdn_recurrent_decode.metal` | Lazy GDN decode: recurrent state update. |
| `gdn_recurrent_prefill.metal` | Lazy GDN prefill: recurrent state updates over packed sequences. |

### D512 NAX prefill

The split pipeline divides D512's output columns across two threadgroups,
each accumulating 256 columns. Both read the full 512-column Q/K vectors
and repeat QK and online softmax, then write disjoint output slices. This
halves live output accumulators without scratch buffers or a reduction pass.
Complete KV tiles in full Q blocks omit bounds checks; partial Q blocks and
partial KV tiles retain the guarded loads and masks.

The host chooses the pipeline from CPU sequence metadata. Callers without
sequence metadata and the MPS launcher retain the original D512 loop.

### NAX prefill admission

Full-attention batches use the same workload score for D64, D96, D128, D256
and D512. It applies to single and multiple prefills, including cached
appends and mixed decode/prefill batches. There is no prefix-length floor,
device-name table or Q/KV-head-pair table. Native eligibility still owns
architecture, dtype, page support and whether decode is separated.

For each sequence, count complete 64-row Q blocks `Fi` and the 32-token KV
tiles visited by complete / partial Q blocks, `Wfi` / `Wpi`. Keep the causal
extent of each sequence; query rows from different requests never form one
complete block. Define:

- `F = sum(Fi)`, `Wf = sum(Wfi)`, `Wp = sum(Wpi)`.
- `U = sum(Wfi / Fi for Fi > 0)`: the sum of per-sequence mean KV-loop depths.
- `H`, `K`, `C`: query heads, KV heads and detected GPU cores.
- `d = max(D, 128) / 128`, `s = ceil(D / 256)`.
- `r = sqrt(H/K)+1`: a conservative shared-KV reuse weight.
- `p = H*F / C + 3*(s-1)`: a shared capacity score for complete blocks and
  the kernel's actual output slices. A nonpositive score keeps the original loop.

The final choice is one net-benefit comparison:

```text
benefit = d² * [Wf + (s-1) * min(1, C/H) * (Wf+Wp)]
guarded_cost = 8*Wp * max(3*(1+K/H)/p, 1/r)
stream_cost = 10*(D/128)*U / [r * min(3, p)]
split = benefit >= max(guarded_cost, stream_cost)
```

`U` prevents independent short KV streams from being mistaken for one long
prefill merely because their aggregate tile counts match. The stream term
accounts for width and partial reuse of KV across query heads; its capacity
discount stops at saturation. Guarded work retains a KV-traffic floor even
when a larger batch supplies many parallel blocks. The slice term accounts for the existing
register-pressure reduction and repeated QK/softmax work. These shared
coefficients form an empirical work score, not predicted milliseconds or a
measured occupancy value. Partial Q blocks and partial KV tiles retain guards.

The host supplies scores for both the whole batch and the potential prefill
suffix. Only native dispatch decides whether ordinary decode rows are split
off, and consumes the suffix score only in that case. This supports a suffix
containing multiple prefills without duplicating the native decode-work gate.
Pure decode and YOCO one-row views retain their decode routes. Sliding-window
attention and callers missing head/core metadata keep the established rule:
`F >= 4`, with `Wf >= Wp` for D256 or `Wf >= 8*Wp` otherwise.

Hardware capacity is captured once per forward. Tile counts and the prefill
subset come from existing CPU metadata without GPU readback. The native
kernel, output slices, cache allocation and decode separation are unchanged.
