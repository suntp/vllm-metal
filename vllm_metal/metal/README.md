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

The common host tile-work hint enables this path with at least four full
Q blocks and eight times as much full-block as partial-block KV-tile work.
Callers without sequence metadata and the MPS launcher retain the original
D512 loop.

### Cached-prefix NAX admission

The common long-prefill rule runs first. Additional admission uses workload
and detected GPU capacity, without a GPU-model or Q/KV-head-combination table.
Native NAX eligibility still owns architecture, dtype and page support.

The extension is limited to one nonempty full-attention sequence with at
least two query rows. A native mixed-batch split can apply it to its remaining
single-request prefill subset through the existing separate hint. Multi-request
launches, sliding-window attention and YOCO's one-row shared layers retain
existing admission.

Let `B = ceil(query_rows / 64)`, `F` be the number of complete 64-row Q blocks,
and `Wf` / `Wp` count the 32-token KV tiles visited by complete / partial
Q blocks, including the causal extent. Both D512 and D256 require
`Wf + Wp >= 256 * B`: an average of 256 KV tiles per Q block, roughly an 8K-token
span. This conservative floor intentionally leaves smaller workloads unchanged.

- **D512:** that work floor is sufficient. Its existing two 256-column output
  slices reduce live accumulator state for complete and partial Q blocks;
  partial blocks keep their guards and masks.
- **D256:** with `H` query heads and `C` detected GPU cores, also require
  `F >= 2`, `2 * H * F >= 3 * C`, and `2 * H * Wf >= 5 * C * Wp`.
  The first capacity condition requires at least 1.5 complete-block threadgroups
  per core. The second gives sparse grids a conservative margin over guarded
  tail work. When `Wp == 0`, the tail condition passes without division.
  Missing core counts retain the common D256 rule.

The same coefficients apply to every head count and supported NAX GPU; KV
head count is not a policy input. D128 retains the common rule. Hardware
capacity is captured once per forward, and the tile counts and prefill subset
come from existing CPU attention metadata without GPU array readback.
