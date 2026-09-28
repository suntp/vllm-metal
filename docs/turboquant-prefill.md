# TurboQuant prefill

TurboQuant normally reads compressed KV directly in the per-token attention
kernel. Substantial prefill chunks can instead gather their referenced cache
pages, dequantize them, and use the existing NAX (M5) or tiled attention kernel.
The same configured key format and value bit width are used for cache writes
and dequantization.

## Routing and memory

A segment qualifies when its number of query tokens is at least
`max(128, head_dim / 2, ceil(256 * num_kv_heads / num_query_heads))`. This is a
conservative crossover rule from the included MHA/GQA/MQA sweep on M5 Pro, not an
online calibration. `VLLM_METAL_TQ_PREFILL=auto` (default) requires NAX availability.
M1–M4 stay on compressed attention unless explicitly enabled with
`VLLM_METAL_TQ_PREFILL=1`; running the tiled backend on M5 does not establish
performance on those GPUs. `VLLM_METAL_TQ_PREFILL=0` disables the lane.
Short suffixes, decode, speculative
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
metadata memo. Dequantized K/V are temporary inputs, not persistent per-layer
caches. Only one batch is materialized, subject to the
`VLLM_METAL_TQ_PREFILL_MAX_MIB` workspace allowance (default **auto**). Auto uses
2% of the device's recommended working-set size, rounded up to 64 MiB, with a
256 MiB floor and a 2 GiB ceiling. This is a stable startup allowance, not a
measurement of currently free memory. A numeric value sets an explicit MiB
limit; `0` disables materialization. The
existing cache planner subtracts this allowance, once, alongside profiled
execution overhead and model weights before reporting KV capacity to upstream
vLLM. It therefore fits **inside** `gpu_memory_utilization`; it is not a new
allocator or an additional per-layer reservation. Set these variables before
starting the worker, rather than changing the budget while a cache is allocated.

Admission accounts for final FP16/BF16 K/V, gather indices and the compact block
table. Its conservative estimate is
`gathered_tokens * (4 * KV_heads * head_dim + 16) + table_entries * 4` bytes.
Mixed selected/fallback batches also charge
`query_tokens * (6 * query_heads * head_dim + 8)` for query/output copies and
reordering indices. Admission can conservatively reject a candidate that would
need a split, even if selecting the entire batch at once could avoid those copies.
The gathered length is rounded to the kernel block size. Requests exceeding the
allowance use compressed attention **before** any dequantization allocation;
smaller candidates can still be selected. Independent histories can batch when
their combined pages fit, and shared physical prefixes count only once. The
original packed query order is restored for mixed selected/fallback batches.

One fused Metal kernel reads the original strided packed cache, unpacks byte
pairs in registers, and writes only the final K/V pair in the query dtype.
There are no context-sized packed gathers or FP32 intermediate arrays. Scale
math and the inverse FWHT remain FP32 in registers to preserve numerical
accuracy; casting the FWHT inputs to BF16 early would change that contract.
The register FWHT follows the existing compressed kernel's shuffle and
snapshot/commit stages. The independent Python decode remains a numerical
reference.

The selected lane evaluates its attention output before returning to model
execution. This releases the large temporary K/V pair before the next layer;
otherwise MLX can retain several pairs in flight, exceeding a single reserved
workspace. Compressed fallback and decode remain lazy. This evaluation boundary
has a scheduling cost, included in both the production-path and whole-model
benchmarks. The allowance covers materialization, not the model's entire memory
footprint; normal projection/attention buffers remain in the existing profiled
execution budget.

For 4 KV heads and dimension 256, a fixed 256 MiB limit admits 65,264 unique
tokens with 16-token kernel pages and one selected request. On the tested
64 GB M5 Pro, auto resolves to 1,088 MiB and admits 262,144 unique tokens in
the same geometry. The allowance is shared across requests: independent
histories add their sizes, while physical prefix pages shared by requests count
once. Different KV head counts, dimensions and mixed-batch copies change these
limits. Histories beyond the allowance still fall back; this is bounded fused
materialization, not streaming attention with constant workspace.

The lane applies to one or multiple requests, including the prefill rows in a
mixed prefill/decode batch. It works with prefix caching enabled or disabled.
After a prefix hit, eligibility depends on the remaining query tokens in the
current scheduler chunk, not the original prompt length. A fully reused prompt
or a short suffix usually stays compressed because little prefill work remains.
Only full-attention SDPA layers using TurboQuant are accelerated; linear/GDN
layers and subsequent one-token decode do not use this lane.

The worker logs the reserved allowance, the first actual lane activation, and
the first budget fallback. Debug logging includes selected/fallback request
counts, gathered tokens and estimated bytes. Kernel block IDs, including IDs
after scheduler-page translation, must fit int32; page/token gather coordinates
avoid subsequently overflowing int32 with an absolute token offset.

## Validation and reproduction

`tests/attention/test_turboquant_prefill.py` exercises `sdpa_forward` with real
Q/K/V projections, fused cache writes and native attention on upstream-allocated
storage. It covers K/V bit widths, FP16/BF16, NAX/tiled selection, nonidentity
tables, shared prefixes, translated and padded pages, mixed-output ordering,
fallbacks, cache read-only calls, budget boundaries, reserved capacity,
cross-layer scratch lifetime, and fused-decode parity with the independent
Python reference for every supported bit width and head dimension. Current-token cache cells
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
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  VLLM_METAL_TQ_PREFILL_MAX_MIB=256 \
  python tools/benchmark/tq_lane_verify.py --suite limits
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite long
```

For model-level comparison, interleave both arms after warmup in one loaded
model. Prefix caching is disabled so repeated prompts still execute prefill:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --arm paired --prompt-tokens 1153 8192 16384 \
  --max-tokens 1 --reps 3 --warmup 1 --output latency.json
```

This tool is an in-process offline benchmark. Its JSON includes actual token
counts, model/quantization choices, dependency versions, generation wall time
and token IDs, cache-hit token counts, and peak additional MLX memory. Statistics
are enabled and TTFT comes from vLLM 0.30's
`RequestStateStats.first_token_latency`, separately from generation wall time.
Every request records actual planner/lane layer calls, selected segments and
maximum workspace. A requested TQ arm with no lane activation or missing TTFT
fails explicitly instead of silently reporting a fallback as an optimization.
Use a separate serving benchmark for network latency and concurrent throughput.

To exercise actual prefix-cache reuse, seed and then reuse a two-block prefix
with suffix lengths 0, 8 and 256, resetting the cache between arms:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --prefix-probe --prompt-tokens 8192 --output prefix.json
```

This records the actual cached tokens and dispatch for both arms. It is a
functional probe, not a repeated latency benchmark; block size and query
thresholds depend on the model. Increase `--prompt-tokens` if two scheduler
blocks plus the suffix exceed that model-length setting.

For a small teacher-forced quality comparison, supply a fixed text corpus:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --quality-text /path/to/wikitext-test.txt \
  --quality-window 1024 --quality-windows 16 --output quality.json
```

The tool tokenizes without additional special tokens, splits the first 16,384
tokens into 16 independent windows, and scores all but the first token of each
window using vLLM prompt logprobs (16,368 scored tokens). Both arms receive the
same ground-truth history, rather than comparing logprobs after greedy outputs
have already diverged. It reports aggregate perplexity, paired-window bootstrap
NLL intervals, top-1 agreement and corpus/token-ID hashes. Record the corpus
source, split and revision with the result. This small subset is a regression
probe, not a full benchmark-quality evaluation. The auxiliary top-1 comparison
uses the first returned rank-1 candidate on ties; it is not a greedy-generation
agreement metric.

Dequantization rounds K/V to the query dtype before attention, and NAX/tiled
accumulation differs from the compressed kernel's deferred value transform.
Numerical tolerance is therefore required; greedy token sequences need not be
identical. Kernel parity and a short model smoke do not establish model-quality
equivalence. A lower perplexity on a small sample also does not establish a
general quality improvement. Streaming attention beyond the workspace limit and using
uncompressed current-chunk K/V remain separate changes that must preserve these
numerical and memory contracts.
