# Capability Matrix and Runner-Contract Status

This page tracks what each backend supports under which upstream vLLM
runner contract, which test anchors each claim, and what remains
unverified. It is the coordination surface for the MRv1 → MRV2
migration (upstream [vllm#63373] removes the V1 model runner stack):
owners, target dates, and per-item progress live in the tracking issue
linked at the bottom; this document holds only stable, reviewable
criteria and the matrix itself.

## Axes

- **Backend**: MLX (default) and PyTorch-MPS (opt-in,
  `VLLM_METAL_BACKEND=mps`; runs on upstream MRV2 directly).
- **Upstream vLLM**: the pinned release (`.github/vllm-release-tag.commit`),
  fixed main, and the [vllm#63373] head (the post-V1-removal world).
- **Runner contract**: the V1 scheduler shape (current MLX default,
  `VLLM_USE_V2_MODEL_RUNNER=0`) and the V2 scheduler shape. The plugin
  rejects `VLLM_USE_V2_MODEL_RUNNER=1` outside MPS today, so MLX reaches
  the V2 shape in tests by setting `scheduler.use_v2_model_runner =
  True` directly (the #1121 approach), not via the environment
  variable; MPS always runs MRV2.

## Status legend

| Mark | Meaning |
|---|---|
| ✅ | Supported on the pinned release, with a test anchor |
| 🟡 | Partial or in flight (open PR / validation pending) |
| ⛔ | Blocked upstream |
| — | Not applicable to that backend |
| ❓ | Believed supported but not test-anchored; tracked in the issue |

## Matrix

Test paths are relative to `tests/`. "V2 shape" for MLX means the V2
scheduler contract with the self-built `MetalModelRunner`; it is not
the default until the migration flips it.

| Capability | MLX · V1 shape | MLX · V2 shape | PyTorch-MPS | Test anchors |
|---|---|---|---|---|
| Text decode + continuous batching | ✅ | 🟡 validation pending | ✅ | `test_v1_model_runner_generate.py`, `test_mps_backend.py` |
| Chunked prefill | ✅ | 🟡 validation pending | ✅ | `test_v1_model_runner_generate.py` |
| Preemption & readmission | ✅ | ✅ (#1121/#1148, tested under forced shape) | ✅ (upstream native) | `test_v1_model_runner_generate.py` |
| Prefix caching | ✅ | 🟡 validation pending | ✅ | `test_paged_prefix_caching.py`, `test_paged_prefix_caching_e2e.py` |
| KV connector (`ActiveKVConnector`) | ✅ (#1053/#1095) | ✅ (same code path) | ✅ | `test_metal_kv_connector.py` |
| KV offload (+ FS tier) | ✅ | 🟡 validation pending | — | `test_kv_offload_worker.py`, `test_kv_offload_tiering.py`, `test_kv_offload_connector_step_order.py` |
| LoRA / QLoRA | ✅ | 🟡 validation pending | — | `test_lora.py`, `test_qlora.py` |
| Multimodal (Gemma4 incl. audio, Qwen3-VL, PaddleOCR-VL) | ✅ | 🟡 validation pending | ❓ | `tests/v1/mm/` |
| Spec decode: ngram | ✅ | 🟡 | 🟡 EAGLE3 in flight (#1179) | `test_ngram_spec_decode.py`, `test_spec_decode_metadata.py` |
| Spec decode: draft-model / EAGLE | ✅ | 🟡 | 🟡 (#1179) | `test_v1_model_runner_generate.py` (spec sections) |
| Spec decode: Gemma4 MTP | ✅ | 🟡 | — | `test_v1_model_runner_generate.py` |
| Spec decode: DSpark / DFlash / block-draft | ✅ (DSpark is upstream-V2-only and served via the V1 guard today; DFlash exists in both upstream stacks; block-draft is Metal-local) | 🟡 (DSpark unblocks at the flip; the rest pending validation) | — | `test_dspark_proposer.py` |
| Spec decode: suffix | 🟡 (#1144 open) | 🟡 (upstream `suffix` is V1-only until [vllm#60339] lands; decision tracked) | — | — |
| Structured output (xgrammar) | ✅ | 🟡 validation pending | ❓ | `test_grammar_bitmask.py` |
| Pooling (embedding / classification) | ✅ | 🟡 validation pending | ❓ | `test_v1_pooling.py` |
| Speech-to-text | ✅ | — (own runner, unaffected) | — | `test_v1_stt_integration.py` |
| Diffusion (DiffusionGemma) | ✅ (upstream-V2-only; served via the V1 guard today) | ✅ natively once flipped | — | `test_diffusion.py` |
| Pipeline parallelism | ❓ (wiring exists; no test runs PP > 1) | ❓ | — | — |
| Sleep / wake (level 1) | ✅ | 🟡 validation pending | — | `test_v1_worker.py` |
| Prompt logprobs / logprobs | ✅ | 🟡 | ✅ (#1001) | `test_prompt_logprobs.py`, MPS parity tool |
| logit_bias & min_tokens | 🟡 (#1096 open) | 🟡 | 🟡 | — |
| thinking_token_budget | 🟡 (#1163 open; note [vllm#63373] removes `vllm.v1.sample.thinking_budget_state`) | 🟡 | 🟡 | — |

Items marked 🟡 under "MLX · V2 shape" are the migration's validation
backlog (the tracking issue), not open product questions: after the
contract flip the V1-shape column disappears and these must be ✅ or
explicitly rejected.

## Runner-contract differences that matter here

- Resumed (preempted) requests come back as **new** `NewRequestData`
  carrying the full token history; the V1 incremental
  `resumed_req_ids` / `prev_step_scheduled_req_ids` /
  `all_token_ids`-delta channel is deleted by [vllm#63373]. MLX
  readmission follows this shape since #1121/#1148.
- The V2 scheduler may skip drafting on steps where no request can use
  drafts (`can_skip_drafts`); spec-decode controllers must tolerate
  empty draft steps.
- Pooling models keep the Engine/Worker output contract; how a backend
  organizes pooling internally (upstream's worker-level two-phase
  `pool()` call is an implementation detail) is free.
- `VLLM_USE_V2_MODEL_RUNNER` itself disappears with [vllm#63373]; the
  plugin's per-backend default (MLX `0` / MPS `1`) has a bounded
  lifetime.

## Deletion criteria (stable; decisions logged in the tracking issue)

Three layers with different lifetimes; "adapted to the new contract"
is never by itself a reason to delete a layer:

1. **V1 compatibility shims** (resumed-request adapter,
   `patches/v1_runner_guard.py`, the forced
   `VLLM_USE_V2_MODEL_RUNNER=0` default): deletable once the MLX
   backend runs the V2 scheduler contract by default and the matrix
   rows it unblocks are green.
2. **Vendored V1 modules** (`vllm.v1.sample.sampler`,
   `...logits_processor`, `...metadata`,
   `...spec_decode.ngram_proposer`; pinned by
   `tests/test_v1_import_surface.py`): deletable when a shared
   replacement (own sampling stack / inlined matcher) covers them.
   The pinned list shrinking to empty is the machine-checked signal.
3. **The self-built `MetalModelRunner`**: deletion is on the agenda
   only if a shared-runtime path covers its supported surface with no
   performance regression; until then it stays regardless of contract
   flips.

## Canary lanes

| Lane | What it tests | Trigger | Gating |
|---|---|---|---|
| Pinned release | This matrix's ✅ rows, incl. the contract regressions | Every PR (`ci.yml` test job) | Blocking |
| Fixed vLLM main | Drift of the above against a pinned main commit | Scheduled (to be added) | Non-blocking, deduplicated alerts |
| [vllm#63373] head | The post-removal world; import probe uses `tests/test_v1_import_surface.py`'s pinned list to tell known blockers from new failures | Scheduled (to be added) | Non-blocking; known-blocker / new-failure / blocker-gone / not-run states kept distinct |

Runner resources, frequency, and trigger permissions for the two
scheduled lanes are being coordinated in the tracking issue.

## Tracking

Owners, dates, per-item progress, and the current unverified (❓/🟡)
backlog: tracking issue (to be linked when filed). This document
changes only by PR.

[vllm#63373]: https://github.com/vllm-project/vllm/pull/63373
[vllm#60339]: https://github.com/vllm-project/vllm/pull/60339
