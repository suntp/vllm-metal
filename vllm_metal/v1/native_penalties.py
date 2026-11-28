# SPDX-License-Identifier: Apache-2.0
"""Lazy MLX penalties with vLLM's prompt/output token semantics."""

from collections.abc import Sequence

import mlx.core as mx
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.sampling_policy import has_penalties


def _token_counts(token_ids: Sequence[list[int]], vocab_size: int) -> mx.array | None:
    """Count IDs per row on device or decline unresolved or invalid histories."""
    # The extra bin at vocab_size absorbs padding, as in the Torch sampler.
    if any(ids and (min(ids) < 0 or max(ids) >= vocab_size) for ids in token_ids):
        return None
    counts = mx.zeros((len(token_ids), vocab_size + 1), dtype=mx.int32)
    width = max(map(len, token_ids), default=0)
    if width:
        tokens = mx.array(
            [ids + [vocab_size] * (width - len(ids)) for ids in token_ids],
            dtype=mx.int32,
        )
        rows = mx.arange(len(token_ids), dtype=mx.int32)[:, None]
        counts = counts.at[rows, tokens].add(1)
    return counts[:, :vocab_size]


def apply_native_penalties(
    logits: mx.array,
    params: Sequence[SamplingParams],
    prompt_token_ids: Sequence[list[int]],
    output_token_ids: Sequence[list[int]],
) -> mx.array | None:
    """Apply repetition, frequency, then presence without evaluating logits.

    As in MRv2's ``gpu/sample/penalties.py``, repetition uses prompt or output
    presence. Frequency and presence use committed output counts. A None
    result leaves sampling to Torch without consuming a native key.
    """
    penalized = [has_penalties(sp) for sp in params]
    if not any(penalized):
        return logits
    vocab_size = logits.shape[-1]
    output_counts = _token_counts(
        [
            ids if active else []
            for active, ids in zip(penalized, output_token_ids, strict=True)
        ],
        vocab_size,
    )
    if output_counts is None:
        return None
    logits = logits.astype(mx.float32)
    output_mask = output_counts > 0
    repetition = [sp.repetition_penalty for sp in params]
    if any(value != 1.0 for value in repetition):
        prompt_counts = _token_counts(
            [
                ids if value != 1.0 else []
                for value, ids in zip(repetition, prompt_token_ids, strict=True)
            ],
            vocab_size,
        )
        if prompt_counts is None:
            return None
        scale = mx.where(
            (prompt_counts > 0) | output_mask, mx.array(repetition)[:, None], 1.0
        )
        logits = mx.where(logits > 0, logits / scale, logits * scale)
    frequency = mx.array([sp.frequency_penalty for sp in params])[:, None]
    presence = mx.array([sp.presence_penalty for sp in params])[:, None]
    return logits - frequency * output_counts - presence * output_mask
