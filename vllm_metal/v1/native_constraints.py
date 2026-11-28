# SPDX-License-Identifier: Apache-2.0
"""Allowed-token and bad-word masks for the MLX sampler."""

from collections.abc import Sequence

import mlx.core as mx
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.sampling.ops.bad_words import matching_bad_word_token_ids


def apply_native_constraints(
    logits: mx.array,
    params: Sequence[SamplingParams],
    output_token_ids: Sequence[list[int]],
) -> mx.array | None:
    """Build a device mask over real vocabulary columns, without reading logits.

    Return None when the constraints leave a row without candidates, have
    out-of-vocabulary indices, or require unresolved output history. The
    caller can then use the existing sampler with the original logits.
    """
    vocab_size = logits.shape[-1]
    allowed_rows: list[int] = []
    allowed_ids: list[int] = []
    banned_rows: list[int] = []
    banned_ids: list[int] = []
    for row, (sp, output_ids) in enumerate(zip(params, output_token_ids, strict=True)):
        bad_words = sp.bad_words_token_ids
        if bad_words and -1 in output_ids:
            return None
        banned = set(matching_bad_word_token_ids(bad_words or [], output_ids))
        if banned and (min(banned) < 0 or max(banned) >= vocab_size):
            return None
        allowed = sp.allowed_token_ids
        if allowed:
            if min(allowed) < 0 or max(allowed) >= vocab_size:
                return None
            if banned and all(token_id in banned for token_id in allowed):
                return None
            allowed_rows.extend([row] * len(allowed))
            allowed_ids.extend(allowed)
        elif len(banned) == vocab_size:
            return None
        banned_rows.extend([row] * len(banned))
        banned_ids.extend(banned)
    if not allowed_ids and not banned_ids:
        return logits

    blocked = mx.broadcast_to(
        mx.array([bool(sp.allowed_token_ids) for sp in params])[:, None],
        logits.shape,
    )
    if allowed_ids:
        blocked[mx.array(allowed_rows), mx.array(allowed_ids)] = False
    if banned_ids:
        blocked[mx.array(banned_rows), mx.array(banned_ids)] = True
    return mx.where(blocked, -mx.inf, logits)
