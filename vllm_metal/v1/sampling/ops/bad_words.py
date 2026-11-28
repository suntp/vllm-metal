# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm b46907cce0f7 (== v0.33.0), Apache-2.0:
# vllm/v1/sample/ops/bad_words.py,
# which vllm-project/vllm#63373 removes. The suffix matcher is shared with MLX.

from collections.abc import Iterator

import torch

_SMALLEST_LOGIT = float("-inf")


def matching_bad_word_token_ids(
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> Iterator[int]:
    """Yield the final token of each bad word whose prefix matches the suffix."""
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        if actual_prefix == bad_word_ids[:prefix_length]:
            yield bad_word_ids[-1]


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    for token_id in matching_bad_word_token_ids(bad_words_token_ids, past_tokens_ids):
        # Assign to slice to avoid cpu->gpu sync.
        logits[token_id : token_id + 1] = _SMALLEST_LOGIT


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])


def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> None:
    start_idx = 0
    remaining = len(bad_words_token_ids)
    for i, n in enumerate(num_draft_tokens):
        if (bad_words_ids := bad_words_token_ids.get(i)) is not None:
            for draft_idx in range(start_idx, start_idx + n):
                _apply_bad_words_single_batch(
                    logits[draft_idx],
                    bad_words_ids,
                    past_tokens_ids[draft_idx],
                )
            remaining -= 1
            if not remaining:
                break
        start_idx += n
