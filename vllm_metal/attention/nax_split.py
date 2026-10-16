# SPDX-License-Identifier: Apache-2.0
"""Host-side choice between the two NAX prefill pipelines.

``pagedattention_nax.metal`` compiles a split pipeline, which skips bounds
checks on complete KV tiles of full 64-row Q blocks, and an unsplit one that
keeps the original single guarded loop.  Outputs are identical; only speed
differs.  The split pipeline speeds up full Q blocks but slows the guarded
tiles of partial blocks (a sequence's last block when its length is not a
multiple of 64, and decode rows), and a launch with only a few full blocks
does not fill the GPU enough to hide that.

The choice needs every sequence's query and KV length, which the kernel
dispatcher can only read from the GPU.  The attention context holds them on
the CPU, so the rule is evaluated here once per forward and passed down as
``nax_split_hint``.  Thresholds were calibrated on an M5 Max (BF16, block
size 16); head_size 512 is never split (see ``dispatch_paged_attention_nax``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_BQ = 64  # Q rows per NAX threadgroup
_TILE_KV = 32  # KV tokens per tile
# Fewer full Q blocks than this leave too little split work to pay for the
# slower partial blocks (measured: one to three full blocks lose 1-12%).
MIN_FULL_BLOCKS = 4
# Split only when full-block KV tiles outnumber partial-block KV tiles by
# this factor.  head_size 256 gains ~45% per full tile, head_size <= 128
# only ~15%, so the latter needs a much larger share of full-block work.
_MIN_FULL_TO_PARTIAL = {256: 1}
_MIN_FULL_TO_PARTIAL_DEFAULT = 8


@dataclass(frozen=True)
class NaxTileWork:
    """KV-tile counts per Q-block kind for one forward's batch."""

    full_blocks: int = 0
    full_tiles: int = 0
    partial_tiles: int = 0


def _tiles(kv_end: int) -> int:
    return (kv_end + _TILE_KV - 1) // _TILE_KV


def nax_tile_work(cu_seqlens: Sequence[int], kv_lens: Sequence[int]) -> NaxTileWork:
    """Count causal KV tiles touched by full and partial Q blocks.

    ``cu_seqlens`` are cumulative query rows and ``kv_lens`` each sequence's
    KV length including this step's tokens (``PagedAttentionContext``).
    """
    if len(cu_seqlens) != len(kv_lens) + 1:
        raise ValueError("cu_seqlens must have one more entry than kv_lens")
    full_blocks = full_tiles = partial_tiles = 0
    for i, kv in enumerate(kv_lens):
        q = cu_seqlens[i + 1] - cu_seqlens[i]
        if q <= 0:
            continue
        context = kv - q
        n_full = q // _BQ
        full_blocks += n_full
        for j in range(n_full):
            full_tiles += _tiles(context + (j + 1) * _BQ)
        if q % _BQ:
            partial_tiles += _tiles(kv)
    return NaxTileWork(full_blocks, full_tiles, partial_tiles)


def nax_split_hint(work: NaxTileWork, head_size: int) -> int:
    """1 to use the split pipeline, 0 for the original loop."""
    if head_size >= 512 or work.full_blocks < MIN_FULL_BLOCKS:
        return 0
    factor = _MIN_FULL_TO_PARTIAL.get(head_size, _MIN_FULL_TO_PARTIAL_DEFAULT)
    return int(work.full_tiles >= factor * work.partial_tiles)


def nax_split_kwargs(
    capabilities: Mapping[str, bool],
    work: NaxTileWork | None,
    head_size: int,
    sliding_window: int,
) -> dict[str, int]:
    """``nax_split_hint`` keyword for one prefill ``paged_attention_primitive``.

    Empty when the native build predates the hint.  Without per-sequence
    metadata (``work`` is None) the original loop is used.  Sliding-window
    layers also use it: ``work`` counts the whole causal prefix, which
    overstates the tiles a windowed layer actually visits.
    """
    if not capabilities.get("nax_split_hint", False):
        return {}
    if work is None or sliding_window >= 0:
        return {"nax_split_hint": 0}
    return {"nax_split_hint": nax_split_hint(work, head_size)}
