# SPDX-License-Identifier: Apache-2.0
"""Pin the ``vllm.v1`` import surface that vLLM's MRv1 removal deletes.

vllm#63373 removes vLLM's Model Runner V1 stack. The MLX backend still
imports a few of the removed modules while it serves on the V1 runner
contract, so a bump past the removal breaks at import time. These tests
pin the exact remaining surface: every hit listed in ``EXPECTED_REMOVED``
is there on purpose, and the migration PRs (own sampling stack, own
logits processors, inline the ngram matcher) shrink it to empty. A new
import of a removed module fails here, so the surface cannot grow
silently mid-migration, and the canary probe for vllm#63373 reuses this
list to tell known blockers from new failures. Planned additions update
the pin in the same PR — #1144 (suffix decoding) will add
``vllm.v1.spec_decode.suffix_decoding`` when it lands.

``REMOVED_BY_63373`` lists only modules under ``vllm.v1`` that this
package actually imports; the upstream PR removes more (the full V1
runner, proposers, and sampler stack) that we never touch.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "vllm_metal"

# Modules deleted by vllm#63373 (files with status "removed" in the PR)
# that vllm_metal imports today. Shrinks to empty as the migration lands.
REMOVED_BY_63373: dict[str, list[str]] = {
    "vllm.v1.sample.sampler": [
        "vllm_metal/pytorch_backend/runtime.py",
        "vllm_metal/stt/sampling.py",
        "vllm_metal/stt/whisper/transcriber.py",
        "vllm_metal/v1/model_runner.py",
        "vllm_metal/v1/prompt_logprobs.py",
        "vllm_metal/v1/sampling_batch.py",
        "vllm_metal/v1/stt_model_runner.py",
    ],
    "vllm.v1.sample.metadata": [
        "vllm_metal/v1/sampling_batch.py",
    ],
    "vllm.v1.sample.logits_processor": [
        "vllm_metal/v1/logits_processors.py",
        "vllm_metal/v1/model_runner.py",
        "vllm_metal/v1/sampling_batch.py",
    ],
    "vllm.v1.spec_decode.ngram_proposer": [
        "vllm_metal/v1/ngram_proposer.py",
    ],
}


def _imported_vllm_paths(tree: ast.AST) -> set[str]:
    """Collect ``vllm.*`` module paths a file imports, statically.

    Covers ``import vllm.x.y``, ``from vllm.x import y`` (recording both
    the package and the ``vllm.x.y`` candidate for each imported name),
    and their ``as`` aliases. Relative imports cannot reach ``vllm``.
    """
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "vllm":
                    paths.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] != "vllm":
                continue
            paths.add(module)
            for alias in node.names:
                if alias.name != "*":
                    paths.add(f"{module}.{alias.name}")
    return paths


def _hits_by_file() -> dict[str, list[str]]:
    """Map each removed module to the package files importing it."""
    hits: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = _imported_vllm_paths(tree)
        rel = path.relative_to(REPO_ROOT).as_posix()
        for removed in REMOVED_BY_63373:
            if any(imp == removed or imp.startswith(removed + ".") for imp in imported):
                hits.setdefault(removed, []).append(rel)
    return hits


def test_removed_module_import_surface_is_pinned() -> None:
    """Every import of a removed module is listed, and only those remain.

    Fails in both directions: a new dependency on a removed module must
    be justified here first, and a migration PR that removes one updates
    this table (shrinking it) in the same change.
    """
    hits = _hits_by_file()
    expected = {m: sorted(files) for m, files in REMOVED_BY_63373.items()}
    assert hits == expected, (
        "vllm.v1 import surface changed; update REMOVED_BY_63373 in the "
        "same PR. added: "
        f"{ {m: f for m, f in hits.items() if f != expected.get(m)} } ; "
        f"removed: { {m: f for m, f in expected.items() if f != hits.get(m)} }"
    )


@pytest.mark.parametrize("module", sorted(REMOVED_BY_63373))
def test_pinned_module_is_importable(module: str) -> None:
    """Each pinned module still exists in the installed vLLM release.

    Failing here one module at a time is the first, clearest signal
    after a bump past the removal — instead of scattered import errors
    across the package. Stale pin entries (modules nothing imports
    anymore) are caught by the surface test above, not here.
    """
    importlib.import_module(module)
