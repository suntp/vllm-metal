# SPDX-License-Identifier: Apache-2.0
"""Evidence-pack helpers: environment capture, statistics, comparisons.

Every benchmark run writes a machine-readable evidence pack (JSON) so
numbers can be pasted into an issue or PR with their full provenance:
source revision, package versions, model config digest, GPU utilization
samples, the actual dispatch family per run and the A/B comparison.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def git_state() -> dict:
    """Source revision of the checkout the benchmark ran against."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            != ""
        )
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None, "note": "not a git checkout"}
    return {"head": head, "dirty": dirty}


def package_versions() -> dict:
    """Versions of the packages that shape the measured behavior."""
    import importlib.metadata

    names = ("vllm", "vllm-metal", "mlx", "mlx-lm", "transformers", "torch")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_digest(model_path: Path) -> dict:
    """Selected model-config fields plus a digest of the full file."""
    config_path = model_path / "config.json"
    raw = config_path.read_text()
    config = json.loads(raw)
    fields = {}
    for key in (
        "model_type",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "num_hidden_layers",
        "max_position_embeddings",
        "quantization_config",
    ):
        if key in config:
            fields[key] = config[key]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        for key in (
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        ):
            if key in text_config and key not in fields:
                fields[key] = text_config[key]
    return {
        "path": str(model_path.resolve()),
        "config_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "fields": fields,
    }


def median_cv(values: list[float]) -> dict:
    """Median, sample count and coefficient of variation of *values*."""
    if not values:
        return {"n": 0, "median": None, "cv": None}
    median = statistics.median(values)
    if len(values) > 1 and median != 0:
        cv = statistics.stdev(values) / abs(median)
    else:
        cv = None
    return {"n": len(values), "median": median, "cv": cv}


def summarize(
    arms: dict[str, dict[str, list[dict]]],
    arm_order: list[str],
) -> dict:
    """Per-length comparison across arms.

    *arms* maps arm name -> length -> list of run records (each with
    ``decode_tps`` and ``text_sha256``).  The comparison reports median
    tok/s per arm, the on/off speedup and whether every run of every arm
    produced byte-identical output for the length.
    """
    reference_arm = arm_order[0]
    lengths = sorted(arms[reference_arm], key=int)
    comparison: dict[str, dict] = {}
    for length in lengths:
        entry: dict = {"median_tps": {}, "cv": {}, "families": {}}
        for arm, per_length in arms.items():
            runs = per_length.get(length, [])
            entry["median_tps"][arm] = median_cv([run["decode_tps"] for run in runs])
            entry["families"][arm] = sorted(
                {run.get("dispatch_family", "unavailable") for run in runs}
            )
        baseline = entry["median_tps"].get(reference_arm, {}).get("median")
        for arm in arm_order[1:]:
            median = entry["median_tps"].get(arm, {}).get("median")
            if median and baseline:
                entry[f"{arm}_vs_{reference_arm}"] = median / baseline
        hashes = {
            run["text_sha256"] for runs in arms.values() for run in runs.get(length, [])
        }
        entry["all_text_equal"] = len(hashes) == 1
        comparison[length] = entry
    return comparison


def write_json(path: Path, payload: dict) -> None:
    """Write JSON incrementally safe for long benchmark runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def platform_record() -> dict:
    """Minimal platform identity for the evidence pack."""
    import platform

    record: dict = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if record["system"] == "Darwin":
        try:
            record["macos"] = platform.mac_ver()[0]
        except Exception:  # noqa: BLE001 - best-effort identity only
            pass
    return record
