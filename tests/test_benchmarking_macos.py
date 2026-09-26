# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure logic of the macOS benchmarking harness.

Everything that talks to a live server or the OS is exercised only in
real benchmark runs; here we pin the parsing, prompt assembly, statistics
and comparison logic so evidence packs stay well-formed.
"""

from __future__ import annotations

import argparse

import pytest

from tools.benchmark.macos import evidence, gpu_state
from tools.benchmark.macos.run_ab import (
    assemble_prompt,
    parse_expect_family,
    parse_key_values,
)

_IOREG_SAMPLE = """\
ooo AGXAccelerator <class AGXAccelerator> 1000
  | "Device Utilization %"=57
  | "Performance Statistics"=()
ooo AGXAccelerator <class AGXAccelerator> 1001
  | "Device Utilization %"=3
  | "Accelerator linked"=Yes
"""


def test_parse_utilization_takes_peak_entry() -> None:
    assert gpu_state.parse_utilization_text(_IOREG_SAMPLE) == 57


def test_parse_utilization_missing_entry_is_none() -> None:
    assert gpu_state.parse_utilization_text("no entries here\n") is None


def test_quiet_window_fails_open_when_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_state, "read_gpu_utilization", lambda: None)
    window = gpu_state.wait_quiet_window(settle_s=0)
    assert window.ok
    assert "unknown" in window.note


def test_quiet_window_requires_consecutive_quiet_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([50, 5, 5, 5])
    monkeypatch.setattr(gpu_state, "read_gpu_utilization", lambda: next(readings))
    window = gpu_state.wait_quiet_window(
        12, required_samples=3, sample_interval_s=0, settle_s=0
    )
    assert window.ok
    assert window.samples == [50, 5, 5, 5]


def test_quiet_window_times_out_when_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_state, "read_gpu_utilization", lambda: 80)
    window = gpu_state.wait_quiet_window(
        12, required_samples=3, sample_interval_s=0, settle_s=0, timeout_s=0.05
    )
    assert not window.ok
    assert "timed out" in window.note


def test_assemble_prompt_exact_length_and_layout() -> None:
    prefix = list(range(10, 30))  # 20 tokens
    suffix = [7, 7, 7]
    prompt = assemble_prompt(prefix, suffix, 63, bos_id=1)
    assert len(prompt) == 63
    assert prompt[0] == 1
    assert prompt[-3:] == suffix
    assert prompt[1:-3] == prefix * (59 // 20) + prefix[: 59 % 20]


def test_assemble_prompt_rejects_undersized_length() -> None:
    with pytest.raises(ValueError, match="too small"):
        assemble_prompt(list(range(20)), [7, 7, 7], 10, bos_id=1)


def test_parse_expect_family_and_key_values() -> None:
    assert parse_expect_family("on=gqa_decode, off=per_token_ps512") == {
        "on": "gqa_decode",
        "off": "per_token_ps512",
    }
    assert parse_expect_family(None) == {}
    with pytest.raises(ValueError, match="bad entry"):
        parse_expect_family("on=")
    assert parse_key_values(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_key_values(["novalue"])


def test_median_cv_reports_spread() -> None:
    stats = evidence.median_cv([10.0, 10.2, 9.8, 10.0])
    assert stats["n"] == 4
    assert stats["median"] == 10.0
    assert stats["cv"] is not None and stats["cv"] < 0.05
    assert evidence.median_cv([5.0])["cv"] is None
    assert evidence.median_cv([])["n"] == 0


def _run(arm: str, length: int, index: int, tps: float, text: str) -> dict:
    return {
        "length": length,
        "run": index,
        "decode_tps": tps,
        "text_sha256": text,
        "dispatch_family": "gqa_decode" if arm == "on" else "per_token_ps512",
    }


def test_summarize_computes_speedup_and_equality() -> None:
    arms = {
        "on": {
            "100": [
                _run("on", 100, 0, 20.0, "a"),
                _run("on", 100, 1, 22.0, "a"),
            ]
        },
        "off": {
            "100": [
                _run("off", 100, 0, 10.0, "a"),
                _run("off", 100, 1, 10.0, "b"),
            ]
        },
    }
    comparison = evidence.summarize(arms, ["on", "off"])
    entry = comparison["100"]
    assert entry["median_tps"]["on"]["median"] == 21.0
    assert entry["median_tps"]["off"]["median"] == 10.0
    assert entry["off_vs_on"] == pytest.approx(10.0 / 21.0)
    assert entry["all_text_equal"] is False
    assert entry["families"]["on"] == ["gqa_decode"]


def test_validate_args_rejects_oversized_lengths(tmp_path) -> None:
    from tools.benchmark.macos.run_ab import validate_args

    (tmp_path / "config.json").write_text('{"max_position_embeddings": 100}')
    args = argparse.Namespace(
        model=tmp_path,
        lengths=[90],
        tokens=20,
        runs=1,
        max_model_len=None,
        # Port 0 always binds (ephemeral), so the port check passes and
        # the length check is what fires.
        port=0,
    )
    with pytest.raises(SystemExit, match="exceeds max-model-len"):
        validate_args(args, {"max_position_embeddings": 100})
