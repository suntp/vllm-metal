# SPDX-License-Identifier: Apache-2.0
"""CPU-only adversarial checks for paired scheduler evidence."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _seed_random_generators():
    pass  # Override the repository fixture; no MLX initialization is needed.


@pytest.fixture
def verifier():
    path = Path(__file__).parents[1] / "tools/compare_shared_state_runs.py"
    spec = importlib.util.spec_from_file_location("compare_shared_state_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair(verifier):
    workload, cases = [], {}
    for name in sorted(verifier.CASE_IDS):
        repeat = "long" if name.startswith("repeat_") else None
        spec = {
            "request_id": name,
            "arrival_step": 160 if repeat else 0,
            "prompt_token_ids": [1] * 100,
            "prompt_tokens": 100,
            "max_output_tokens": 3,
            "repeats_prefix_of": repeat,
            "cancel_after_output_tokens": 2 if name == "cancel" else None,
            "cancel_during_prefill": name == "cancel_prefill",
        }
        cancelled = name in {"cancel", "cancel_prefill"}
        workload.append(spec)
        cases[name] = {
            **spec,
            "token_ids": []
            if name == "cancel_prefill"
            else [3, 4]
            if cancelled
            else [3, 4, 5],
            "text": "" if name == "cancel_prefill" else "output",
            "terminal": True,
            "engine_finished": not cancelled,
            "cancelled": cancelled,
            "finish_reason": "abort" if cancelled else "length",
            "stop_reason": None,
            "num_cached_tokens": 32 if repeat else 0,
            "predecessor_finished_at_arrival": bool(repeat),
        }
    cases["cancel_prefill"]["abort_snapshot"] = {
        "num_computed_tokens": 32,
        "num_in_flight_tokens": 32,
    }
    sources = {"vllm_metal/v1/model_runner.py": "a" * 64}
    upstream = dict.fromkeys(
        (
            "v1/kv_cache_interface.py",
            "v1/worker/utils.py",
            "model_executor/layers/mamba/abstract.py",
            "v1/core/sched/scheduler.py",
        ),
        "b" * 64,
    )
    base = {
        "schema_version": 1,
        "status": "completed",
        "evidence_valid": True,
        "scope": "controlled arrivals",
        "scenario": "pressure",
        "max_num_seqs": 4,
        "sampling": {"temperature": 0, "logprobs": None, "ignore_eos": True},
        "collector_sha256": "c" * 64,
        "model": "model-snapshot",
        "requested_blocks": 16,
        "scheduler_blocks": 16,
        "max_model_len": 4096,
        "memory_fraction": 0.25,
        "max_steps": 1024,
        "timeout_seconds": 300,
        "python_executable": "python",
        "source_root": "/unused",
        "source_sha256_before_imports": sources,
        "source_sha256_after_run": sources.copy(),
        "upstream_source_sha256": upstream,
        "upstream_source_sha256_after_run": upstream.copy(),
        "decode_pipeline_submissions": 10,
        "versions": dict.fromkeys(
            ("mlx", "mlx-lm", "mlx-vlm", "numpy", "vllm", "torch"), "1"
        ),
        "vllm_version": "1",
        "block_size": 32,
        "async_scheduling": True,
        "max_concurrent_batches": 2,
        "workload": workload,
        "workload_sha256": verifier.fingerprint(workload),
        "cases": cases,
        "final_gpu_synchronized": True,
        "source_changed_during_run": [],
        "missing_required_coverage": [],
        "scheduler_trace": [
            {
                "preempted": ["long"],
                "resumed": ["long"],
                "finished": ["cancel", "cancel_prefill"],
                "new": [],
            }
        ],
        "forward_trace": [
            {
                "packed_request_order": ["long", "cancel_prefill"],
                "mixed_prefill_decode": True,
                "surviving_request_slot_changes": {"long": {"before": 1, "after": 0}},
                "prefill": [{"request_id": "cancel_prefill", "final_chunk": False}],
            }
        ],
        "events": [
            {
                "kind": "arrival",
                "request_id": "short",
                "other_unfinished": True,
                "engine_step": 3,
            },
            {"kind": "abort_prefill", "request_id": "cancel_prefill"},
        ],
    }
    observed = verifier.observed_coverage(base)
    base["coverage"] = {"required_flags": list(observed), "flags": observed.copy()}
    main, candidate = copy.deepcopy(base), copy.deepcopy(base)
    main.update(
        variant="main",
        checkout_head="1" * 40,
        storage=[],
        legacy_storage=[{"blocks": 16}],
    )
    candidate.update(
        variant="candidate",
        checkout_head="2" * 40,
        storage=[{"blocks": 16}],
        legacy_storage=[],
    )
    return main, candidate


def compare(verifier, pair, **kwargs):
    return verifier.compare_reports(
        *pair,
        main_head="1" * 40,
        candidate_head="2" * 40,
        main_exit_code=0,
        candidate_exit_code=kwargs.pop("candidate_exit_code", 0),
        **kwargs,
    )


def test_complete_pair_passes(verifier, pair):
    assert compare(verifier, pair)["passed"]


def test_normal_budget_capacity_may_differ_but_override_must_match(verifier, pair):
    pair[0]["requested_blocks"] = pair[1]["requested_blocks"] = 0
    pair[1]["scheduler_blocks"] = 425
    assert compare(verifier, pair)["passed"]
    pair[0]["requested_blocks"] = pair[1]["requested_blocks"] = 16
    result = compare(verifier, pair)
    assert not result["passed"]
    assert "explicit override" in result["structural_errors"][0]


@pytest.mark.parametrize("malformed", [None, [], "not-cases"])
def test_malformed_cases_are_structural_errors(verifier, pair, malformed):
    pair[1]["cases"] = malformed
    result = compare(verifier, pair)
    assert not result["passed"]
    assert result["structural_errors"]


def test_duplicate_json_fields_are_rejected(verifier, tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"failed","status":"completed"}')
    with pytest.raises(verifier.EvidenceError, match="duplicate JSON"):
        verifier.load_report(path)


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "missing_sync",
        "wrong_head",
        "unknown_case",
        "missing_case",
        "missing_tokens",
        "source_changed",
        "upstream_changed",
        "upstream_missing",
        "pipeline_forged",
        "cleanup_forged",
        "pressure_forged",
        "partial_output",
        "collector_differs",
        "workload_differs",
        "config_missing",
    ],
)
def test_incomplete_or_inconsistent_evidence_fails_closed(verifier, pair, change):
    candidate = pair[1]
    if change == "failed":
        candidate["status"] = "failed"
    elif change == "missing_sync":
        del candidate["final_gpu_synchronized"]
    elif change == "wrong_head":
        candidate["checkout_head"] = "3" * 40
    elif change == "unknown_case":
        candidate["cases"]["unknown"] = candidate["cases"].pop("short")
    elif change == "missing_case":
        del candidate["cases"]["short"]
    elif change == "missing_tokens":
        del candidate["cases"]["short"]["token_ids"]
    elif change == "source_changed":
        candidate["source_sha256_after_run"]["new.py"] = "d" * 64
    elif change == "upstream_changed":
        candidate["upstream_source_sha256_after_run"]["new.py"] = "d" * 64
    elif change == "upstream_missing":
        del candidate["upstream_source_sha256"]["v1/worker/utils.py"]
        del candidate["upstream_source_sha256_after_run"]["v1/worker/utils.py"]
    elif change == "pipeline_forged":
        candidate["decode_pipeline_submissions"] = 0
    elif change == "cleanup_forged":
        candidate["scheduler_trace"][0]["finished"] = []
    elif change == "pressure_forged":
        candidate["scheduler_trace"][0]["preempted"] = []
    elif change == "partial_output":
        candidate["cases"]["short"]["token_ids"].pop()
    elif change == "collector_differs":
        candidate["collector_sha256"] = "d" * 64
    elif change == "workload_differs":
        candidate["workload"][0]["arrival_step"] += 1
    else:
        del candidate["max_concurrent_batches"]
    result = compare(verifier, pair)
    assert not result["passed"]
    assert result["structural_errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_ids", [3, 99, 5]),
        ("text", "different text"),
        ("stop_reason", 99),
    ],
)
def test_full_output_differences_are_separate_from_provenance(
    verifier, pair, field, value
):
    pair[1]["cases"]["short"][field] = value
    result = compare(verifier, pair)
    assert result["provenance_valid"]
    assert not result["passed"]
    assert {"request_id": "short", "field": field}.items() <= result[
        "output_mismatches"
    ][0].items()


@pytest.mark.parametrize("code", [None, -11, 1])
def test_process_failure_cannot_hide_behind_completed_json(verifier, pair, code):
    result = compare(verifier, pair, candidate_exit_code=code)
    assert not result["passed"]
    assert "exit code" in result["structural_errors"][0]


def test_git_verification_uses_pinned_blobs_and_exact_inventory(verifier, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "vllm_metal/model.py"
    source.parent.mkdir()
    source.write_text("original\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    expected = verifier.git_snapshot(tmp_path, head)
    source.write_text("different worktree\n")
    source.with_name("untracked.py").write_text("extra\n")
    assert verifier.git_snapshot(tmp_path, head) == expected
    assert set(expected) == {"vllm_metal/model.py"}


def test_verify_sources_rejects_recorded_content_not_at_pin(
    verifier, pair, monkeypatch
):
    monkeypatch.setattr(
        verifier, "git_snapshot", lambda *args: {"changed.py": "d" * 64}
    )
    result = compare(verifier, pair, verify_sources=True)
    assert not result["passed"]
    assert "Git blobs" in result["structural_errors"][0]


def test_cli_requires_exit_evidence_and_returns_nonzero_for_mismatch(
    verifier, pair, monkeypatch, tmp_path
):
    left, right = tmp_path / "main.json", tmp_path / "candidate.json"
    left.write_text(json.dumps(pair[0]))
    pair[1]["cases"]["short"]["text"] = "different"
    right.write_text(json.dumps(pair[1]))
    base = [
        "compare",
        "--main",
        str(left),
        "--candidate",
        str(right),
        "--main-head",
        "1" * 40,
        "--candidate-head",
        "2" * 40,
    ]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="2"):
        verifier.main()
    monkeypatch.setattr(
        sys, "argv", base + ["--main-exit-code", "0", "--candidate-exit-code", "0"]
    )
    with pytest.raises(SystemExit, match="2"):
        verifier.main()
