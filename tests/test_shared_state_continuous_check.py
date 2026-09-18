# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for evidence validity in the scheduler collector."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _seed_random_generators():
    # Override the repository fixture: these tests never need MLX or its GPU.
    pass


@pytest.fixture
def collector():
    path = Path(__file__).parents[1] / "tools/shared_state_continuous_check.py"
    spec = importlib.util.spec_from_file_location("shared_state_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete_result():
    cases = {
        "cancel_prefill": {
            "terminal": True,
            "cancelled": True,
            "token_ids": [],
            "prompt_tokens": 1024,
            "abort_snapshot": {"num_computed_tokens": 128, "num_in_flight_tokens": 64},
        },
        "cancel": {"terminal": True, "cancelled": True, "token_ids": [1, 2]},
        "long": {"terminal": True, "engine_finished": True},
        "repeat_long": {
            "terminal": True,
            "engine_finished": True,
            "repeats_prefix_of": "long",
            "arrival_step": 160,
            "predecessor_finished_at_arrival": True,
            "num_cached_tokens": 128,
        },
    }
    return {
        "scenario": "pressure",
        "max_num_seqs": 4,
        "async_scheduling": True,
        "decode_pipeline_submissions": 1,
        "cases": cases,
        "workload": [{"request_id": name} for name in cases],
        "scheduler_trace": [
            {
                "schedule_index": 0,
                "preempted": ["long"],
                "resumed": ["long"],
                "finished": ["cancel", "cancel_prefill"],
                "new": [{"request_id": "repeat_long", "num_computed_tokens": 128}],
            }
        ],
        "forward_trace": [
            {
                "mixed_prefill_decode": True,
                "surviving_request_slot_changes": {"long": {"before": 1, "after": 0}},
                "surviving_relative_order_changed": False,
            }
        ],
        "events": [{"kind": "arrival", "other_unfinished": True, "engine_step": 3}],
    }


def test_complete_coverage_uses_dynamic_workload(collector, complete_result):
    complete_result["workload"].append({"request_id": "another_terminal_request"})
    complete_result["cases"]["another_terminal_request"] = {"terminal": True}
    _, missing = collector.collect_coverage(complete_result)
    assert not missing


def test_same_count_cannot_hide_missing_workload_identity(collector, complete_result):
    complete_result["workload"][-1]["request_id"] = "never_submitted"
    _, missing = collector.collect_coverage(complete_result)
    assert "all_requests_terminal" in missing


def test_async_run_requires_actual_native_pipeline(collector, complete_result):
    complete_result["decode_pipeline_submissions"] = 0
    _, missing = collector.collect_coverage(complete_result)
    assert "native_decode_pipeline_observed" in missing


@pytest.mark.parametrize("asynchronous", [False, True])
def test_inflight_prefill_cancel_required_only_for_async(
    collector, complete_result, asynchronous
):
    complete_result["async_scheduling"] = asynchronous
    complete_result["cases"]["cancel_prefill"]["abort_snapshot"][
        "num_in_flight_tokens"
    ] = 0
    _, missing = collector.collect_coverage(complete_result)
    assert ("prefill_cancel_with_inflight_work" in missing) == asynchronous


@pytest.mark.parametrize(
    "omitted,expected",
    [
        ("prefill_abort", "prefill_cancel_observed"),
        ("cancel_cleanup", "cancel_cleanup_delivered"),
        ("prefix_hits", "prefix_cache_hit"),
        ("late_prefix_hit", "late_repeat_prefix_cache_hit"),
        ("preemption", "preemption_observed"),
        ("resume", "resume_observed"),
        ("recovery", "preempted_request_completed"),
    ],
)
def test_missing_observations_prevent_complete_coverage(
    collector, complete_result, omitted, expected
):
    if omitted == "prefill_abort":
        complete_result["cases"]["cancel_prefill"].pop("abort_snapshot")
    elif omitted == "cancel_cleanup":
        complete_result["scheduler_trace"][0]["finished"] = ["cancel"]
    elif omitted in ("prefix_hits", "late_prefix_hit"):
        complete_result["cases"]["repeat_long"]["num_cached_tokens"] = 0
        if omitted == "prefix_hits":
            complete_result["scheduler_trace"][0]["new"] = []
    elif omitted == "preemption":
        complete_result["scheduler_trace"][0]["preempted"] = []
    elif omitted == "resume":
        complete_result["scheduler_trace"][0]["resumed"] = []
    elif omitted == "recovery":
        complete_result["cases"]["long"]["engine_finished"] = False
    _, missing = collector.collect_coverage(complete_result)
    assert expected in missing


@pytest.fixture
def main_environment(collector, tmp_path, monkeypatch):
    runner = tmp_path / "vllm_metal/v1/model_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# Selected checkout\n")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--model",
            "unused",
            "--variant",
            "candidate",
            "--source-root",
            str(tmp_path),
            "--scenario",
            "pressure",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        collector.subprocess, "check_output", lambda *a, **k: "abc123\n"
    )
    monkeypatch.setattr(
        collector.faulthandler, "dump_traceback_later", lambda *a, **k: None
    )
    monkeypatch.setattr(
        collector.faulthandler, "cancel_dump_traceback_later", lambda: None
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VLLM_METAL_UNIFIED_CACHE", "0")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    return output, runner


def test_runtime_failure_is_saved_and_cannot_become_pass(
    collector, main_environment, monkeypatch
):
    output, _ = main_environment

    def fail(args, result):
        raise RuntimeError("test failure must survive collection")

    monkeypatch.setattr(collector, "run", fail)
    with pytest.raises(SystemExit) as exc:
        collector.main()
    saved = json.loads(output.read_text())
    assert exc.value.code == 2
    assert saved["status"] == "failed"
    assert not saved["evidence_valid"]
    assert "test failure must survive collection" in saved["error"]


@pytest.mark.parametrize("change", ["edit", "add", "remove"])
def test_source_changes_invalidate_completed_run(
    collector, main_environment, monkeypatch, change
):
    output, runner = main_environment

    def mutate(args, result):
        result["status"] = "completed"
        if change == "edit":
            runner.write_text("# Changed implementation\n")
        elif change == "add":
            runner.with_name("new_module.py").write_text("# Added implementation\n")
        else:
            runner.unlink()

    monkeypatch.setattr(collector, "run", mutate)
    with pytest.raises(SystemExit) as exc:
        collector.main()
    saved = json.loads(output.read_text())
    assert exc.value.code == 2
    assert saved["status"] == "invalid_source_changed"
    assert not saved["evidence_valid"]
    assert saved["source_changed_during_run"]


def test_incomplete_coverage_cannot_exit_success(
    collector, complete_result, main_environment, monkeypatch
):
    output, _ = main_environment

    def incomplete(args, result):
        result.update(copy.deepcopy(complete_result))
        result["cases"]["cancel_prefill"].pop("abort_snapshot")
        coverage, missing = collector.collect_coverage(result)
        result.update(
            coverage=coverage,
            missing_required_coverage=missing,
            status="coverage_incomplete" if missing else "completed",
        )

    monkeypatch.setattr(collector, "run", incomplete)
    with pytest.raises(SystemExit) as exc:
        collector.main()
    saved = json.loads(output.read_text())
    assert exc.value.code == 2
    assert saved["status"] == "coverage_incomplete"
    assert "prefill_cancel_observed" in saved["missing_required_coverage"]


@pytest.mark.parametrize("change", ["edit", "add", "remove"])
def test_upstream_changes_invalidate_completed_run(
    collector, main_environment, monkeypatch, tmp_path, change
):
    output, _ = main_environment
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    init = upstream / "__init__.py"
    init.write_text("# runtime version\n")

    def mutate(args, result):
        result.update(status="completed", vllm_package=str(init))
        result["upstream_source_sha256"] = collector.upstream_snapshot(upstream)
        if change == "edit":
            init.write_text("# changed runtime\n")
        elif change == "add":
            (upstream / "new_module.py").write_text("# new runtime source\n")
        else:
            init.unlink()

    monkeypatch.setattr(collector, "run", mutate)
    with pytest.raises(SystemExit):
        collector.main()
    saved = json.loads(output.read_text())
    assert saved["status"] == "invalid_source_changed"
    assert not saved["evidence_valid"]
    assert "upstream runtime sources" in saved["source_changed_during_run"]
