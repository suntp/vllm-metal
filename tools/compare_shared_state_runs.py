# SPDX-License-Identifier: Apache-2.0
"""Fail-closed comparison of two controlled shared-state collector reports.

Exit status 0 means both provenance and complete outputs match; 2 means either
check failed. Process exit codes and full pinned commit IDs are mandatory.
--verify-sources additionally compares the recorded implementation inventory to
Git blobs at each pinned commit, independently of the present working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

CASE_IDS = {
    "cancel_prefill",
    "edge_minus",
    "edge_plus",
    "long",
    "cancel",
    "short",
    "long_second",
    "edge_exact",
    "shared_suffix",
    "edge_late",
    "repeat_edge",
    "repeat_long",
    "repeat_long_again",
}
CONFIG_FIELDS = (
    "schema_version",
    "scope",
    "model",
    "scenario",
    "max_num_seqs",
    "sampling",
    "requested_blocks",
    "block_size",
    "max_model_len",
    "memory_fraction",
    "max_steps",
    "timeout_seconds",
    "async_scheduling",
    "max_concurrent_batches",
    "python_executable",
    "versions",
    "vllm_version",
    "collector_sha256",
    "workload_sha256",
    "workload",
    "upstream_source_sha256",
)
OUTPUT_FIELDS = (
    "token_ids",
    "text",
    "finish_reason",
    "stop_reason",
    "cancelled",
    "terminal",
    "engine_finished",
)
SPEC_FIELDS = (
    "request_id",
    "arrival_step",
    "prompt_token_ids",
    "prompt_tokens",
    "max_output_tokens",
    "repeats_prefix_of",
    "cancel_after_output_tokens",
    "cancel_during_prefill",
)


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_report(path):
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_fields)


def valid_snapshot(snapshot):
    require(isinstance(snapshot, dict) and snapshot, "missing source inventory")
    for name, digest in snapshot.items():
        require(
            isinstance(name, str)
            and name
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts,
            "invalid source inventory path",
        )
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"invalid source fingerprint: {name}",
        )


def git_snapshot(root, head):
    listing = subprocess.check_output(
        ["git", "ls-tree", "-rz", head, "--", "vllm_metal"], cwd=root
    )
    entries = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        metadata, filename = record.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        name = filename.decode()
        if Path(name).suffix in {".py", ".cpp", ".metal"}:
            require(
                kind == b"blob" and mode in (b"100644", b"100755"),
                f"unsupported source object: {name}",
            )
            entries.append((name, oid))
    require(entries, "pinned Git tree contains no implementation sources")
    data = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=b"\n".join(oid for _, oid in entries) + b"\n",
    )
    result, offset = {}, 0
    for name, oid in entries:
        end = data.index(b"\n", offset)
        actual, kind, size = data[offset:end].split()
        require(actual == oid and kind == b"blob", "invalid Git blob response")
        offset = end + 1
        size = int(size)
        result[name] = hashlib.sha256(data[offset : offset + size]).hexdigest()
        offset += size + 1
    return result


def observed_coverage(report):
    """Derive required observations independently of saved coverage booleans."""
    cases = report["cases"]
    schedule, forwards, events = (
        report["scheduler_trace"],
        report["forward_trace"],
        report["events"],
    )
    require(
        schedule and forwards and events, "missing scheduler/forward/event evidence"
    )
    preempted, resumed, finished = set(), set(), set()
    admitted = []
    for row in schedule:
        for field, target in (
            ("preempted", preempted),
            ("resumed", resumed),
            ("finished", finished),
        ):
            require(set(row[field]) <= CASE_IDS, f"unknown scheduler {field} request")
            target.update(row[field])
        for req in row["new"]:
            require(req["request_id"] in CASE_IDS, "unknown admitted request")
            admitted.append(req)
    for row in forwards:
        require(
            set(row["packed_request_order"]) <= CASE_IDS,
            "unknown packed-forward request",
        )
    for event in events:
        require(event["request_id"] in CASE_IDS, "unknown event request")
    cancelled = {name for name, case in cases.items() if case["cancelled"]}
    prefill = cases["cancel_prefill"]
    snapshot = prefill["abort_snapshot"]
    late = [
        case
        for case in cases.values()
        if case["repeats_prefix_of"] and case["arrival_step"] >= 96
    ]
    actual = {
        "all_requests_terminal": all(
            case["terminal"] is True for case in cases.values()
        ),
        "prefill_cancel_observed": prefill["cancelled"] is True
        and not prefill["token_ids"]
        and 0 < snapshot["num_computed_tokens"] < prefill["prompt_tokens"]
        and any(
            req["request_id"] == "cancel_prefill" and not req["final_chunk"]
            for row in forwards
            for req in row["prefill"]
        )
        and any(event["kind"] == "abort_prefill" for event in events),
        "cancel_cleanup_delivered": cancelled == {"cancel", "cancel_prefill"}
        and cancelled <= finished,
        "cancelled_after_exactly_two_output_tokens": cases["cancel"]["cancelled"]
        is True
        and len(cases["cancel"]["token_ids"]) == 2,
        "arrivals_during_unfinished_work": any(
            row["kind"] == "arrival"
            and row["other_unfinished"] is True
            and row["engine_step"] > 0
            for row in events
        ),
        "prefix_cache_hit": any(req["num_computed_tokens"] > 0 for req in admitted)
        or any(case["num_cached_tokens"] > 0 for case in cases.values()),
        "late_repeat_after_predecessor_completed": bool(late)
        and all(case["predecessor_finished_at_arrival"] is True for case in late),
        "late_repeat_prefix_cache_hit": any(
            case["num_cached_tokens"] > 0 for case in late
        ),
    }
    if report["max_num_seqs"] != 1:
        actual.update(
            mixed_prefill_decode=any(
                row["mixed_prefill_decode"] is True for row in forwards
            ),
            batch_slot_reassignment=any(
                row["surviving_request_slot_changes"] for row in forwards
            ),
        )
    if report["async_scheduling"]:
        actual.update(
            prefill_cancel_with_inflight_work=snapshot["num_in_flight_tokens"] > 0,
            native_decode_pipeline_observed=report["decode_pipeline_submissions"] > 0,
        )
    if report["scenario"] == "pressure":
        actual.update(
            preemption_observed=bool(preempted),
            resume_observed=bool(preempted & resumed),
            preempted_request_completed=any(
                cases[name]["engine_finished"] is True for name in preempted & resumed
            ),
        )
    return actual


def validate_report(report, role, expected_head, exit_code, verify_sources=False):
    require(
        type(exit_code) is int and exit_code == 0, "process exit code missing/nonzero"
    )
    require(isinstance(report, dict), "report is not an object")
    require(
        re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected_head),
        "expected HEAD must be a full pinned commit ID",
    )
    require(report["checkout_head"] == expected_head, "checkout HEAD differs from pin")
    require(report["variant"] == role, "wrong variant identity")
    require(report["schema_version"] == 1, "unsupported report schema")
    require(report["status"] == "completed", "run did not complete")
    require(report["evidence_valid"] is True, "run evidence is invalid")
    require(
        report["final_gpu_synchronized"] is True, "final GPU synchronization absent"
    )
    require(report["source_changed_during_run"] == [], "sources changed during run")
    require(report["missing_required_coverage"] == [], "coverage incomplete")
    require(
        not report.get("error") and not report.get("traceback"),
        "report contains failure",
    )
    require(report["scenario"] in ("mixed", "pressure"), "unknown scenario")
    require(type(report["async_scheduling"]) is bool, "scheduling mode missing")
    for field in CONFIG_FIELDS:
        require(field in report, f"missing configuration: {field}")
    require(
        type(report["scheduler_blocks"]) is int and report["scheduler_blocks"] > 0,
        "missing actual scheduler capacity",
    )
    require(
        type(report["requested_blocks"]) is int and report["requested_blocks"] >= 0,
        "invalid requested block capacity",
    )
    if report["requested_blocks"]:
        require(
            report["scheduler_blocks"] == report["requested_blocks"],
            "actual scheduler capacity differs from explicit override",
        )
    require(
        report["sampling"] == {"temperature": 0, "logprobs": None, "ignore_eos": True},
        "unexpected sampling configuration",
    )
    require(
        set(report["versions"])
        >= {"mlx", "mlx-lm", "mlx-vlm", "numpy", "vllm", "torch"},
        "missing dependency versions",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", report["collector_sha256"]),
        "invalid collector hash",
    )
    for before, after in (
        ("source_sha256_before_imports", "source_sha256_after_run"),
        ("upstream_source_sha256", "upstream_source_sha256_after_run"),
    ):
        valid_snapshot(report[before])
        require(report[before] == report[after], f"unstable source inventory: {before}")
    require(
        {
            "v1/kv_cache_interface.py",
            "v1/worker/utils.py",
            "model_executor/layers/mamba/abstract.py",
            "v1/core/sched/scheduler.py",
        }
        <= set(report["upstream_source_sha256"]),
        "upstream inventory omits critical sources",
    )
    if verify_sources:
        require(
            report["source_sha256_before_imports"]
            == git_snapshot(report["source_root"], expected_head),
            "implementation inventory differs from pinned Git blobs",
        )
    require(
        bool(report["storage"]) == (role == "candidate")
        and bool(report["legacy_storage"]) == (role == "main"),
        "initialization evidence does not match variant",
    )
    workload = report["workload"]
    require(
        isinstance(workload, list) and len(workload) == len(CASE_IDS),
        "workload must contain all 13 cases",
    )
    require(
        {row["request_id"] for row in workload} == CASE_IDS,
        "unknown, duplicate, or missing workload case",
    )
    require(fingerprint(workload) == report["workload_sha256"], "workload hash differs")
    require(set(report["cases"]) == CASE_IDS, "unknown or missing output case")
    for spec in workload:
        name, case = spec["request_id"], report["cases"][spec["request_id"]]
        for field in SPEC_FIELDS:
            require(
                case[field] == spec[field],
                f"{name}: workload identity differs at {field}",
            )
        require(
            len(spec["prompt_token_ids"]) == spec["prompt_tokens"],
            f"{name}: prompt length differs",
        )
        require(
            isinstance(case["token_ids"], list)
            and all(type(token) is int and token >= 0 for token in case["token_ids"]),
            f"{name}: invalid token IDs",
        )
        require(isinstance(case["text"], str), f"{name}: missing output text")
        require(case["terminal"] is True, f"{name}: not terminal")
        cancelled = name in {"cancel", "cancel_prefill"}
        require(
            case["cancelled"] is cancelled and case["engine_finished"] is not cancelled,
            f"{name}: invalid cancellation/completion identity",
        )
        expected_length = (
            (0 if name == "cancel_prefill" else 2)
            if cancelled
            else spec["max_output_tokens"]
        )
        require(len(case["token_ids"]) == expected_length, f"{name}: incomplete output")
        require(
            case["finish_reason"] == ("abort" if cancelled else "length"),
            f"{name}: unexpected finish reason",
        )
        require("stop_reason" in case, f"{name}: missing stop reason")
    observed = observed_coverage(report)
    require(
        set(report["coverage"]["required_flags"]) == set(observed),
        "required coverage flags missing or unknown",
    )
    for flag, actual in observed.items():
        require(
            actual and report["coverage"]["flags"][flag] is True,
            f"required coverage unsupported: {flag}",
        )


def compare_reports(
    main,
    candidate,
    *,
    main_head,
    candidate_head,
    main_exit_code,
    candidate_exit_code,
    verify_sources=False,
):
    structural, mismatches = [], []
    for role, report, head, code in (
        ("main", main, main_head, main_exit_code),
        ("candidate", candidate, candidate_head, candidate_exit_code),
    ):
        try:
            validate_report(report, role, head, code, verify_sources)
        except (
            EvidenceError,
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            structural.append(f"{role}: {type(exc).__name__}: {exc}")
    if isinstance(main, dict) and isinstance(candidate, dict):
        for field in CONFIG_FIELDS:
            if field in main and field in candidate and main[field] != candidate[field]:
                structural.append(f"pair configuration differs: {field}")
        main_cases = main.get("cases") if isinstance(main.get("cases"), dict) else {}
        candidate_cases = (
            candidate.get("cases") if isinstance(candidate.get("cases"), dict) else {}
        )
        for name in sorted(CASE_IDS):
            left, right = (
                main_cases.get(name),
                candidate_cases.get(name),
            )
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue  # Already a structural error; never an equality claim.
            for field in OUTPUT_FIELDS:
                if field in left and field in right and left[field] != right[field]:
                    difference = {"request_id": name, "field": field}
                    if (
                        field == "token_ids"
                        and isinstance(left[field], list)
                        and isinstance(right[field], list)
                    ):
                        difference["main_length"] = len(left[field])
                        difference["candidate_length"] = len(right[field])
                        difference["first_difference"] = next(
                            (
                                i
                                for i, (a, b) in enumerate(
                                    zip(left[field], right[field], strict=False)
                                )
                                if a != b
                            ),
                            min(len(left[field]), len(right[field])),
                        )
                    mismatches.append(difference)
    return {
        "passed": not structural and not mismatches,
        "provenance_valid": not structural,
        "outputs_equal": not structural and not mismatches,
        "source_content_verified": verify_sources and not structural,
        "structural_errors": structural,
        "output_mismatches": mismatches,
        "expected_case_count": len(CASE_IDS),
        "main_head": main_head,
        "candidate_head": candidate_head,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--main-head", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--main-exit-code", type=int, required=True)
    parser.add_argument("--candidate-exit-code", type=int, required=True)
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare_reports(
            load_report(args.main),
            load_report(args.candidate),
            main_head=args.main_head,
            candidate_head=args.candidate_head,
            main_exit_code=args.main_exit_code,
            candidate_exit_code=args.candidate_exit_code,
            verify_sources=args.verify_sources,
        )
    except (OSError, ValueError) as exc:
        result = {
            "passed": False,
            "provenance_valid": False,
            "outputs_equal": False,
            "structural_errors": [f"unreadable report: {exc}"],
            "output_mismatches": [],
        }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
