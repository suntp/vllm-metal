# SPDX-License-Identifier: Apache-2.0
"""Controlled continuous-arrival scheduler regression, not a throughput benchmark.

Run each storage mode in a fresh process. Arrivals use fixed LLMEngine.step()
indices, including idle steps. Inspect coverage flags before using any result:
completion alone does not establish pressure preemption, reuse, or batch reorder.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".cpp", ".metal"}


def source_snapshot() -> dict[str, str]:
    """Hash the explicitly selected implementation independently of collector."""
    paths = [
        path
        for path in (ROOT / "vllm_metal").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def save_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def upstream_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def make_workload(tokenizer, block_size: int) -> list[dict]:
    """Different early prefixes create pressure; explicit repeats test reuse."""
    texts = {
        "a": "Attention pages retain keys and values across generation steps. ",
        "b": "Recurrent checkpoints preserve a running summary of previous tokens. ",
        "c": "A scheduler assigns free memory blocks to independent requests. ",
        "d": "Cancellation releases the resources held by an unfinished request. ",
        "e": "Preemption temporarily stops a request and recomputes its context. ",
        "f": "A copied page must retain the original checkpoint without changes. ",
        "g": "Batch positions change when requests arrive and others finish. ",
    }
    corpora = {key: tokenizer(text * 600)["input_ids"] for key, text in texts.items()}
    # Four requests start together. Subsequent admission opportunities overlap
    # their different output lengths; late repeats follow a quiet interval.
    plan = [
        ("cancel_prefill", 0, "g", 4 * block_size + 97, 48, None),
        ("edge_minus", 0, "a", block_size - 1, 48, None),
        ("edge_plus", 0, "b", block_size + 1, 24, None),
        ("long", 0, "c", 2 * block_size + 97, 48, None),
        ("cancel", 0, "d", block_size + 1, 48, None),
        ("short", 3, "e", max(16, block_size // 3), 8, None),
        ("long_second", 9, "e", 4 * block_size + 97, 24, None),
        ("edge_exact", 20, "f", block_size, 8, None),
        ("shared_suffix", 20, "b", block_size + 37, 24, "edge_plus"),
        ("edge_late", 48, "g", block_size - 1, 8, None),
        ("repeat_edge", 96, "b", block_size + 1, 8, "edge_plus"),
        ("repeat_long", 160, "c", 2 * block_size + 97, 8, "long"),
        ("repeat_long_again", 224, "c", 2 * block_size + 97, 24, "repeat_long"),
    ]
    result = []
    for name, arrival, corpus_key, length, output_length, repeats in plan:
        ids = list(corpora[corpus_key][:length])
        if len(ids) != length or length + output_length > 4096:
            raise ValueError(f"workload {name} does not fit the selected block size")
        result.append(
            {
                "request_id": name,
                "arrival_step": arrival,
                "prompt_token_ids": ids,
                "prompt_tokens": length,
                "max_output_tokens": output_length,
                "repeats_prefix_of": repeats,
                "cancel_after_output_tokens": 2 if name == "cancel" else None,
                "cancel_during_prefill": name == "cancel_prefill",
            }
        )
    return result


class Trace:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.engine_step = -1
        self.internal_to_external: dict[str, str] = {}
        self.scheduler_outputs: dict[int, int] = {}
        self.request_preemptions: dict[str, int] = {}
        self.previous_order: list[str] = []
        self.patches = []
        self.current_scheduler = None

    def name(self, request_id: str) -> str:
        return self.internal_to_external.get(request_id, request_id)

    def patch(self, owner, name, replacement) -> None:
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def restore(self) -> None:
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)

    def scheduler(self, original, scheduler, *args, **kwargs):
        self.current_scheduler = scheduler
        output = original(scheduler, *args, **kwargs)
        index = len(self.result["scheduler_trace"])
        self.scheduler_outputs[id(output)] = index
        cached = output.scheduled_cached_reqs
        new_rows = [
            {
                "request_id": self.name(req.req_id),
                "num_computed_tokens": int(req.num_computed_tokens),
                "prompt_tokens": int(req.prompt_len),
            }
            for req in output.scheduled_new_reqs
        ]
        cached_rows = [
            {
                "request_id": self.name(req_id),
                "num_computed_tokens": int(computed),
                "num_output_tokens_including_placeholders": int(num_outputs),
                "resumed": req_id in cached.resumed_req_ids,
            }
            for req_id, computed, num_outputs in zip(
                cached.req_ids,
                cached.num_computed_tokens,
                cached.num_output_tokens,
                strict=True,
            )
        ]
        for req_id, request in scheduler.requests.items():
            count = getattr(request, "num_preemptions", None)
            if count is not None:
                name = self.name(req_id)
                self.request_preemptions[name] = max(
                    self.request_preemptions.get(name, 0), int(count)
                )
        available_counters = {
            name: int(value)
            for name in ("num_preemptions", "num_cumulative_preemption")
            if isinstance(value := getattr(scheduler, name, None), int)
        }
        row = {
            "schedule_index": index,
            "engine_step": self.engine_step,
            "scheduler_class": type(scheduler).__name__,
            "scheduled_order": [self.name(req) for req in output.num_scheduled_tokens],
            "num_scheduled_tokens": {
                self.name(req): int(count)
                for req, count in output.num_scheduled_tokens.items()
            },
            "total_num_scheduled_tokens": int(output.total_num_scheduled_tokens),
            "new": new_rows,
            "cached": cached_rows,
            "preempted": sorted(
                self.name(req) for req in output.preempted_req_ids or ()
            ),
            "resumed": sorted(self.name(req) for req in cached.resumed_req_ids),
            "finished": sorted(self.name(req) for req in output.finished_req_ids),
            "request_preemption_counts": dict(self.request_preemptions),
            "cumulative_request_preemptions": sum(self.request_preemptions.values()),
            "scheduler_preemption_counters": available_counters,
            "num_common_prefix_blocks": list(output.num_common_prefix_blocks),
        }
        self.result["scheduler_trace"].append(row)
        return output

    def forward(self, original, runner, batch, prefill, decode, scheduler_output):
        decode_ids = [self.name(req_id) for req_id, _ in decode]
        prefill_ids = [self.name(req.req_id) for req in prefill]
        order = decode_ids + prefill_ids
        common = set(order) & set(self.previous_order)
        previous_survivors = [name for name in self.previous_order if name in common]
        current_survivors = [name for name in order if name in common]
        moved = {
            name: {
                "before": self.previous_order.index(name),
                "after": order.index(name),
            }
            for name in sorted(common)
            if self.previous_order.index(name) != order.index(name)
        }
        self.result["forward_trace"].append(
            {
                "engine_step": self.engine_step,
                "schedule_index": self.scheduler_outputs.get(id(scheduler_output)),
                "packed_request_order": order,
                "decode": decode_ids,
                "prefill": [
                    {
                        "request_id": self.name(req.req_id),
                        "start_pos": int(req.start_pos),
                        "tokens": len(req.token_ids),
                        "final_chunk": req.prompt_len is not None,
                    }
                    for req in prefill
                ],
                "mixed_prefill_decode": bool(decode_ids and prefill_ids),
                "surviving_request_slot_changes": moved,
                "surviving_relative_order_changed": previous_survivors
                != current_survivors,
            }
        )
        self.previous_order = order
        return original(runner, batch, prefill, decode, scheduler_output)


def collect_coverage(result: dict) -> tuple[dict, list[str]]:
    cases = result["cases"]
    scheduler = result["scheduler_trace"]
    forwards = result["forward_trace"]
    preempted = {name for row in scheduler for name in row["preempted"]}
    resumed = {name for row in scheduler for name in row["resumed"]}
    recovered = sorted(
        name
        for name in preempted & resumed
        if cases.get(name, {}).get("engine_finished", False)
    )
    prefix_admissions = [
        {"schedule_index": row["schedule_index"], **req}
        for row in scheduler
        for req in row["new"]
        if req["num_computed_tokens"] > 0
    ]
    public_hits = {
        name: case["num_cached_tokens"]
        for name, case in cases.items()
        if case.get("num_cached_tokens", 0) > 0
    }
    late_repeats = [
        case
        for case in cases.values()
        if case.get("repeats_prefix_of") and case["arrival_step"] >= 96
    ]
    cancelled = {name for name, case in cases.items() if case.get("cancelled")}
    finished = {name for row in scheduler for name in row["finished"]}
    prefill_abort = cases.get("cancel_prefill", {}).get("abort_snapshot", {})
    flags = {
        "all_requests_terminal": set(cases)
        == {row["request_id"] for row in result.get("workload", [])}
        and bool(cases)
        and all(case.get("terminal", False) for case in cases.values()),
        "prefill_cancel_observed": cases.get("cancel_prefill", {}).get(
            "cancelled", False
        )
        and not cases.get("cancel_prefill", {}).get("token_ids")
        and 0
        < prefill_abort.get("num_computed_tokens", 0)
        < cases.get("cancel_prefill", {}).get("prompt_tokens", 0),
        "cancel_cleanup_delivered": cancelled == {"cancel", "cancel_prefill"}
        and cancelled <= finished,
        "prefill_cancel_with_inflight_work": prefill_abort.get(
            "num_in_flight_tokens", 0
        )
        > 0,
        "native_decode_pipeline_observed": result.get("decode_pipeline_submissions", 0)
        > 0,
        "cancelled_after_exactly_two_output_tokens": cases.get("cancel", {}).get(
            "cancelled", False
        )
        and len(cases.get("cancel", {}).get("token_ids", [])) == 2,
        "arrivals_during_unfinished_work": any(
            event["kind"] == "arrival"
            and event["other_unfinished"]
            and event["engine_step"] > 0
            for event in result["events"]
        ),
        "mixed_prefill_decode": any(row["mixed_prefill_decode"] for row in forwards),
        "batch_slot_reassignment": any(
            row["surviving_request_slot_changes"] for row in forwards
        ),
        "surviving_relative_order_reversal": any(
            row["surviving_relative_order_changed"] for row in forwards
        ),
        "preemption_observed": bool(preempted),
        "resume_observed": bool(preempted & resumed),
        "preempted_request_completed": bool(recovered),
        "prefix_cache_hit": bool(prefix_admissions or public_hits),
        "late_repeat_after_predecessor_completed": bool(late_repeats)
        and all(
            case.get("predecessor_finished_at_arrival", False) for case in late_repeats
        ),
        "late_repeat_prefix_cache_hit": any(
            case.get("num_cached_tokens", 0) > 0 for case in late_repeats
        ),
    }
    # A relative inversion is stronger than a moved batch slot; preserve both
    # observations instead of claiming either from membership changes alone.
    required = [
        "all_requests_terminal",
        "prefill_cancel_observed",
        "cancel_cleanup_delivered",
        "cancelled_after_exactly_two_output_tokens",
        "arrivals_during_unfinished_work",
        "mixed_prefill_decode",
        "batch_slot_reassignment",
        "prefix_cache_hit",
        "late_repeat_after_predecessor_completed",
        "late_repeat_prefix_cache_hit",
    ]
    if result["scenario"] == "pressure":
        required += [
            "preemption_observed",
            "resume_observed",
            "preempted_request_completed",
        ]
    if result["max_num_seqs"] == 1:
        required.remove("mixed_prefill_decode")
        required.remove("batch_slot_reassignment")
    if result.get("async_scheduling"):
        required.append("prefill_cancel_with_inflight_work")
        required.append("native_decode_pipeline_observed")
    return {
        "flags": flags,
        "required_flags": required,
        "preempted_requests": sorted(preempted),
        "resumed_requests": sorted(resumed),
        "recovered_requests": recovered,
        "prefix_admissions": prefix_admissions,
        "public_num_cached_tokens": public_hits,
    }, [name for name in required if not flags[name]]


def run(args, result: dict) -> None:
    # The caller has already captured all local sources before these imports.
    import mlx.core as mx
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.core.sched.scheduler import Scheduler

    import vllm_metal.v1.model_runner as runner_module
    from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
    from vllm_metal.v1.decode_pipeline import DecodePipeline

    if Path(runner_module.__file__).resolve() != ROOT / "vllm_metal/v1/model_runner.py":
        raise RuntimeError("model runner was imported from a different checkout")
    result["vllm_version"] = importlib.metadata.version("vllm")
    result["vllm_package"] = str(Path(vllm.__file__).resolve())
    upstream = Path(vllm.__file__).resolve().parent
    result["upstream_source_sha256"] = upstream_snapshot(upstream)
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "vllm", "torch")
    }
    result["scheduler_source"] = str(
        Path(sys.modules[Scheduler.__module__].__file__).resolve()
    )
    trace = Trace(result)
    original_schedule = Scheduler.schedule
    original_forward = runner_module.MetalModelRunner._start_paged_forward
    original_submit = DecodePipeline.submit
    original_shared = getattr(
        HybridPagedAttentionRuntime, "initialize_from_config", None
    )
    original_legacy = getattr(HybridPagedAttentionRuntime, "initialize", None)
    if args.variant == "candidate" and original_shared is None:
        raise RuntimeError(
            "this checkout does not implement unified cache initialization"
        )

    def schedule(self, *positional, **keywords):
        return trace.scheduler(original_schedule, self, *positional, **keywords)

    def forward(self, batch, prefill_reqs, decode_reqs, scheduler_output):
        return trace.forward(
            original_forward, self, batch, prefill_reqs, decode_reqs, scheduler_output
        )

    def submit(self, pending_step):
        result["decode_pipeline_submissions"] += 1
        return original_submit(self, pending_step)

    def shared(self, config, **kwargs):
        assert original_shared is not None
        active_before = mx.get_active_memory()
        original_shared(self, config, **kwargs)
        owner = self.storage
        assert self.kv_cache.key_caches.storage is owner
        assert self.state_cache.conv_states.storage is owner
        result["storage"].append(
            {
                "bytes": owner.nbytes,
                "blocks": config.num_blocks,
                "active_before_allocation": active_before,
                "active_after_allocation": mx.get_active_memory(),
            }
        )

    def legacy(self, num_blocks):
        active_before = mx.get_active_memory()
        original_legacy(self, num_blocks)
        result["legacy_storage"].append(
            {
                "blocks": num_blocks,
                "active_before_allocation": active_before,
                "active_after_allocation": mx.get_active_memory(),
            }
        )

    trace.patch(Scheduler, "schedule", schedule)
    trace.patch(runner_module.MetalModelRunner, "_start_paged_forward", forward)
    trace.patch(DecodePipeline, "submit", submit)
    if args.variant == "candidate":
        trace.patch(HybridPagedAttentionRuntime, "initialize_from_config", shared)
    else:
        if original_legacy is None:
            raise RuntimeError("main role requires the legacy hybrid initializer")
        trace.patch(HybridPagedAttentionRuntime, "initialize", legacy)
    llm = None
    try:
        start = time.perf_counter()
        llm = LLM(
            model=args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=args.memory_fraction,
            num_gpu_blocks_override=args.blocks or None,
            enable_prefix_caching=True,
            async_scheduling=not args.sync,
        )
        engine = llm.llm_engine
        config = engine.vllm_config
        assert config.cache_config.mamba_cache_mode == "align"
        assert bool(result["storage"]) == (args.variant == "candidate")
        assert bool(result["legacy_storage"]) == (args.variant == "main")
        block = config.cache_config.block_size
        result.update(
            scheduler_blocks=config.cache_config.num_gpu_blocks,
            block_size=block,
            async_scheduling=config.scheduler_config.async_scheduling,
            max_concurrent_batches=config.max_concurrent_batches,
            active_after_startup=mx.get_active_memory(),
            startup_seconds=time.perf_counter() - start,
            rss_high_water_after_startup=resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        )
        workload = make_workload(llm.get_tokenizer(), block)
        result["workload"] = workload
        result["workload_sha256"] = hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()
        ).hexdigest()
        arrivals = {}
        for spec in workload:
            arrivals.setdefault(spec["arrival_step"], []).append(spec)
        mx.reset_peak_memory()
        generation_start = time.perf_counter()
        save_result(args.output, result)
        for step in range(args.max_steps):
            trace.engine_step = step
            for spec in arrivals.get(step, ()):
                name = spec["request_id"]
                predecessor = spec["repeats_prefix_of"]
                case = {
                    **spec,
                    "token_ids": [],
                    "text": "",
                    "terminal": False,
                    "engine_finished": False,
                    "cancelled": False,
                    "num_cached_tokens": 0,
                    "predecessor_finished_at_arrival": bool(
                        predecessor
                        and result["cases"].get(predecessor, {}).get("engine_finished")
                    ),
                }
                result["cases"][name] = case
                result["events"].append(
                    {
                        "kind": "arrival",
                        "engine_step": step,
                        "request_id": name,
                        "other_unfinished": bool(engine.has_unfinished_requests()),
                    }
                )
                internal_id = engine.add_request(
                    name,
                    {"prompt_token_ids": spec["prompt_token_ids"]},
                    SamplingParams(
                        temperature=0,
                        max_tokens=spec["max_output_tokens"],
                        ignore_eos=True,
                        seed=0,
                        output_kind=RequestOutputKind.CUMULATIVE,
                    ),
                )
                trace.internal_to_external[internal_id] = name
                case["internal_request_id"] = internal_id
            outputs = engine.step()
            for output in outputs:
                name = trace.name(output.request_id)
                case = result["cases"][name]
                if case["terminal"]:
                    raise AssertionError(f"received output for terminal request {name}")
                if len(output.outputs) != 1:
                    raise AssertionError(
                        "collector requires one completion per request"
                    )
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                old_tokens = case["token_ids"]
                if tokens[: len(old_tokens)] != old_tokens:
                    raise AssertionError(
                        f"cumulative output changed its prefix: {name}"
                    )
                case.update(token_ids=tokens, text=completion.text)
                if tokens and "first_output_step" not in case:
                    case["first_output_step"] = step
                case["num_cached_tokens"] = max(
                    case["num_cached_tokens"], int(output.num_cached_tokens or 0)
                )
                result["events"].append(
                    {
                        "kind": "output",
                        "engine_step": step,
                        "request_id": name,
                        "new_token_ids": tokens[len(old_tokens) :],
                        "output_tokens": len(tokens),
                        "finished": bool(output.finished),
                    }
                )
                cancel_after = case["cancel_after_output_tokens"]
                if cancel_after is not None and len(tokens) >= cancel_after:
                    if len(tokens) != cancel_after or output.finished:
                        raise AssertionError(
                            "cancel request did not stop at exactly two observed tokens"
                        )
                    engine.abort_request([name])
                    case.update(
                        terminal=True,
                        cancelled=True,
                        finish_step=step,
                        finish_reason="abort",
                        stop_reason=None,
                    )
                    result["events"].append(
                        {
                            "kind": "abort",
                            "engine_step": step,
                            "request_id": name,
                            "observed_output_tokens": len(tokens),
                        }
                    )
                elif output.finished:
                    if len(tokens) != case["max_output_tokens"]:
                        raise AssertionError(f"unexpected output length for {name}")
                    case.update(
                        terminal=True,
                        engine_finished=True,
                        finish_step=step,
                        finish_reason=completion.finish_reason,
                        stop_reason=completion.stop_reason,
                    )
            # Cancel only after a real partial prefill forward was dispatched,
            # while the request still has uncomputed prompt tokens. Record
            # actual scheduler counters rather than inferring in-flight work.
            pending_prefill = result["cases"].get("cancel_prefill")
            if pending_prefill is not None and not pending_prefill["terminal"]:
                request = trace.current_scheduler.requests.get(
                    pending_prefill["internal_request_id"]
                )
                partial = any(
                    row["request_id"] == "cancel_prefill" and not row["final_chunk"]
                    for forward_row in result["forward_trace"]
                    for row in forward_row["prefill"]
                )
                if (
                    request is not None
                    and partial
                    and not pending_prefill["token_ids"]
                    and (
                        0
                        < request.num_computed_tokens
                        < pending_prefill["prompt_tokens"]
                    )
                ):
                    snapshot = {
                        "num_computed_tokens": int(request.num_computed_tokens),
                        "num_in_flight_tokens": int(request.num_in_flight_tokens),
                        "num_output_placeholders": int(request.num_output_placeholders),
                    }
                    engine.abort_request(["cancel_prefill"])
                    pending_prefill.update(
                        terminal=True,
                        cancelled=True,
                        finish_step=step,
                        finish_reason="abort",
                        stop_reason=None,
                        abort_snapshot=snapshot,
                    )
                    result["events"].append(
                        {
                            "kind": "abort_prefill",
                            "engine_step": step,
                            "request_id": "cancel_prefill",
                            **snapshot,
                        }
                    )
            result["engine_steps"] = step + 1
            if step % 16 == 0:
                save_result(args.output, result)
            if step >= max(arrivals) and not engine.has_unfinished_requests():
                if not all(case["terminal"] for case in result["cases"].values()):
                    raise AssertionError(
                        "engine became idle before every request terminated"
                    )
                break
        else:
            raise TimeoutError(
                f"continuous workload exceeded {args.max_steps} engine steps"
            )
        mx.synchronize()
        result.update(
            final_gpu_synchronized=True,
            active_after_generation=mx.get_active_memory(),
            peak_active_bytes=mx.get_peak_memory(),
            allocator_cache_bytes=mx.get_cache_memory(),
            generation_seconds=time.perf_counter() - generation_start,
            rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        if not result["scheduler_trace"] or not result["forward_trace"]:
            raise AssertionError(
                "scheduler or actual packed-forward instrumentation did not run"
            )
        result["coverage"], missing = collect_coverage(result)
        result["missing_required_coverage"] = missing
        result["status"] = "coverage_incomplete" if missing else "completed"
    finally:
        # Each run owns a fresh process, as the offline LLM collector does.
        # The private core.shutdown path segfaults in torch emptyHostCache in
        # this CPU-wheel/MPS environment on unchanged main as well. Requests
        # are drained and MLX synchronized above; process teardown is used.
        trace.restore()


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=("main", "candidate"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scenario", choices=("mixed", "pressure"), required=True)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.blocks < 0 or args.max_steps <= 224 or args.timeout_seconds <= 0:
        parser.error("blocks must be nonnegative, timeout positive, max-steps >224")
    ROOT = args.source_root.resolve()
    if not (ROOT / "vllm_metal/v1/model_runner.py").is_file():
        parser.error("source-root does not contain the model runner")
    os.environ["VLLM_METAL_UNIFIED_CACHE"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Using this checkout is explicit even when the collector is run by path.
    sys.path.insert(0, str(ROOT))
    before = source_snapshot()
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "status": "running",
        "evidence_valid": False,
        "scope": "controlled engine-step arrivals; scheduler/output regression; no throughput claim",
        "variant": args.variant,
        "scenario": args.scenario,
        "max_num_seqs": args.max_num_seqs,
        "sampling": {"temperature": 0, "logprobs": None, "ignore_eos": True},
        "collector_sha256": collector_hash,
        "model": args.model,
        "requested_blocks": args.blocks,
        "max_model_len": args.max_model_len,
        "memory_fraction": args.memory_fraction,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "python_executable": sys.executable,
        "source_root": str(ROOT),
        "source_sha256_before_imports": before,
        "checkout_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "storage": [],
        "legacy_storage": [],
        "decode_pipeline_submissions": 0,
        "cases": {},
        "scheduler_trace": [],
        "forward_trace": [],
        "events": [],
    }
    save_result(args.output, result)
    # A native watchdog terminates even if an engine call never returns. The
    # checkpoint file remains explicitly running/evidence_valid=False then.
    faulthandler.dump_traceback_later(args.timeout_seconds, exit=True)
    try:
        run(args, result)
    except Exception as exc:
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        result["coverage"], result["missing_required_coverage"] = collect_coverage(
            result
        )
    finally:
        after = source_snapshot()
        changed = sorted(
            name
            for name in before.keys() | after.keys()
            if before.get(name) != after.get(name)
        )
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != collector_hash:
            changed.append("collector")
        if "upstream_source_sha256" in result:
            upstream = Path(result["vllm_package"]).parent
            result["upstream_source_sha256_after_run"] = upstream_snapshot(upstream)
            if (
                result["upstream_source_sha256_after_run"]
                != result["upstream_source_sha256"]
            ):
                changed.append("upstream runtime sources")
        result["source_sha256_after_run"] = after
        result["source_changed_during_run"] = changed
        if changed:
            result["status"] = "invalid_source_changed"
        result["evidence_valid"] = not changed and result["status"] in (
            "completed",
            "coverage_incomplete",
        )
        save_result(args.output, result)
        faulthandler.cancel_dump_traceback_later()
    print(f"CONTINUOUS CHECK {result['status'].upper()}: {args.output}", flush=True)
    if result.get("missing_required_coverage"):
        print(
            "COVERAGE NOT REACHED: " + ", ".join(result["missing_required_coverage"]),
            flush=True,
        )
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
