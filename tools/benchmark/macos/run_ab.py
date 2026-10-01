# SPDX-License-Identifier: Apache-2.0
"""Server-protocol A/B benchmark for macOS (#713 discipline, automated).

Boots one ``vllm serve`` process per arm, alternates arm order across
repeats, gates every measurement block on a quiet GPU window, warms each
(arm, length) by real decode seconds, and writes a machine-readable
evidence pack.  Absolute numbers therefore come from the server topology
only; the actual dispatch family is read back from the worker when
available.  Arms default to an on/off pair over one gate environment
variable; ``--arm`` generalizes this to named arms with full environment
control (including ``PYTHONPATH`` to compare source trees).

Example::

    python -m tools.benchmark.macos.run_ab \
        --model /path/to/local/snapshot \
        --output reports/ab-run \
        --lengths 32768 65536 131072

The model must already exist locally; this tool never downloads weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

from tools.benchmark.macos import completions, evidence, gpu_state
from tools.benchmark.macos.server import (
    ServeConfig,
    request_json,
    start_server,
    stop_server,
)

SCHEMA = "vllm-metal-macos-ab/1"
DEFAULT_GATE_ENV = "VLLM_METAL_DISABLE_GQA_DECODE"
DEFAULT_PREFIX = (
    "Technical notes on inference, scheduling, memory bandwidth and attention.\n"
)
DEFAULT_SUFFIX = "\nContinue counting, one integer per line, to 500:\n1\n2\n3\n"


def assemble_prompt(
    prefix_ids: list[int],
    suffix_ids: list[int],
    length: int,
    bos_id: int | None = None,
) -> list[int]:
    """Deterministic prompt of exactly *length* tokens: BOS + prefix fill
    + suffix.  Pure so tests can pin the layout."""
    head = 1 if bos_id is not None else 0
    body = length - len(suffix_ids) - head
    if body < len(prefix_ids):
        raise ValueError(f"length {length} too small for prefix/suffix")
    ids = ([bos_id] if bos_id is not None else []) + (
        prefix_ids * (body // len(prefix_ids) + 1)
    )[:body]
    ids = ids + suffix_ids
    if len(ids) != length:
        raise RuntimeError(f"assembled {len(ids)} tokens, expected {length}")
    return ids


def parse_expect_family(raw: str | None) -> dict[str, str]:
    """Parse ``on=gqa_decode,off=per_token_ps512`` into a dict."""
    expected: dict[str, str] = {}
    if not raw:
        return expected
    for part in raw.split(","):
        arm, _, family = part.partition("=")
        if not arm or not family:
            raise ValueError(f"--expect-family: bad entry {part!r}")
        expected[arm.strip()] = family.strip()
    return expected


def parse_key_values(raw: list[str]) -> dict[str, str]:
    """Parse repeatable ``KEY=VALUE`` arguments into a dict."""
    parsed: dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        parsed[key] = value
    return parsed


def parse_arms(raw: list[str]) -> list[tuple[str, dict[str, str]]]:
    """Parse repeatable ``NAME[:KEY=VAL[,KEY=VAL...]]`` arm specs.

    The first arm is the baseline every other arm is compared against.
    An arm with no ``KEY=VAL`` part differs from the others only in what
    its own environment sets.  Arms that override ``PYTHONPATH`` (to run
    a different source tree) should keep the harness checkout on it so
    the worker probe stays importable inside the server process.
    """
    arms: list[tuple[str, dict[str, str]]] = []
    seen: set[str] = set()
    for item in raw:
        name, _, env_raw = item.partition(":")
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"--arm needs unique non-empty names, got {item!r}")
        entries = [entry for entry in env_raw.split(",") if entry.strip()]
        arms.append((name, parse_key_values(entries)))
        seen.add(name)
    return arms


def arm_order(names: list[str], rep: int) -> list[str]:
    """Arm order for one repeat: rotate by *rep* so no arm always runs
    first.  A fixed arm order reads as a systematic gain for whichever
    arm happens to be second (#713)."""
    count = len(names)
    return [names[(rep + index) % count] for index in range(count)]


def run_fingerprint(arms: list[tuple[str, dict[str, str]]], args) -> str:
    """Identity of a benchmark configuration, for ``--resume``.

    Covers everything that changes the numbers (model, lengths, arms,
    timing parameters); excludes the output location and the port.
    """
    payload = {
        "model": str(args.model.resolve()),
        "lengths": sorted(args.lengths),
        "tokens": args.tokens,
        "runs": args.runs,
        "reps": args.reps,
        "seed": args.seed,
        "warmup_decode_seconds": args.warmup_decode_seconds,
        "arms": arms,
        "extra_env": args.extra_env,
        "expect_family": args.expect_family,
        "server_arg": sorted(args.server_arg),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_args(args: argparse.Namespace, config: dict) -> int:
    """Shared validation; returns the derived max-model-len."""
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"no config.json under {args.model}")
    if args.runs < 1 or args.tokens < 2:
        raise SystemExit("--runs >= 1 and --tokens >= 2 required")
    if not args.lengths:
        raise SystemExit("at least one --length required")

    def cfg_value(*keys: str) -> int | None:
        for key in keys:
            value = config.get(key)
            if isinstance(value, int):
                return value
            nested = config.get("text_config")
            if isinstance(nested, dict) and isinstance(nested.get(key), int):
                return nested[key]
        return None

    max_len = args.max_model_len or cfg_value("max_position_embeddings")
    if not max_len:
        raise SystemExit(
            "no max_position_embeddings in config.json; pass --max-model-len"
        )
    longest = max(args.lengths) + args.tokens
    if longest > max_len:
        raise SystemExit(f"longest context {longest} exceeds max-model-len {max_len}")
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError:
            raise SystemExit(
                f"port {args.port} already in use; refusing to share a "
                "server with a benchmark"
            ) from None
    return max_len


def read_dispatch(base_url: str) -> dict:
    """Actual dispatch metadata via the worker probe, or a placeholder."""
    try:
        response = request_json(
            base_url,
            "/collective_rpc",
            {"method": "paged_dispatch_probe"},
            timeout=30,
        )
    except Exception:  # noqa: BLE001 - probe is best-effort by design
        return {"family": "unavailable"}
    results = response.get("results") or [response]
    entry = results[0] if results else {}
    if not isinstance(entry, dict):
        return {"family": "unavailable"}
    return entry


def measure_lengths(
    args: argparse.Namespace,
    arm: str,
    handle,
    prompts: dict[int, list[int]],
) -> tuple[dict[int, list[dict]], dict[int, float]]:
    """All lengths for one already-booted arm.

    Returns ``(run records, warmup decode seconds per length)``.  Each
    length is warmed by at least ``--warmup-decode-seconds`` of real
    decode (page cache, clocks and shader caches settle) before the
    measured runs; short spot warmups read as a regression that is not
    (#713).
    """
    records: dict[int, list[dict]] = {}
    warmup: dict[int, float] = {}
    for length in sorted(args.lengths):
        prompt_ids = prompts[length]
        # Cold prefill primes the prefix cache; its decode is discarded.
        completions.stream_completion(
            handle.base_url,
            args.served_model_name,
            prompt_ids,
            max_tokens=8,
            seed=args.seed,
        )
        # Discarded warmup decode until the configured seconds accumulate.
        warm_seconds = 0.0
        while warm_seconds < args.warmup_decode_seconds:
            warm = completions.stream_completion(
                handle.base_url,
                args.served_model_name,
                prompt_ids,
                max_tokens=args.tokens,
                seed=args.seed,
            )
            warm_seconds += warm.decode_s
        warmup[length] = warm_seconds
        print(
            f"  length {length}: warm decode {warm_seconds:.1f}s, measuring",
            flush=True,
        )
        runs: list[dict] = []
        for index in range(args.runs):
            utilization = gpu_state.read_gpu_utilization()
            result = completions.stream_completion(
                handle.base_url,
                args.served_model_name,
                prompt_ids,
                max_tokens=args.tokens,
                seed=args.seed + index,
            )
            if result.prompt_tokens != length:
                raise RuntimeError(
                    f"prompt reshaped: sent {length}, server saw {result.prompt_tokens}"
                )
            dispatch = read_dispatch(handle.base_url)
            family = dispatch.get("family", "unavailable")
            expected = args.expect_family.get(arm)
            if expected and family != expected:
                raise RuntimeError(
                    f"arm {arm}: dispatch reported {family!r}, expected {expected!r}"
                )
            run = {
                "length": length,
                "run": index,
                "ttft_s": result.ttft_s,
                "decode_s": result.decode_s,
                "decode_tps": result.decode_tps,
                "completion_tokens": result.completion_tokens,
                "text_sha256": result.text_sha256,
                "token_ids": result.token_ids,
                "token_ids_sha256": result.token_ids_sha256,
                "token_event_sizes": result.token_event_sizes,
                "token_event_elapsed_s": result.token_event_elapsed_s,
                "dispatch_family": family,
                "dispatch": dispatch,
                "utilization_before": utilization,
            }
            runs.append(run)
            print(
                f"  length {length} run {index}: "
                f"{result.decode_tps:.2f} tok/s "
                f"({family}, util {utilization}%)",
                flush=True,
            )
        records[length] = runs
    return records, warmup


def run(args: argparse.Namespace) -> dict:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model / "config.json").read_text())
    max_model_len = validate_args(args, config)

    environment = {
        "git": evidence.git_state(),
        "platform": evidence.platform_record(),
        "versions": evidence.package_versions(),
        "model": evidence.model_digest(args.model),
        "server_args": {
            "max_model_len": max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": 1,
            "enable_prefix_caching": True,
        },
    }
    result: dict = {
        "schema": SCHEMA,
        "started_utc": datetime.now(UTC).isoformat(),
        "args": {
            "lengths": sorted(args.lengths),
            "gate_env": args.gate_env,
            "reps": args.reps,
            "runs": args.runs,
            "tokens": args.tokens,
            "seed": args.seed,
            "extra_env": args.extra_env,
            "expect_family": args.expect_family,
            "quiet_threshold_pct": args.quiet_threshold,
        },
        "environment": environment,
    }
    evidence.write_json(output / "results.json", result)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix_ids = tokenizer.encode(DEFAULT_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(DEFAULT_SUFFIX, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    prompts = {
        length: assemble_prompt(prefix_ids, suffix_ids, length, bos)
        for length in sorted(args.lengths)
    }
    result["prompt_sha256"] = {
        str(length): hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        for length, ids in prompts.items()
    }
    evidence.write_json(output / "results.json", result)

    arm_plan = (
        parse_arms(args.arm)
        if args.arm
        else [
            ("on", {args.gate_env: args.gate_on_value}),
            ("off", {args.gate_env: args.gate_off_value}),
        ]
    )
    arm_env = dict(arm_plan)
    arm_names = [name for name, _ in arm_plan]
    result["arms_plan"] = [{"name": name, "env": env} for name, env in arm_plan]
    result["fingerprint"] = run_fingerprint(arm_plan, args)

    arms: dict[str, dict[int, list[dict]]] = {name: {} for name in arm_names}
    arm_orders = [arm_order(arm_names, rep) for rep in range(args.reps)]
    result["arm_orders"] = arm_orders

    # --resume picks up a crashed or interrupted run at exactly the same
    # configuration; a different fingerprint refuses rather than mixing.
    units_done: set[str] = set()
    if args.resume and (output / "results.json").exists():
        saved = json.loads((output / "results.json").read_text())
        if saved.get("fingerprint") != result["fingerprint"]:
            raise SystemExit(
                "existing results.json was produced with a different "
                "configuration; choose a new --output"
            )
        for name, per_length in saved.get("arms", {}).items():
            arms[name] = {
                int(length): list(runs) for length, runs in per_length.items()
            }
        units_done = set(saved.get("units_done", []))
        print(f"resuming: {len(units_done)} unit(s) already complete", flush=True)

    for rep, order in enumerate(arm_orders):
        for arm in order:
            unit = f"rep{rep}:{arm}"
            if unit in units_done:
                print(f"rep {rep} arm {arm}: already complete, skipping", flush=True)
                continue
            quiet = (
                gpu_state.wait_quiet_window(args.quiet_threshold)
                if not args.no_quiet_wait
                else gpu_state.QuietWindow(
                    True, 0.0, [], note="disabled by --no-quiet-wait"
                )
            )
            print(
                f"rep {rep} arm {arm}: quiet window ok={quiet.ok} "
                f"({quiet.waited_s:.0f}s, {quiet.note or 'clear'})",
                flush=True,
            )
            serve_config = ServeConfig(
                model=args.model,
                port=args.port,
                served_model_name=args.served_model_name,
                env={**arm_env[arm], **args.extra_env},
                shared_env={"VLLM_SERVER_DEV_MODE": "1"},
                worker_extension_cls=(
                    "tools.benchmark.macos.dispatch_probe.MacosBenchmarkProbe"
                ),
                extra_args=[
                    "--max-model-len",
                    str(max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-num-seqs",
                    "1",
                    "--enable-prefix-caching",
                    *args.server_arg,
                ],
                log_path=output / f"{arm}.rep{rep}.server.log",
            )
            handle = None
            try:
                handle = start_server(serve_config)
                runs, warm = measure_lengths(args, arm, handle, prompts)
                for length, length_runs in runs.items():
                    arms[arm].setdefault(length, []).extend(length_runs)
                result.setdefault("warmup_decode_s", {}).setdefault(arm, {}).update(
                    {str(length): seconds for length, seconds in warm.items()}
                )
                units_done.add(unit)
                result["units_done"] = sorted(units_done)
            finally:
                if handle is not None:
                    stop_server(handle)
            result["arms"] = {
                arm: {
                    str(length): length_runs
                    for length, length_runs in per_length.items()
                }
                for arm, per_length in arms.items()
            }
            evidence.write_json(output / "results.json", result)

    reference_order = arm_names if args.arm else ["off", "on"]
    result["comparison"] = evidence.summarize(arms, reference_order)
    result["finished_utc"] = datetime.now(UTC).isoformat()
    evidence.write_json(output / "results.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local model snapshot (never downloaded)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="evidence-pack directory"
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[32768, 65536, 131072],
        help="prompt token lengths to measure",
    )
    parser.add_argument(
        "--gate-env",
        default=DEFAULT_GATE_ENV,
        help="A/B switch: 'on' arm sets it to --gate-on-value, 'off' to "
        "--gate-off-value",
    )
    parser.add_argument(
        "--gate-on-value",
        default="0",
        help="value for the 'on' arm (default 0: the gate is a disable flag)",
    )
    parser.add_argument(
        "--gate-off-value",
        default="1",
        help="value for the 'off' arm (default 1: the disable flag set)",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME[:KEY=VAL,...]",
        help="repeatable explicit arm; replaces the on/off gate pair. The "
        "first arm is the baseline others are compared against. Example: "
        "--arm base --arm staged:VLLM_METAL_GATE=1",
    )
    parser.add_argument(
        "--warmup-decode-seconds",
        type=float,
        default=10.0,
        help="discarded decode seconds per (arm, length) before measuring",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run in --output with the same "
        "configuration; completed (rep, arm) units are skipped",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="full on/off cycles; order alternates per rep",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="measured runs per (rep, arm, length)"
    )
    parser.add_argument(
        "--tokens", type=int, default=128, help="generated tokens per run"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--served-model-name", default="benchmark")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="defaults to the checkpoint's max_position_embeddings",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="KEY=VALUE applied to both arms",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra CLI flag passed through to vllm serve",
    )
    parser.add_argument(
        "--quiet-threshold",
        type=int,
        default=12,
        help="GPU utilization %% considered quiet",
    )
    parser.add_argument(
        "--no-quiet-wait",
        action="store_true",
        help="record utilization but do not wait for quiet",
    )
    parser.add_argument(
        "--expect-family",
        default=None,
        help="optional on=FAMILY,off=FAMILY dispatch check",
    )
    args = parser.parse_args(argv)
    args.extra_env = parse_key_values(args.extra_env)
    args.expect_family = parse_expect_family(args.expect_family)
    if args.arm:
        if len(args.arm) < 2:
            raise SystemExit("--arm needs at least two arms to compare")
        try:
            parse_arms(args.arm)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        gate_changed = (
            args.gate_env != DEFAULT_GATE_ENV
            or args.gate_on_value != "0"
            or args.gate_off_value != "1"
        )
        if gate_changed:
            raise SystemExit(
                "--arm replaces --gate-env/--gate-on-value/"
                "--gate-off-value; pass one or the other"
            )
    try:
        result = run(args)
    except KeyboardInterrupt:
        sys.exit(130)
    comparison = result["comparison"]
    for length, entry in comparison.items():
        medians = entry["median_tps"]
        line = ", ".join(
            f"{arm}={stats['median']:.2f} tok/s"
            for arm, stats in medians.items()
            if stats["median"]
        )
        speedups = {key: value for key, value in entry.items() if "_vs_" in key}
        print(
            f"length {length}: {line} | {speedups} | "
            f"all_text_equal={entry['all_text_equal']}"
        )


if __name__ == "__main__":
    main()
