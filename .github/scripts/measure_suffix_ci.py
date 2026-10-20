"""Measure suffix CI overhead without loading or downloading any model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--measure-cached-flow", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
        VIRTUAL_ENV=sys.prefix,
        PYTHONPATH=str(args.repo),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    records = []

    def run(label, command, *, extra_env=None):
        start = time.perf_counter()
        result = subprocess.run(
            command, cwd=args.repo, env=env | (extra_env or {}),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        item = {
            "label": label, "seconds": time.perf_counter() - start,
            "returncode": result.returncode, "command": command,
        }
        (args.output / f"{label}.log").write_text(result.stdout)
        records.append(item)
        (args.output / "partial.json").write_text(json.dumps(records, indent=2))
        print(json.dumps({k: v for k, v in item.items() if k != "command"}), flush=True)
        if result.returncode:
            print(result.stdout, flush=True)
            raise SystemExit(result.returncode)
        return item

    before = list(args.wheel_dir.glob("arctic_inference-0.3.0-*.whl"))
    if not before:
        run("build-tools", [sys.executable, "-m", "pip", "install",
                           "cmake", "ninja", "grpcio-tools", "wheel"])
        run("build-wheel", [
            sys.executable, "-m", "pip", "wheel", "--no-cache-dir",
            "--no-build-isolation", "--no-deps", "--no-binary", "arctic-inference",
            "--wheel-dir", str(args.wheel_dir), "arctic-inference==0.3.0",
        ], extra_env={
            "Python_ROOT_DIR": sys.prefix,
            "Python_FIND_STRATEGY": "LOCATION",
            "MACOSX_DEPLOYMENT_TARGET": "15.0",
        })
    wheels = list(args.wheel_dir.glob("arctic_inference-0.3.0-*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]

    # Every target is empty: existing Arctic installations cannot satisfy these.
    for i in range(args.repeats):
        target = args.output / f"installed-{i}"
        assert not target.exists(), target
        run(f"wheel-install-{i}", [
            "uv", "pip", "install", "--python", sys.executable,
            "--target", str(target), "--no-deps", "--no-index", str(wheel),
        ])
        run(f"backend-check-{i}", [
            sys.executable, "-c",
            "import arctic_inference; "
            "from arctic_inference.suffix_decoding import SuffixDecodingCache; "
            "SuffixDecodingCache(max_tree_depth=24); "
            "print(arctic_inference.__file__)",
        ], extra_env={"PYTHONPATH": str(target) + os.pathsep + str(args.repo)})
        loaded = (args.output / f"backend-check-{i}.log").read_text()
        assert str(target) in loaded, loaded
    env["PYTHONPATH"] = str(args.output / "installed-0") + os.pathsep + str(args.repo)

    common = [
        "tests/test_continuation_audit.py",
        "tests/test_spec_decode_metadata.py",
        "tests/test_suffix_proposer.py",
    ]

    def pytest_run(label, paths, marker):
        xml = args.output / f"{label}.xml"
        item = run(label, [
            sys.executable, "-m", "pytest", *paths, "-m", marker,
            "-q", "--strict-markers", f"--junitxml={xml}",
        ])
        cases = list(ET.parse(xml).iter("testcase"))
        assert not any(c.find("skipped") is not None for c in cases), label
        item["cases"] = len(cases)
        item["case_seconds"] = sum(float(c.get("time", 0)) for c in cases)
        item["node_ids"] = [c.get("classname", "") + "::" + c.get("name", "") for c in cases]
        return item

    pytest_run("warmup", common, "not slow")
    pairs = []
    for i in range(args.repeats):
        # Alternate order; use fresh Python processes for every pytest call.
        order = ["split", "merged"] if i % 2 == 0 else ["merged", "split"]
        pair = {}
        for arm in order:
            if arm == "split":
                baseline = pytest_run(f"baseline-{i}", common, "not slow and not suffix")
                extra = pytest_run(f"suffix-alone-{i}", ["tests/test_suffix_proposer.py"], "not slow")
                pair["baseline"] = baseline
                pair["extra"] = extra
            else:
                pair["merged"] = pytest_run(f"merged-{i}", common, "not slow")
        assert set(pair["baseline"]["node_ids"]) | set(pair["extra"]["node_ids"]) == set(pair["merged"]["node_ids"])
        pairs.append({
            "baseline_seconds": pair["baseline"]["seconds"],
            "split_seconds": pair["baseline"]["seconds"] + pair["extra"]["seconds"],
            "merged_seconds": pair["merged"]["seconds"],
            "saved_process_seconds": pair["baseline"]["seconds"] + pair["extra"]["seconds"] - pair["merged"]["seconds"],
            "merged_increment_seconds": pair["merged"]["seconds"] - pair["baseline"]["seconds"],
            "baseline_cases": pair["baseline"]["cases"],
            "merged_cases": pair["merged"]["cases"],
            "same_unique_tests": True,
        })
    metrics = {name: statistics.median(p[name] for p in pairs) for name in pairs[0] if name.endswith("_seconds")}
    metrics["wheel_install_seconds"] = statistics.median(r["seconds"] for r in records if r["label"].startswith("wheel-install-"))
    metrics["backend_check_seconds"] = statistics.median(r["seconds"] for r in records if r["label"].startswith("backend-check-"))
    flows = []
    if args.measure_cached_flow:
        for i in range(args.repeats):
            order = ["baseline", "flow"] if i % 2 == 0 else ["flow", "baseline"]
            row = {}
            for arm in order:
                if arm == "baseline":
                    row["baseline_seconds"] = pytest_run(
                        f"flow-baseline-{i}", common, "not slow and not suffix"
                    )["seconds"]
                else:
                    target = args.output / f"flow-installed-{i}"
                    assert not target.exists()
                    start = time.perf_counter()
                    run(f"flow-install-{i}", [
                        "uv", "pip", "install", "--python", sys.executable,
                        "--target", str(target), "--no-deps", "--no-index", str(wheel),
                    ])
                    env["PYTHONPATH"] = str(target) + os.pathsep + str(args.repo)
                    run(f"flow-check-{i}", [
                        sys.executable, "-c",
                        "from arctic_inference.suffix_decoding import SuffixDecodingCache; "
                        "SuffixDecodingCache(max_tree_depth=24)",
                    ])
                    pytest_run(f"flow-pytest-{i}", common, "not slow")
                    row["flow_seconds"] = time.perf_counter() - start
            row["increment_seconds"] = row["flow_seconds"] - row["baseline_seconds"]
            flows.append(row)
        metrics["cached_flow_seconds"] = statistics.median(r["flow_seconds"] for r in flows)
        metrics["cached_flow_increment_seconds"] = statistics.median(r["increment_seconds"] for r in flows)
    summary = {
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "python": sys.version,
        "wheel": str(wheel),
        "wheel_bytes": wheel.stat().st_size,
        "wheel_cache_hit": bool(before),
        "repeats": args.repeats,
        "scope": "Focused existing unit suite, not full CI or a model/E2E workload. Installation targets are empty; common dependencies are already prepared. Cache service transfer is excluded.",
        "metrics_median": metrics,
        "pairs": pairs,
        "cached_flows": flows,
        "records": records,
    }
    (args.output / "RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
