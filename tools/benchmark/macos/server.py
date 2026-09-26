# SPDX-License-Identifier: Apache-2.0
"""``vllm serve`` lifecycle for server-protocol benchmarks (#713).

Absolute performance claims come from the server topology only: an
in-process engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) shares the
interpreter and the GIL with the driver's output processing, understating
long-context decode by 1.5-2.1x and compressing A/B ratios.  This module
boots one ``vllm serve`` arm per A/B leg with an explicit environment and
ALWAYS terminates it -- on success, on error and on interruption.  A
benchmark must never leave a server behind.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServeConfig:
    """One arm of an A/B comparison."""

    model: Path
    port: int
    served_model_name: str
    # Arm-specific environment (the A/B switch, e.g. a gate override).
    env: dict[str, str] = field(default_factory=dict)
    # Environment applied to every arm (offline flags, feature toggles).
    shared_env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    worker_extension_cls: str | None = None
    startup_timeout_s: float = 600.0
    log_path: Path = Path("server.log")


@dataclass
class ServeHandle:
    proc: subprocess.Popen[bytes]
    base_url: str
    log_path: Path
    env: dict[str, str]


def request_json(
    base_url: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 10.0,
) -> dict:
    """POST (or GET when *payload* is None) and parse a JSON response."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body.strip() else {}


def health_ok(base_url: str, timeout: float = 5.0) -> bool:
    """True when ``/health`` answers 200 (its body may be empty)."""
    try:
        urllib.request.urlopen(base_url + "/health", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def start_server(config: ServeConfig) -> ServeHandle:
    """Boot one ``vllm serve`` arm and block until ``/health`` passes."""
    base_url = f"http://127.0.0.1:{config.port}"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--served-model-name",
        config.served_model_name,
        *config.extra_args,
    ]
    if config.worker_extension_cls:
        command += ["--worker-extension-cls", config.worker_extension_cls]
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.update(config.shared_env)
    env.update(config.env)
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + config.startup_timeout_s
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited during startup; see {config.log_path}"
                    )
                if health_ok(base_url):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"server startup timeout; see {config.log_path}")
                time.sleep(1.0)
        except BaseException:
            stop_server(
                ServeHandle(
                    proc=proc,
                    base_url=base_url,
                    log_path=config.log_path,
                    env={},
                )
            )
            raise
    recorded = {
        key: value
        for key, value in env.items()
        if key.startswith(("VLLM_", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
    }
    return ServeHandle(
        proc=proc, base_url=base_url, log_path=config.log_path, env=recorded
    )


def stop_server(handle: ServeHandle) -> None:
    """Terminate a server arm; safe to call more than once."""
    proc = handle.proc
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
