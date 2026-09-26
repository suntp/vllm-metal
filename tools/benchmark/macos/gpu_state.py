# SPDX-License-Identifier: Apache-2.0
"""GPU co-tenancy probes for macOS benchmarking (#713).

WindowServer and app renderers time-share the GPU with any workload.  A
benchmark that runs inside a busy window reads ~-16% on real decode (and
far worse on short probes) and mimics a machine regression that it is
not.  This module reads the device utilization the OS already exposes and
waits for a quiet window so measurements start from a known co-tenancy
state; the observed utilization is recorded alongside every result.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

_IOREG_ARGS = ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"]
_UTILIZATION_PREFIX = '"Device Utilization %"='


def parse_utilization_text(text: str) -> int | None:
    """Peak ``Device Utilization %`` across GPU entries in *ioreg* output.

    The marker may appear anywhere in a line (ioreg prefixes tree lines
    with ``| ``), so the search is positional like ``grep -oE``.
    Returns ``None`` when no utilization entry is found: callers must
    treat the co-tenancy state as unknown, never as quiet.
    """
    values: list[int] = []
    for line in text.splitlines():
        position = line.find(_UTILIZATION_PREFIX)
        if position < 0:
            continue
        digits = line[position + len(_UTILIZATION_PREFIX) :].strip()
        if digits.isdigit():
            values.append(int(digits))
    return max(values) if values else None


def read_gpu_utilization() -> int | None:
    """Current GPU device utilization in percent, or ``None`` if unknown.

    ``None`` means the OS answer could not be obtained or parsed
    (non-macOS platform, ioreg layout change, timeout).
    """
    try:
        result = subprocess.run(
            _IOREG_ARGS,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_utilization_text(result.stdout)


@dataclass
class QuietWindow:
    """Outcome of a quiet-window wait; always recorded into the evidence."""

    ok: bool
    waited_s: float
    samples: list[int] = field(default_factory=list)
    note: str = ""


def wait_quiet_window(
    threshold_pct: int = 12,
    *,
    required_samples: int = 3,
    sample_interval_s: float = 5.0,
    timeout_s: float = 3600.0,
    settle_s: float = 5.0,
) -> QuietWindow:
    """Block until GPU utilization stays at or below *threshold_pct*.

    Recent GPU work (including the benchmark's own previous request)
    inflates the reading, so *settle_s* elapses before the first sample.
    A window counts as quiet only after *required_samples* consecutive
    readings at or below the threshold.

    Fails open with an explanatory note when utilization cannot be read:
    a benchmark must still run, it just loses the co-tenancy guarantee
    and says so in the evidence pack.
    """
    time.sleep(settle_s)
    start = time.monotonic()
    deadline = start + timeout_s
    samples: list[int] = []
    quiet_streak = 0
    while True:
        value = read_gpu_utilization()
        if value is None:
            return QuietWindow(
                True,
                time.monotonic() - start,
                samples,
                note="utilization unavailable; co-tenancy state unknown",
            )
        samples.append(value)
        quiet_streak = quiet_streak + 1 if value <= threshold_pct else 0
        if quiet_streak >= required_samples:
            return QuietWindow(True, time.monotonic() - start, samples)
        if time.monotonic() >= deadline:
            return QuietWindow(
                False,
                time.monotonic() - start,
                samples,
                note=f"timed out waiting for utilization <= {threshold_pct}%",
            )
        time.sleep(sample_interval_s)
