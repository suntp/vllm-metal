# SPDX-License-Identifier: Apache-2.0
"""macOS benchmarking harness for server-protocol A/B measurements (#713).

The modules here automate the measurement discipline documented in
``docs/benchmarking-macos.md``: absolute numbers come from ``vllm serve``
arms (never an in-process engine), every measurement is gated on a quiet
GPU window, arm order alternates across repeats, and results land in a
machine-readable evidence pack.  Entry point::

    python -m tools.benchmark.macos.run_ab --model /path/to/snapshot ...
"""
