# SPDX-License-Identifier: Apache-2.0
"""Worker extension used to record the actual C++ dispatch per request.

Passed to ``vllm serve`` as ``--worker-extension-cls``; the harness polls
it through ``/collective_rpc`` after every measured request so the
evidence pack records which kernel family actually ran, not which one
the gate was expected to choose.  Routing facts are topology-independent
(in-process vs server does not change which kernel a request dispatches
to), so reading them from the worker is safe even though throughput
claims must come from the server protocol (#713).

The class must be importable inside the worker process.  From an
editable/source checkout that is automatic; from a wheel install the
harness records ``unavailable`` instead of failing.
"""

from __future__ import annotations


class MacosBenchmarkProbe:
    """Worker extension: one-arg-free RPC targets for the harness."""

    def paged_dispatch_probe(self) -> dict:
        """Dispatch family of the most recent paged-attention eval."""
        try:
            from vllm_metal.metal import get_ops

            ops = get_ops()
            reader = getattr(ops, "last_paged_dispatch", None)
            family = reader() if callable(reader) else "unavailable"
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            return {"family": "unavailable", "error": repr(exc)}
        return {"family": family}

    def memory_probe(self) -> dict:
        """Peak MLX memory since process start (high-water mark)."""
        try:
            import mlx.core as mx

            return {"peak_memory_bytes": int(mx.get_peak_memory())}
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            return {"peak_memory_bytes": None, "error": repr(exc)}
