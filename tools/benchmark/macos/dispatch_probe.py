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
    """Worker RPC targets for benchmark setup and observations."""

    def paged_dispatch_probe(self, enable: bool = False) -> dict:
        """Opt in before requests; subsequent reads preserve the observation."""
        try:
            from vllm_metal.metal import get_ops

            ops = get_ops()
            if enable:
                import mlx.core as mx

                mx.synchronize()
                setter = getattr(ops, "_set_paged_dispatch_diagnostics", None)
                if callable(setter):
                    setter(True)
            reader = getattr(ops, "last_paged_dispatch", None)
            family = reader() if callable(reader) else "unavailable"
            partition_reader = getattr(ops, "last_gqa_partition_size", None)
            partition = partition_reader() if callable(partition_reader) else None
            core_reader = getattr(ops, "detected_gpu_core_count", None)
            cores = core_reader() if callable(core_reader) else None
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            return {"family": "unavailable", "error": repr(exc)}
        return {"family": family, "partition": partition, "gpu_cores": cores}

    def memory_probe(self) -> dict:
        """Peak MLX memory since process start (high-water mark)."""
        try:
            import mlx.core as mx

            return {"peak_memory_bytes": int(mx.get_peak_memory())}
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            return {"peak_memory_bytes": None, "error": repr(exc)}
