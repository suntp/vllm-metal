# SPDX-License-Identifier: Apache-2.0
"""Public native ABI capabilities are distinct from test-kernel availability."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_metal.metal import paged_attention_capabilities


def test_structured_capabilities_take_precedence_over_legacy_probes():
    query = MagicMock(
        return_value={
            "gqa_decode": False,
            "gqa_disable": False,
            "decode_routing_metadata": True,
        }
    )
    legacy = MagicMock(side_effect=AssertionError("must not query legacy ABI"))
    ops = SimpleNamespace(
        paged_attention_capabilities=query,
        supports_gqa_decode_control=legacy,
        supports_decode_routing_metadata=legacy,
        _has_gqa_decode_kernel=legacy,
    )
    assert paged_attention_capabilities(ops) == {
        "gqa_decode": False,
        "gqa_disable": False,
        "decode_routing_metadata": True,
    }
    query.assert_called_once_with()
    legacy.assert_not_called()


@pytest.mark.parametrize("kind", ["pre_gqa", "control", "selector", "unknown_gqa"])
def test_legacy_capabilities_preserve_disable_safety(kind):
    ops = SimpleNamespace()
    pipeline = MagicMock(side_effect=AssertionError("must not create pipelines"))
    if kind == "control":
        ops.supports_gqa_decode_control = lambda: True
    elif kind == "selector":
        ops.gqa_decode_shape_eligible = lambda: True
    elif kind == "unknown_gqa":
        ops._has_gqa_decode_kernel = pipeline
    capabilities = paged_attention_capabilities(ops)
    assert capabilities["gqa_decode"] is (kind != "pre_gqa")
    assert capabilities["gqa_disable"] is (kind in {"control", "selector"})
    assert capabilities["decode_routing_metadata"] is False
    pipeline.assert_not_called()


def test_missing_capabilities_fail_closed():
    ops = SimpleNamespace(paged_attention_capabilities=lambda: {})
    assert not any(paged_attention_capabilities(ops).values())
