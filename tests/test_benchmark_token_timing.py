# SPDX-License-Identifier: Apache-2.0
"""Streaming timing follows token events, including buffered empty text."""

import json

import pytest

from tools.benchmark.macos import completions


def _stream(monkeypatch, events, completion_tokens):
    clock = [0.0]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for stamp, text, ids in events:
                clock[0] = stamp
                yield (
                    "data: "
                    + json.dumps({"choices": [{"text": text, "token_ids": ids}]})
                    + "\n"
                ).encode()
            clock[0] = 20.0  # Usage delivery must not extend decode timing.
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": 1,
                        },
                    }
                )
                + "\n"
            ).encode()
            yield b"data: [DONE]\n"

    def urlopen(request, **kwargs):
        assert json.loads(request.data)["return_token_ids"] is True
        return Response()

    monkeypatch.setattr(completions.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(completions.time, "perf_counter", lambda: clock[0])
    return completions.stream_completion(
        "http://benchmark.invalid", "model", [1], completion_tokens
    )


def test_empty_boundary_text_preserves_token_timing(monkeypatch):
    result = _stream(monkeypatch, [(1, "", [11]), (2, "x", [12]), (3, "", [13])], 3)
    assert result.ttft_s == 1
    assert result.decode_s == 2
    assert result.decode_tps == 1
    assert result.text == "x"
    assert result.token_ids == [11, 12, 13]
    assert result.token_event_elapsed_s == [1, 2, 3]


def test_missing_token_ids_are_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match="streamed 2 token IDs"):
        _stream(monkeypatch, [(1, "a", [11]), (2, "b", None), (3, "c", [13])], 3)


def test_bundled_first_token_is_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match="first token event"):
        _stream(monkeypatch, [(1, "ab", [11, 12]), (2, "c", [13])], 3)
