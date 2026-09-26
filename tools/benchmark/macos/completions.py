# SPDX-License-Identifier: Apache-2.0
"""Streaming ``/v1/completions`` client for decode-throughput measurement.

Decode tok/s is counted from returned token IDs, not streamed text
chunks: ``(completion_tokens - 1) / (last_token_time - first_token_time)``
— the first token is prefill (TTFT), every later token is decode.  The
response text and its hash are returned so A/B arms can be compared for
output equality.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class CompletionResult:
    ttft_s: float
    decode_s: float
    decode_tps: float
    prompt_tokens: int
    completion_tokens: int
    text: str
    text_sha256: str


def stream_completion(
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    *,
    seed: int = 0,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    timeout_s: float = 1800.0,
) -> CompletionResult:
    """One streaming completion; raises when usage does not add up."""
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first: float | None = None
    last: float | None = None
    pieces: list[str] = []
    usage: dict | None = None
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(f"server error: {data['error']}")
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                piece = choice.get("text", "")
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    pieces.append(piece)
    if usage is None or first is None or last is None or last <= first:
        raise RuntimeError(f"incomplete streaming response: usage={usage}")
    text = "".join(pieces)
    completion_tokens = int(usage["completion_tokens"])
    if completion_tokens != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} completion tokens, got {completion_tokens}"
        )
    decode_s = last - first
    return CompletionResult(
        ttft_s=first - start,
        decode_s=decode_s,
        decode_tps=(completion_tokens - 1) / decode_s,
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=completion_tokens,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
