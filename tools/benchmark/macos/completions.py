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
    token_ids: list[int]
    token_ids_sha256: str
    token_event_sizes: list[int]
    token_event_elapsed_s: list[float]


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
        "return_token_ids": True,
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
    token_ids: list[int] = []
    token_event_sizes: list[int] = []
    token_event_elapsed_s: list[float] = []
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
                    pieces.append(piece)
                ids = choice.get("token_ids") or []
                if ids:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    token_ids.extend(ids)
                    token_event_sizes.append(len(ids))
                    token_event_elapsed_s.append(now - start)
    if usage is None or first is None or last is None or last <= first:
        raise RuntimeError(f"incomplete streaming response: usage={usage}")
    text = "".join(pieces)
    completion_tokens = int(usage["completion_tokens"])
    if len(token_ids) != completion_tokens:
        raise RuntimeError(
            f"streamed {len(token_ids)} token IDs, usage reports {completion_tokens}"
        )
    if token_event_sizes[0] != 1:
        raise RuntimeError("the first token event must contain exactly one token")
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
        token_ids=token_ids,
        token_ids_sha256=hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        token_event_sizes=token_event_sizes,
        token_event_elapsed_s=token_event_elapsed_s,
    )
