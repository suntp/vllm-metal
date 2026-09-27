# SPDX-License-Identifier: Apache-2.0
"""One e2e arm: real model + TurboQuant prefill lane end to end.

Runs offline vLLM on the local MiniCPM5-2B (hs=128, 16q/2kv, 42 layers),
greedy-decodes a ~2k-token prefill prompt, and reports token ids + TTFT
metrics as JSON. Invoked once per arm by the driver; TurboQuant is
enabled through additional_config (the production switch).

    python tools/benchmark/tq_e2e_arm.py --arm bf16|tq
"""

import argparse
import json
import os
import sys
import time

MODEL = os.path.expanduser(
    "~/.cache/modelscope/models/mlx-community--Qwen3.8-27B-4bit/snapshots/master"
)
PARA = (
    "The city library opened its doors at eight in the morning, and by nine "
    "the reading rooms were already half full. Students spread their notes "
    "across the long oak tables, older visitors settled into the armchairs "
    "by the tall windows, and the librarians moved quietly between the "
    "shelves, returning books to their places. "
)
PROMPT = (
    "Below is a passage from a local newspaper. Read it carefully.\n\n"
    + PARA * 18
    + "\n\nBased on the passage above, describe in detail what a typical "
    "morning at the library looks like. Your description:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("bf16", "tq"), required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    kwargs = {}
    if args.arm == "tq":
        kwargs["additional_config"] = {
            "turboquant": True,
            "k_quant": "q8_0",
            "v_quant": "q3_0",
        }
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.7,
        **kwargs,
    )
    load_s = time.perf_counter() - t0

    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    t1 = time.perf_counter()
    outs = llm.generate([PROMPT], sp)
    gen_s = time.perf_counter() - t1

    o = outs[0]
    toks = list(o.outputs[0].token_ids)
    m = o.metrics
    ttft = None
    try:
        ttft = m.first_token_time - m.arrival_time
    except (AttributeError, TypeError):
        ttft = None
    text = o.outputs[0].text
    print(
        json.dumps(
            {
                "arm": args.arm,
                "prompt_tokens": len(o.prompt_token_ids),
                "tokens": toks,
                "ttft_s": round(ttft, 3) if ttft else None,
                "gen_wall_s": round(gen_s, 3),
                "load_s": round(load_s, 1),
                "text_head": text[:60],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
