# SPDX-License-Identifier: Apache-2.0
"""One e2e arm: real model + TurboQuant prefill lane end to end.

Runs offline vLLM on a supplied model (originally Qwen3.8-27B-4bit),
greedy-decodes the newspaper prompt, and reports actual prompt/token counts
and generation wall time as JSON. TTFT is null when vLLM does not return it.
This is an in-process benchmark, not serving throughput. The tq-reference
arm disables only the prefill planner; both TQ arms use sdpa_forward.

    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model --arm tq
"""

import argparse
import importlib.metadata
import json
import os
import sys
import time

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
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=("bf16", "tq", "tq-reference"), required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--k-quant", default="q8_0")
    ap.add_argument("--v-quant", default="q3_0")
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    kwargs = {}
    if args.arm == "tq-reference":
        from vllm_metal.attention.impls import sdpa

        sdpa._turboquant_prefill_plan = lambda *args, **kwargs: None
    if args.arm != "bf16":
        kwargs["additional_config"] = {
            "turboquant": True,
            "k_quant": args.k_quant,
            "v_quant": args.v_quant,
        }
    t0 = time.perf_counter()
    llm = LLM(
        model=os.path.expanduser(args.model),
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
                "model": args.model,
                "k_quant": args.k_quant,
                "v_quant": args.v_quant,
                "versions": {
                    package: importlib.metadata.version(package)
                    for package in ("vllm", "mlx", "mlx-lm")
                },
                "mlx_enable_tf32": os.getenv("MLX_ENABLE_TF32"),
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
