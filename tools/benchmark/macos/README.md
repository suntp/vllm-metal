# macOS benchmarking harness

A/B benchmarks for vLLM Metal that implement the measurement discipline
from #713: server topology only (never an in-process engine for
performance claims), a quiet GPU window before every block, arm order
rotated across repeats, warmup by real decode seconds, dispatch-family
readback from the worker, and a self-describing evidence pack.

The harness is measurement infrastructure, not a campaign tool: nothing
in it knows about a particular kernel change.  Whatever a perf PR gates
behind an environment variable (or a source tree), these tools can A/B.

## Layout

| Module | Role |
|---|---|
| `run_ab.py` | End-to-end serving A/B: one `vllm serve` process per arm |
| `primitive_ab.py` | Op-level A/B on the paged-attention primitive, one process |
| `server.py` | `vllm serve` lifecycle; a benchmark never leaves a server behind |
| `completions.py` | Streaming completion client timed from returned token IDs |
| `gpu_state.py` | GPU co-tenancy probe and quiet-window wait |
| `dispatch_probe.py` | Worker extension: records the kernel family that actually ran |
| `evidence.py` | Evidence-pack helpers: environment capture, statistics, comparison |
| `setup_env.sh` | One command to make a checkout benchmark-ready |

## Setup

```sh
tools/benchmark/macos/setup_env.sh
```

Creates a per-checkout virtual environment via the repository's own
`install.sh` (pinned vllm wheel, editable install, native Metal build)
plus `pytest`.  Models are never downloaded: pass a local snapshot path
to every tool.

## Protocol 1: serving A/B (`run_ab.py`)

The client requests token IDs and measures from the first emitted token event
to the last, including events with empty decoded text. It checks the streamed
ID count against usage and rejects a bundled first-token event. Evidence keeps
token IDs, event sizes/timestamps, and the worker's kernel family and partition.
Text equality and token-ID equality are reported separately.

Dispatch recording is opt-in on newer native builds. The HTTP harness enables
it through a worker RPC before warmup; the primitive harness enables it in its
own process. Subsequent reads do not reset observations. Both timing arms use
the same diagnostic setting. Older builds with always-on recording remain
supported. This instrumentation does not alter routing or force a partition.

Arms default to an on/off pair over one gate variable:

```sh
python -m tools.benchmark.macos.run_ab \
    --model /path/to/snapshot \
    --output /path/to/evidence-dir \
    --lengths 32768 65536 131072
```

`--gate-env` / `--gate-on-value` / `--gate-off-value` parameterize the
switch (any env var, any values), and `--expect-family on=gqa_decode,off=per_token_ps512`
fails the run if an arm routed elsewhere.

Named arms generalize this to any environment difference, including
`PYTHONPATH` for comparing source trees.  The first arm is the baseline
every other arm is reported against:

```sh
python -m tools.benchmark.macos.run_ab \
    --model /path/to/snapshot \
    --output /path/to/evidence-dir \
    --arm base:PYTHONPATH=/path/to/harness:/path/to/tree-a \
    --arm cand:PYTHONPATH=/path/to/harness:/path/to/tree-b
```

An arm that overrides `PYTHONPATH` must keep the harness checkout on it:
the dispatch probe is imported inside the server process.

`--reps` full cycles with rotated arm order; `--runs` measured runs per
(rep, arm, length); `--warmup-decode-seconds` discarded decode per (arm,
length) before measuring; `--resume` continues an interrupted run in the
same `--output` at the same configuration.

## Protocol 2: op-level A/B (`primitive_ab.py`)

Times the paged-attention primitive directly, interleaving arms inside
one process (no server): the GQA kernel at forced partition sizes vs the
production split-KV path, or a single production shape in
`--candidate none` mode.  Each result row records the dispatch family,
numerics against the tree's reference implementation, per-arm medians
and the paired gain; `--previous` compares arms across runs (e.g. two
source trees measured in two invocations):

```sh
python -m tools.benchmark.macos.primitive_ab \
    --source /path/to/checkout \
    --label tree-b --out /path/to/tree-b.json \
    --previous /path/to/tree-a.json
```

`--geometry` selects preset attention shapes; `--heads/--kv-heads/
--head-size/--block-size` bench any other shape.  Without `--kv-len`
each partition is measured at its auto-route threshold
`P * ceil(33 * cores / q_heads)`.

## Evidence packs

Both protocols write JSON containing the source revision, platform,
package versions, model-config digest, full argument vector, prompt
hashes, per-run dispatch families, GPU utilization samples and the
arm-vs-baseline comparison.  Two packs from different machines or trees
can be compared directly: everything that shaped the numbers is inside
the file.
