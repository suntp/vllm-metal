# Recipes by Mac Memory

Run a model on your own Mac for chat, coding agents and tool use. Pick your
Mac's memory below and the rest of the page follows that choice. To see it,
open the Apple menu, choose **About This Mac**, and look for **Memory**.

!!! note "Estimated starting points"
    These settings were measured on 128 GB Macs with vLLM limited to the
    memory a smaller Mac gives its graphics chip. A real Mac of that size, with
    other apps open, may need smaller settings. Start here and adjust.

## Before you start

1. Install vLLM Metal, see [Installation](installation.md).
2. Open **Terminal**: press Command-Space and type "Terminal". You paste each
   command on this page into Terminal and press Return.
3. If you used the install script, turn on vLLM Metal in that window. Do this
   in every new Terminal window:

    ```bash
    source ~/.venv-vllm-metal/bin/activate
    ```

A **token** is the unit a model reads and writes. One token is about three
quarters of a word.

## Start the server

=== "16 GB"

    Qwen3.5-9B, with room for a conversation of about 49,000 words.

    ```bash
    vllm serve mlx-community/Qwen3.5-9B-MLX-4bit \
      --host 127.0.0.1 --max-model-len 65536 --max-num-batched-tokens 2048 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3
    ```

    - The first run downloads 6.0 GB.
    - `--max-num-batched-tokens 2048` reads long text in smaller steps, so the
      model needs less memory. Without it this model does not fit on 16 GB.

=== "32 GB"

    Qwen3.5-27B, a larger model, with room for a conversation of about 24,000
    words.

    ```bash
    vllm serve mlx-community/Qwen3.5-27B-4bit \
      --host 127.0.0.1 --max-model-len 32768 --max-num-batched-tokens 1024 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3
    ```

    - The first run downloads 16 GB.
    - It is slow to read long text. For faster replies, use the 16 GB recipe.
    - `--max-num-batched-tokens 1024` reads long text in smaller steps, so the
      model needs less memory. Without it this model does not fit on 32 GB.

=== "64 GB"

    Qwen3.5-35B-A3B, with room for many conversations of about 24,000 words
    at once.

    ```bash
    vllm serve mlx-community/Qwen3.5-35B-A3B-4bit \
      --host 127.0.0.1 --max-model-len 32768 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3
    ```

    - The first run downloads 20 GB.
    - It is the fastest of the three, because only part of the model runs for
      each word.

What the options do:

- `--host 127.0.0.1` keeps the server to your Mac. See
  [Share with other computers](#share-with-other-computers) to change that.
- `--max-model-len` is the longest conversation in tokens: your messages and
  the replies together.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder` lets the model
  call tools, which coding agents and other apps need.
- `--reasoning-parser qwen3` puts the model's thinking in a separate field, so
  replies hold only the answer.

The server is ready when Terminal shows `Application startup complete`. Leave
that window open.

## Check that it works

Open a second Terminal window with Command-N and paste:

=== "16 GB"

    ```bash
    curl http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "mlx-community/Qwen3.5-9B-MLX-4bit",
           "messages": [{"role": "user", "content": "What is unified memory?"}]}'
    ```

=== "32 GB"

    ```bash
    curl http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "mlx-community/Qwen3.5-27B-4bit",
           "messages": [{"role": "user", "content": "What is unified memory?"}]}'
    ```

=== "64 GB"

    ```bash
    curl http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "mlx-community/Qwen3.5-35B-A3B-4bit",
           "messages": [{"role": "user", "content": "What is unified memory?"}]}'
    ```

The answer is the text after `"content"`. Qwen3.5 thinks before it answers,
which can take a while. To skip the thinking for a faster reply, add
`"chat_template_kwargs": {"enable_thinking": false}` to the request.

## Use it from an app or agent

Apps and coding agents that let you set an OpenAI API address, often called
the base URL, can use the server. Set it to `http://127.0.0.1:8000/v1` and use
the model name from your recipe. If the app asks for an API key, type any
text.

## Share with other computers

To reach the server from other computers on your network, replace
`--host 127.0.0.1` with `--host 0.0.0.0` and add `--api-key` with a secret of
your choice, for example `--api-key my-secret`. Apps then need that key.
Without it, anyone on your network can use the server.

## Stop the server

Click the Terminal window running the server and press Control-C. Run the
same command to start it again. The model is already downloaded.

## If it does not start

Read the error just above the last line in Terminal.

- **"command not found: vllm"**: run the `source` command from
  [Before you start](#before-you-start) in this window.
- **"To serve at least one request with the model's max seq len"**: the
  conversation length does not fit. Replace the number after
  `--max-model-len` with the smaller one the message suggests.
- **"not enough Metal memory"**: the model does not fit in the memory free
  right now. Quit other large apps and try again. Halving the number after
  `--max-num-batched-tokens` also helps, or use a smaller model.
- **It fits but other apps slow down**: add `--gpu-memory-utilization 0.8` to
  give vLLM a little less memory. The default is 0.92.
- **It stops for about 10 minutes after "init engine", then carries on**:
  vLLM is checking the model on Hugging Face and the connection is slow.
  Once the model is downloaded, start with `HF_HUB_OFFLINE=1` in front of the
  command, for example `HF_HUB_OFFLINE=1 vllm serve ...`.

## Or with llmman

[llmman](https://github.com/llmmanorg/llmman) can start the same models. On a
Mac it uses MLX's own server by default. To use vLLM Metal, run the `source`
step from [Before you start](#before-you-start), then, for the 16 GB recipe:

```bash
LLMMAN_SAFETENSORS_ENGINE=vllm LLMMAN_CONTEXT_LENGTH=65536 \
  LLMMAN_VLLM_ARGS="--max-num-batched-tokens 2048" \
  llmman run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

`LLMMAN_CONTEXT_LENGTH` is the same as `--max-model-len` above, and
`LLMMAN_VLLM_ARGS` passes other options to vLLM. llmman keeps its own copy of
the model, so the first run downloads it again.

## Reuse long prompts

KV offloading helps when the same long text comes back, such as a long chat
or one document asked about many times. vLLM reloads the work it already did
on that text instead of redoing it. It works with Qwen3 models such as
`mlx-community/Qwen3-8B-4bit`, not with the Qwen3.5 models above. It needs a
vLLM Metal release newer than v0.30.0 or the install script's development
build. See [KV Cache Offloading](kv_offloading.md).

## How these were tested

You can skip this section.

Each command was run as written with vLLM 0.31 and vLLM Metal main (October
2026), on a 128 GB M3 Max and a 128 GB M4 Max. `--gpu-memory-utilization` was
added to limit vLLM to the memory a smaller Mac gives its graphics chip, taken
as 2/3 of the Mac's memory. Measured Macs gave between 67% and 84%. Both Macs
gave the same results:

| Recipe | KV cache | Full-length conversations at once | 14,000-token prompt (M4 Max) |
|---|---|---|---|
| 16 GB | 69,037 tokens | 1 | 19 s |
| 32 GB | 47,786 tokens | 1.5 | 65 s |
| 64 GB | 716,584 tokens | 22 | 12 s |

The llmman command was run on the M4 Max with llmman 0.1.535. Tool calls were
checked with a request that defines one function.

If a recipe does not work on your Mac, please open an issue.
