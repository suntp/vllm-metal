#!/bin/bash
# Benchmark environment for one vllm-metal source checkout.
#
# Wraps the repository's own install.sh (per-checkout venv, pinned vllm
# wheel, editable install, native Metal build) and adds the test runner
# the harness needs.  Safe to re-run; install.sh is idempotent per tree.
# Works on any Apple Silicon Mac; nothing here is machine-specific.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

if [[ ! -f "$repo_root/install.sh" ]]; then
    echo "error: $repo_root/install.sh not found" >&2
    exit 1
fi

"$repo_root/install.sh"

venv_python="$repo_root/.venv-vllm-metal/bin/python"
if [[ ! -x "$venv_python" ]]; then
    echo "error: expected install.sh to create $venv_python" >&2
    exit 1
fi

uv pip install --python "$venv_python" pytest

echo "Benchmark-ready:"
echo "  python: $venv_python"
echo "  harness: python -m tools.benchmark.macos.run_ab --help  (from $repo_root)"
