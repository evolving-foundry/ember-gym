#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Install nemo-gym first. Its console scripts are intentionally shadowed by the
# AgentGym compatibility launcher in the second step, which is what keeps every
# legacy command name routing through one resolver.
uv sync --no-install-project --extra dev --extra sandbox
uv pip install --no-deps -e .
uv run --no-sync agent-gym doctor --entrypoints
