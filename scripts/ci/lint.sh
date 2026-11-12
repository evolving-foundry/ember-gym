#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx is required to run the pinned Ruff check" >&2
  exit 1
fi

# overlay/ contains forked upstream components with intentional import ordering
# (for example tau2 must set TAU2_DATA_DIR before importing tau2). Lint only the
# AgentGym-owned runtime and release tooling here.
uvx ruff@0.16.8 check agent_gym eval_service tests scripts/refactor
