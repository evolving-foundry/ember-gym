#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
python="${1:-${root}/.venv/bin/python}"

if [[ ! -x "$python" ]]; then
  echo "overlay test interpreter is not executable: $python" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to install overlay test dependencies" >&2
  exit 1
fi

uv pip install --python "$python" -r "${root}/scripts/refactor/overlay-test-requirements.txt"

# tau2's metadata still pins litellm<1.82.7; harbor requires >=1.92. The import and
# test suite are verified against 1.92, and production isolates components in their
# own venvs, so install the tau2 package without re-resolving the shared env.
uv pip install --python "$python" --no-deps "tau2[knowledge] @ git+https://github.com/bxyu-nvidia/tau2-bench@bxyu/nemo_gym_stable"
