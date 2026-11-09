#!/usr/bin/env bash
# openpi/.venv: pi0.5 fine-tuning, norm stats and LeRobot conversion (uv.lock pinned).
set -euo pipefail
RECAP_ROOT="${RECAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PATH="$HOME/.local/bin:$PATH"
cd "$RECAP_ROOT/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
env -u PYTHONPATH .venv/bin/python -c "import openpi, jax; print('[RECAP] openpi env ok, jax', jax.__version__)"
