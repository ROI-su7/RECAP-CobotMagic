#!/usr/bin/env bash
# SigLIP2 and Gemma 3 270M for the RECAP Value Model. Both are gated Google models:
# accept their licences on huggingface.co, then run `huggingface-cli login` (never store the token in a file here).
set -euo pipefail
RECAP_ROOT="${RECAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RLINF="$RECAP_ROOT/openpi/RLinf"
CLI="$RLINF/.venv-recap/bin/huggingface-cli"
if ! env -u PYTHONPATH "$CLI" whoami >/dev/null 2>&1; then
  echo "[RECAP] Hugging Face login not found; skipping the gated models. Accept the licences of"
  echo "[RECAP] google/siglip2-so400m-patch14-224 and google/gemma-3-270m, run:"
  echo "[RECAP]   $CLI login"
  echo "[RECAP] then: bash env/install_all.sh --only 06"
  exit 0
fi
env -u PYTHONPATH "$CLI" download google/siglip2-so400m-patch14-224 --local-dir "$RLINF/recap_artifacts/siglip2-so400m-patch14-224"
env -u PYTHONPATH "$CLI" download google/gemma-3-270m --local-dir "$RLINF/recap_artifacts/gemma-3-270m"
