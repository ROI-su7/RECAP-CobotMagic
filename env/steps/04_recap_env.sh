#!/usr/bin/env bash
# openpi/RLinf/.venv-recap: native JAX pi0.5 server, RECAP training/serving, robot client and recorder.
set -euo pipefail
RECAP_ROOT="${RECAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PATH="$HOME/.local/bin:$PATH"
RLINF="$RECAP_ROOT/openpi/RLinf"
cd "$RLINF"
if [[ ! -x .venv-recap/bin/python ]]; then
  uv venv --python 3.11 --seed .venv-recap
fi
# Exact versions of the reference machine; --no-deps reproduces them as-is.
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv-recap/bin/python --no-deps -r "$RECAP_ROOT/env/requirements-recap.txt"
uv pip install --python .venv-recap/bin/python --no-deps -e "$RLINF"
# openpi's PyTorch models need its patched transformers files (SigLIP/Gemma/PaliGemma mixed precision).
SITE="$RLINF/.venv-recap/lib/python3.11/site-packages"
cp -r "$SITE/openpi/models_pytorch/transformers_replace/"* "$SITE/transformers/"
env -u PYTHONPATH .venv-recap/bin/python - <<'PY'
import jax, torch, ray, openpi, rlinf
print("[RECAP] env ok: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| jax", jax.__version__, jax.devices()[0].platform, "| ray", ray.__version__)
PY
