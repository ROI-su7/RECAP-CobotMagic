#!/usr/bin/env bash
# Check an installed tree: Python environments, ROS workspaces and the unit tests. Nothing moves the robot.
#
#   bash env/verify.sh            # environments + ROS + unit tests
#   bash env/verify.sh --no-ros   # skip the ROS checks (training server)
set -uo pipefail
RECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLINF="$RECAP_ROOT/openpi/RLinf"
CHECK_ROS=1
[[ "${1:-}" == "--no-ros" ]] && CHECK_ROS=0
FAIL=0
ok() { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; FAIL=1; }

echo "== Python environments"
if env -u PYTHONPATH "$RECAP_ROOT/openpi/.venv/bin/python" -c "import openpi, jax, lerobot" 2>/dev/null; then
  ok "openpi/.venv (openpi, jax, lerobot)"
else
  bad "openpi/.venv: run env/steps/03_openpi_env.sh"
fi
if env -u PYTHONPATH "$RLINF/.venv-recap/bin/python" - <<'PY'
import jax, torch, ray, hydra, openpi, rlinf
import rlinf.recap_piper.cli, rlinf.recap_piper.policy, rlinf.recap_piper.data
from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker
import filecmp, pathlib, transformers
site = pathlib.Path(openpi.__file__).parent / "models_pytorch" / "transformers_replace" / "models" / "siglip" / "modeling_siglip.py"
assert filecmp.cmp(site, pathlib.Path(transformers.__file__).parent / "models" / "siglip" / "modeling_siglip.py", shallow=False), "transformers is missing the openpi patch"
assert torch.cuda.is_available(), "torch cannot see a GPU"
assert jax.devices()[0].platform == "gpu", "jax cannot see a GPU"
PY
then
  ok "openpi/RLinf/.venv-recap (torch+jax on GPU, RECAP modules)"
else
  bad "openpi/RLinf/.venv-recap: run env/steps/04_recap_env.sh"
fi
for d in siglip2-so400m-patch14-224 gemma-3-270m; do
  if [[ -f "$RLINF/recap_artifacts/$d/config.json" ]]; then ok "recap_artifacts/$d"; else echo "  [todo] recap_artifacts/$d: run env/steps/06_recap_assets.sh after huggingface-cli login"; fi
done

if [[ $CHECK_ROS == 1 ]]; then
  echo "== ROS"
  for ws in cobot_magic/Piper_ros_private-ros-noetic cobot_magic/camera_ws; do
    if [[ -f "$RECAP_ROOT/$ws/devel/setup.bash" ]]; then ok "$ws built"; else bad "$ws not built: run env/steps/02_ros.sh"; fi
  done
  if /usr/bin/python3 -c "import piper_sdk" 2>/dev/null; then ok "system python piper_sdk"; else bad "system python lacks piper_sdk"; fi
  if (set +u; source /opt/ros/noetic/setup.bash; cd "$RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic/src/piper" &&
      /usr/bin/python3 -m unittest discover -s test -p "test_*.py" >/dev/null 2>&1); then
    ok "Piper ROS takeover/mux tests"
  else
    bad "Piper ROS takeover/mux tests (run: cd cobot_magic/Piper_ros_private-ros-noetic/src/piper && python3 -m unittest discover -s test)"
  fi
  if (set +u; source /opt/ros/noetic/setup.bash; source "$RECAP_ROOT/cobot_magic/camera_ws/devel/setup.bash"
      source "$RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
      cd "$RECAP_ROOT/openpi" && PYTHONPATH="$RLINF:$RECAP_ROOT/openpi:$PYTHONPATH" "$RLINF/.venv-recap/bin/python" -c "import rlinf.recap_piper.client, examples.piper_real.main, rospy") >/dev/null 2>&1; then
    ok "robot client imports (.venv-recap + ROS)"
  else
    bad "robot client imports failed (.venv-recap + ROS)"
  fi
fi

echo "== Unit tests (RECAP)"
cd "$RLINF"
if env -u PYTHONPATH .venv-recap/bin/python -m pytest -q -p no:cacheprovider tests/unit_tests/test_piper_recap.py tests/unit_tests/test_recap_*.py tests/unit_tests/test_openpi_lora_conversion.py tests/unit_tests/test_openpi_native_lora.py; then
  ok "RECAP unit tests"
else
  bad "RECAP unit tests"
fi

[[ $FAIL == 0 ]] && echo "All checks passed." || { echo "Some checks failed."; exit 1; }
