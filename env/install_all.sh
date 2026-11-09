#!/usr/bin/env bash
# One-click environment setup for RECAP on Cobot Magic (Ubuntu 20.04 + ROS Noetic).
#
#   bash env/install_all.sh              # everything
#   bash env/install_all.sh --no-ros     # Python environments only (training server)
#   bash env/install_all.sh --only 04    # run a single step
#
# Steps (env/steps/):
#   01_system_deps.sh   apt packages and the system-Python recorder deps (sudo)
#   02_ros.sh           ROS Noetic + catkin workspaces (Piper, cameras) (sudo for apt)
#   03_openpi_env.sh    openpi/.venv             pi0.5 fine-tuning, norm stats, LeRobot conversion
#   04_recap_env.sh     openpi/RLinf/.venv-recap native JAX pi0.5 server, RECAP training/serving, robot client
#   05_base_assets.sh   public pi0.5 base weights and PaliGemma tokenizer
#   06_recap_assets.sh  SigLIP2 and Gemma 3 for the Value Model (gated: needs `huggingface-cli login`)
# Datasets and fine-tuned weights are not part of this repository.
set -euo pipefail
RECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECAP_ROOT
STEPS=(01_system_deps 02_ros 03_openpi_env 04_recap_env 05_base_assets 06_recap_assets)
SKIP_ROS=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-ros) SKIP_ROS=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
for step in "${STEPS[@]}"; do
  if [[ -n "$ONLY" && "$step" != "$ONLY"* ]]; then continue; fi
  if [[ $SKIP_ROS == 1 && ( "$step" == 01_system_deps || "$step" == 02_ros ) ]]; then
    echo "[RECAP] skip $step (--no-ros)"; continue
  fi
  echo "[RECAP] ===== $step ====="
  bash "$RECAP_ROOT/env/steps/$step.sh"
done
echo "[RECAP] Done. Verify with: bash env/verify.sh"
