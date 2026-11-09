#!/usr/bin/env bash
# Server-side paths for Piper RECAP; sourced by the recap_round_*.sbatch jobs.
# Mirrors examples/recap_piper/env.sh, whose defaults point at the robot host.
REPO_DIR="${REPO_DIR:-$HOME/RECAP/openpi/RLinf}"
export RECAP_NATIVE_OPENPI="${RECAP_NATIVE_OPENPI:-$HOME/RECAP/openpi}"
export RECAP_PYTHON="${RECAP_PYTHON:-$REPO_DIR/.venv-recap/bin/python}"
export RECAP_ROUND="${RECAP_ROUND:-round_001}"
export RECAP_RUN_DIR="${RECAP_RUN_DIR:-$REPO_DIR/recap_runs/$RECAP_ROUND}"
export RECAP_DATASET="${RECAP_DATASET:-$REPO_DIR/recap_data/lerobot_v1}"
export RECAP_SFT_DATASET="${RECAP_SFT_DATASET:-$REPO_DIR/recap_data/sft_all_wire_data}"
source "$REPO_DIR/examples/recap_piper/env.sh"
export PYTHONUNBUFFERED=1
# The cluster has no system FFmpeg 4-7 for torchcodec 0.2 (PyAV bundles its own).
export RECAP_VIDEO_BACKEND="${RECAP_VIDEO_BACKEND:-pyav}"

# Demonstrations (type sft, all positive) + rollouts (C/X, takeovers), the
# same four-stage data overrides as README_PIPER_RECAP.md section 8.
# RECAP_ROLLOUT_DATASETS lists every rollout snapshot of the round (space
# separated, e.g. round 1 + round 2); it defaults to RECAP_DATASET alone.
read -r -a RECAP_ROLLOUTS <<<"${RECAP_ROLLOUT_DATASETS:-$RECAP_DATASET}"
_recap_paths() {  # <extra keys for every entry>
  local entries="{dataset_path:$RECAP_SFT_DATASET,type:sft$1}" path
  for path in "${RECAP_ROLLOUTS[@]}"; do
    entries+=",{dataset_path:$path,type:rollout$1}"
  done
  echo "data.train_data_paths=[$entries]"
}
export RET_DATA="$(_recap_paths "")"
export VAL_DATA="$(_recap_paths ",weight:1.0,robot_type:piper,model_type:pi05")"
export ADV_DATA="$(_recap_paths ",weight:1.0,robot_type:piper")"
export CFG_DATA="$(_recap_paths ",weight:1.0")"
