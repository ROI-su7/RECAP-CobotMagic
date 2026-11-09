#!/usr/bin/env bash
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/env.sh"
cd "$RECAP_REPO_DIR"
STAGE="${1:-check}"
shift || true
CONFIG_DIR="$RECAP_REPO_DIR/examples/offline_rl/config"
case "$STAGE" in
  check)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.preflight "$@"
    ;;
  launch)
    # Piper ROS drivers + mux. followers MUST use the fixed lingbot_piper mux:
    # the factory follower mux disabled (dropped) both
    # followers on Space/H/I and is not shipped. takeover is the four-arm
    # launch whose four-state mux holds with torque on.
    PROFILE="${1:-}"
    shift || true
    source /opt/ros/noetic/setup.bash
    source $RECAP_COBOT_DIR/Piper_ros_private-ros-noetic/devel/setup.bash
    export PYTHONPATH="$RECAP_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    case "$PROFILE" in
      followers)
        "$RECAP_PYTHON" -m rlinf.recap_piper.mux_guard --before-launch \
          --package "$RECAP_FOLLOWER_MUX_ROS"
        export ROS_PACKAGE_PATH="$RECAP_FOLLOWER_MUX_ROS:$ROS_PACKAGE_PATH"
        FOUND="$(rospack find lingbot_piper)"
        if [[ "$(realpath "$FOUND")" != "$(realpath "$RECAP_FOLLOWER_MUX_ROS")" ]]; then
          echo "REFUSED: lingbot_piper resolves to $FOUND, not $RECAP_FOLLOWER_MUX_ROS" >&2
          exit 1
        fi
        exec roslaunch lingbot_piper start_inference_followers.launch \
          puppet_left_can:=can_l_follower puppet_right_can:=can_r_follower \
          publish_hz:=200 "$@"
        ;;
      takeover)
        "$RECAP_PYTHON" -m rlinf.recap_piper.mux_guard --before-launch
        exec roslaunch piper start_inference_takeover.launch \
          master_left_can:=can_l_leader master_right_can:=can_r_leader \
          puppet_left_can:=can_l_follower puppet_right_can:=can_r_follower \
          publish_hz:=200 "$@"
        ;;
      *)
        echo "Usage: bash examples/recap_piper/run.sh launch {followers|takeover} [roslaunch args...]" >&2
        exit 2
        ;;
    esac
    ;;
  collect)
    source /opt/ros/noetic/setup.bash
    source $RECAP_COBOT_DIR/Piper_ros_private-ros-noetic/devel/setup.bash
    export PYTHONPATH="$RECAP_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.collect \
      --output "$RECAP_RAW_DIR" --task "$RECAP_TASK" "$@"
    ;;
  export)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli export \
      --raw "$RECAP_RAW_DIR" --output "$RECAP_DATASET" "$@"
    ;;
  score)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli score --raw "$RECAP_RAW_DIR" "$@"
    ;;
  export-interventions)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli export \
      --raw "$RECAP_RAW_DIR" --output "$RECAP_INT_DATASET" --interventions-only "$@"
    ;;
  convert)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli convert \
      --source "$RECAP_PI05_JAX_DIR" --output "$RECAP_PI05_TORCH_DIR" \
      --config-file "$RECAP_SOURCE_CONFIG_FILE" --config-name "$RECAP_SOURCE_CONFIG" "$@"
    ;;
  returns)
    ENTRY="examples/offline_rl/advantage_labeling/recap/process/compute_returns.py"
    CONFIG=piper_recap_returns
    ;;
  value)
    ENTRY="examples/offline_rl/advantage_labeling/recap/train_value.py"
    CONFIG=piper_recap_value
    ;;
  advantages)
    ENTRY="examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py"
    CONFIG=piper_recap_advantages
    ;;
  cfg)
    ENTRY="examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
    CONFIG=piper_recap_cfg
    ;;
  sftint)
    ENTRY="examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
    CONFIG=piper_recap_sft_int
    ;;
  serve)
    "$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage serve --gpu-idle
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli serve "$@"
    ;;
  *)
    echo "Usage: bash examples/recap_piper/run.sh {check|launch|collect|score|export|export-interventions|convert|returns|value|advantages|cfg|sftint|serve} [args...]" >&2
    exit 2
    ;;
esac
CMD=("$RECAP_PYTHON" "$ENTRY" --config-path "$CONFIG_DIR" --config-name "$CONFIG")
if [[ "$STAGE" == value || "$STAGE" == cfg || "$STAGE" == sftint ]]; then
  CMD+=("runner.logger.log_path=$RECAP_RUN_DIR/$STAGE")
fi
# --dry-run composes/resolves the exact config without initializing workers/GPU.
if [[ "${1:-}" == --dry-run ]]; then
  shift
  # Keep Hydra overrides contiguous; argparse rejects positionals after options.
  exec "${CMD[@]}" "$@" --cfg job --resolve
fi
"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage "$STAGE"
if [[ "$STAGE" != returns ]]; then
  "$RECAP_PYTHON" -m rlinf.recap_piper.preflight --gpu-idle
fi
if [[ "$STAGE" == value || "$STAGE" == cfg || "$STAGE" == sftint ]]; then
  if ! "$(dirname "$RECAP_PYTHON")/ray" status --address="$RECAP_RAY_ADDRESS" >/dev/null 2>&1; then
    echo "RECAP Ray head is not running at $RECAP_RAY_ADDRESS. See README_PIPER_RECAP.md." >&2
    exit 1
  fi
fi
mkdir -p "$RECAP_RUN_DIR/$STAGE"
"${CMD[@]}" "$@" 2>&1 | tee "$RECAP_RUN_DIR/$STAGE/run.log"
