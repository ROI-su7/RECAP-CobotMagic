#!/usr/bin/env bash
# Read-only Piper/ROS diagnostic snapshot. This script never publishes a ROS
# topic, changes a CAN interface, or starts/stops a process.
set -u

log_root="${1:-$HOME/piper_runtime_logs}"
timestamp="$(date +%Y%m%d_%H%M%S)"
snapshot_dir="$log_root/$timestamp"
mkdir -p "$snapshot_dir/ros_logs"

summary="$snapshot_dir/summary.txt"
exec > >(tee "$summary") 2>&1

echo "Piper diagnostic snapshot: $snapshot_dir"
date '+time=%F %T %Z'
echo "host=$(hostname)"
echo "user=$(id -un)"

echo
echo "===== CAN interfaces ====="
ip -details -statistics link show type can || true

echo
echo "===== CAN USB topology and persistent counters ====="
# Keep the physical USB path beside every runtime trace.  This is deliberately
# read-only and makes a follower adapter/port fault distinguishable from the
# independent leader 0x470 role handshake on later runs.
for net_path in /sys/class/net/can*; do
  [[ -e "$net_path" ]] || continue
  iface="${net_path##*/}"
  device_path="$(readlink -f "$net_path/device" 2>/dev/null || true)"
  printf '%s device=%s\n' "$iface" "${device_path:-unknown}"
  printf '  '
  for counter in rx_bytes rx_packets rx_errors rx_dropped tx_bytes tx_packets tx_errors tx_dropped; do
    value="$(cat "$net_path/statistics/$counter" 2>/dev/null || printf '?')"
    printf '%s=%s ' "$counter" "$value"
  done
  printf '\n'
done

echo
echo "===== Relevant processes ====="
ps -eo pid,lstart,stat,cmd --sort=start_time \
  | grep -E 'roscore|rosmaster|roslaunch|piper|serve_pi05|examples.piper_real|openpi|new_collect_data' \
  | grep -v -E 'grep|piper_log_snapshot' || true

echo
echo "===== ROS graph ====="
if timeout 3 rosnode list >"$snapshot_dir/rosnodes.txt" 2>&1; then
  cat "$snapshot_dir/rosnodes.txt"
  rostopic list >"$snapshot_dir/rostopics.txt" 2>&1 || true
  for topic in \
    /enable_flag \
    /teleop/follower_enable \
    /teleop/mode \
    /teleop/takeover_phase \
    /teleop/hold_active \
    /teleop/policy_hold_ack \
    /teleop/leader_tx_enable \
    /teleop/leader/tx_quiet_left \
    /teleop/leader/tx_quiet_right \
    /teleop/emergency_stop \
    /teleop/inference_active \
    /teleop/policy/joint_left \
    /teleop/policy/joint_right \
    /teleop/leader/joint_left \
    /teleop/leader/joint_right \
    /teleop/command/joint_left \
    /teleop/command/joint_right \
    /puppet/joint_left \
    /puppet/joint_right; do
    echo "--- $topic ---" >>"$snapshot_dir/topic_info.txt"
    timeout 3 rostopic info "$topic" >>"$snapshot_dir/topic_info.txt" 2>&1 || true
  done
  for topic in /enable_flag /teleop/follower_enable /teleop/mode \
               /teleop/takeover_phase /teleop/hold_active \
               /teleop/policy_hold_ack /teleop/leader_tx_enable \
               /teleop/leader/tx_quiet_left /teleop/leader/tx_quiet_right \
               /teleop/emergency_stop /teleop/inference_active; do
    echo "--- $topic ---" >>"$snapshot_dir/topic_values.txt"
    timeout 2 rostopic echo -n 1 "$topic" \
      >>"$snapshot_dir/topic_values.txt" 2>&1 || true
  done
else
  cat "$snapshot_dir/rosnodes.txt"
  echo "ROS master is not reachable; topic inspection skipped."
fi

echo
echo "===== ROS logs ====="
ros_latest="$(readlink -f $HOME/.ros/log/latest 2>/dev/null || true)"
echo "ros_latest=$ros_latest"
if [[ -n "$ros_latest" && -d "$ros_latest" ]]; then
  for pattern in rosout.log master.log 'roslaunch-*.log' 'piper*.log' 'openpi*.log'; do
    find "$ros_latest" -maxdepth 1 -type f -name "$pattern" -exec cp -a {} "$snapshot_dir/ros_logs/" \;
  done
  rg -n -i \
    'ERROR|WARN|MASTER|OUTPUT|0x470|0x191|0x2A1|ctrl_mode|TX_ECHO|CAN_ERROR|ESTOP|emergency|stale|timeout|enable|policy|SEND_MESSAGE_FAILED|bus-off' \
    "$snapshot_dir/ros_logs" >"$snapshot_dir/important_ros_lines.txt" 2>&1 || true
  echo "Copied selected logs from $ros_latest"
else
  echo "No $HOME/.ros/log/latest directory found."
fi

echo
echo "Snapshot complete: $snapshot_dir"
