#!/usr/bin/env bash
# ROS Noetic + catkin workspaces: Piper (four-CAN mux, takeover) and Astra cameras.
set -euo pipefail
RECAP_ROOT="${RECAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y ros-noetic-desktop-full
fi
sudo apt-get install -y python3-catkin-tools python3-rosdep \
  ros-noetic-image-geometry ros-noetic-camera-info-manager ros-noetic-image-transport \
  ros-noetic-image-publisher ros-noetic-libuvc-ros
set +u; source /opt/ros/noetic/setup.bash; set -u
(cd "$RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic" && catkin_make)
(cd "$RECAP_ROOT/cobot_magic/camera_ws" && catkin_make)
echo "[RECAP] Now set your CAN ports and camera serials: docs/FOUR_CAN.md section 1"
