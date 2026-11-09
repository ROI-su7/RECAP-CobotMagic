#!/usr/bin/env bash
# System packages for Cobot Magic (Ubuntu 20.04). Requires sudo.
set -euo pipefail
RECAP_ROOT="${RECAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs curl tmux ffmpeg can-utils ethtool net-tools \
  libgflags-dev libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev libkdl-parser-dev \
  python3-pip python3-dev
# Piper ROS nodes and the demonstration recorder run under the system Python 3.8.
/usr/bin/python3 -m pip install --user "piper_sdk==0.6.1" "python-can==4.5.0" \
  -r "$RECAP_ROOT/cobot_magic/collect_data/requirements.txt"
# uv manages the two Python 3.11 environments.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
