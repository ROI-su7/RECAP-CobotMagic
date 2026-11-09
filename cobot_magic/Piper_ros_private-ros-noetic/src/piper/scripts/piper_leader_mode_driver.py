#!/usr/bin/env python3
"""Piper leader driver used by the four-CAN inference/takeover mux.

Supervised inference uses the leaders as 0xFC motion-output arms.  TAKEOVER
uses the official leader/follower role command (0x470 byte 0 = 0xFA) to enter
the leader zero-force-drag mode, then forwards its live 0x155-0x159 stream via
ROS.  Returning to inference writes 0xFC and repeats the measured current pose
to cover the documented v1.8-5--v1.8-8 Leader->Follower firmware workaround.
"""

import errno
import math
import socket
import struct
import threading
import time
from collections import deque

import rospy
from piper_can_tx_guard import CanLinkWatch, CanTxGuard, reset_command
from piper_sdk import C_PiperInterface_V2
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String


RAD_TO_MILLI_DEG = 180000.0 / math.pi
MILLI_DEG_TO_RAD = math.pi / 180000.0
CAN_FRAME_FORMAT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)
CAN_ID_MASK = 0x1FFFFFFF
CAN_ERR_FLAG = 0x20000000
CAN_ERR_MASK = 0x1FFFFFFF
CAN_RAW_ERR_FILTER = 2
ROLE_CONFIG_CAN_ID = 0x470
MOTION_CTRL_2_CAN_ID = 0x151
RESTORE_MASTER_SLAVE_CAN_ID = 0x191
ARM_STATUS_CAN_ID = 0x2A1
MASTER_CONTROL_OFFSETS = (0x00, 0x10, 0x20)
MASTER_JOINT_SUFFIXES = (0x05, 0x06, 0x07)
MASTER_GRIPPER_SUFFIX = 0x09


class RawMasterObserver:
    """Receive Piper teaching-input frames without ever transmitting CAN.

    The SDK rate accessor has returned ``0 Hz`` on this installation even
    when the proven fixed-MASTER SocketCAN reader can decode the same arm.
    Capture is enabled only while OUTPUT commands have stopped for a role
    transition, so local 0x155--0x157 JointCtrl transmissions cannot be
    mistaken for teaching input.
    """

    def __init__(self, can_port):
        self.can_port = can_port
        self._lock = threading.Lock()
        self._capture_enabled = False
        self._running = True
        self._positions = {
            offset: [0.0] * 7 for offset in MASTER_CONTROL_OFFSETS
        }
        self._counts = {offset: {} for offset in MASTER_CONTROL_OFFSETS}
        self._group_count = {offset: 0 for offset in MASTER_CONTROL_OFFSETS}
        self._group_times = {
            offset: deque(maxlen=64) for offset in MASTER_CONTROL_OFFSETS
        }
        self._gripper_times = {
            offset: deque(maxlen=64) for offset in MASTER_CONTROL_OFFSETS
        }
        self._active_offset = None
        self._last_pose = None
        self._last_group_at = 0.0
        self._last_gripper_at = 0.0
        self._capture_started_at = 0.0
        self._role_config_tx = deque(maxlen=32)
        self._motion_ctrl_2_tx = deque(maxlen=32)
        self._restore_master_slave_tx = deque(maxlen=32)
        self._status_segments = deque(maxlen=64)
        self._can_error_frames = deque(maxlen=32)
        self._socket = self._open_socket()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="dynamic-master-observer-{}".format(can_port),
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _joint_ids(offset):
        return tuple(
            0x150 + offset + suffix for suffix in MASTER_JOINT_SUFFIXES
        )

    @staticmethod
    def _gripper_id(offset):
        return 0x150 + offset + MASTER_GRIPPER_SUFFIX

    def start_capture(self, seed_pose):
        with self._lock:
            self._capture_enabled = True
            self._positions = {
                offset: list(seed_pose) for offset in MASTER_CONTROL_OFFSETS
            }
            self._counts = {offset: {} for offset in MASTER_CONTROL_OFFSETS}
            self._group_count = {
                offset: 0 for offset in MASTER_CONTROL_OFFSETS
            }
            self._group_times = {
                offset: deque(maxlen=64) for offset in MASTER_CONTROL_OFFSETS
            }
            self._gripper_times = {
                offset: deque(maxlen=64) for offset in MASTER_CONTROL_OFFSETS
            }
            self._active_offset = None
            self._last_pose = None
            self._last_group_at = 0.0
            self._last_gripper_at = 0.0
            self._capture_started_at = time.monotonic()
            self._role_config_tx = deque(maxlen=32)
            self._motion_ctrl_2_tx = deque(maxlen=32)
            self._restore_master_slave_tx = deque(maxlen=32)
            self._status_segments = deque(maxlen=64)
            self._can_error_frames = deque(maxlen=32)

    def stop_capture(self):
        with self._lock:
            self._capture_enabled = False

    def _open_socket(self):
        sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sock.settimeout(0.1)
        # Receive SocketCAN error frames during the role hand-off.  A 0x470
        # frame observed by this second socket is the kernel/driver TX echo of
        # the SDK socket's write; combining that echo with the absence of CAN
        # errors and the subsequent 0x2A1 sequence separates a host/adapter TX
        # problem from a controller that did not apply 0xFA.
        try:
            sock.setsockopt(
                getattr(socket, "SOL_CAN_RAW", 101),
                getattr(socket, "CAN_RAW_ERR_FILTER", CAN_RAW_ERR_FILTER),
                struct.pack("=I", CAN_ERR_MASK),
            )
        except OSError as exc:
            rospy.logwarn(
                "%s could not enable raw CAN error-frame capture: %s",
                self.can_port,
                exc,
            )
        sock.bind((self.can_port,))
        return sock

    def _read_loop(self):
        while self._running and not rospy.is_shutdown():
            try:
                frame = self._socket.recv(CAN_FRAME_SIZE)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._running:
                    return
                # A down/up (CAN TX stall reset) reports ENETDOWN once and the
                # socket keeps receiving afterwards. Any other error means the
                # netdev was replaced (can_config.sh rebind): bind anew.
                if exc.errno != errno.ENETDOWN:
                    rospy.logwarn_throttle(
                        5.0,
                        "%s raw MASTER receive failed: %s; reopening",
                        self.can_port,
                        exc,
                    )
                    time.sleep(0.5)
                    self._reopen_socket()
                continue
            if len(frame) != CAN_FRAME_SIZE:
                continue
            can_id, dlc, data = struct.unpack(CAN_FRAME_FORMAT, frame)
            self._process_frame(can_id, bytes(data[:dlc]), time.monotonic())

    def _process_frame(self, raw_can_id, data, timestamp):
        with self._lock:
            if not self._capture_enabled:
                return
            if raw_can_id & CAN_ERR_FLAG:
                self._can_error_frames.append(
                    (timestamp, raw_can_id & CAN_ERR_MASK, bytes(data))
                )
                return
            can_id = raw_can_id & CAN_ID_MASK
            if can_id == ROLE_CONFIG_CAN_ID:
                self._role_config_tx.append((timestamp, bytes(data)))
                return
            if can_id == MOTION_CTRL_2_CAN_ID:
                self._motion_ctrl_2_tx.append((timestamp, bytes(data)))
                return
            if can_id == RESTORE_MASTER_SLAVE_CAN_ID:
                self._restore_master_slave_tx.append((timestamp, bytes(data)))
                return
            if can_id == ARM_STATUS_CAN_ID and len(data) >= 8:
                status = (
                    data[0],
                    data[1],
                    data[2],
                    data[3],
                    data[4],
                    int.from_bytes(data[6:8], byteorder="big", signed=False),
                )
                if self._status_segments and self._status_segments[-1][3] == status:
                    segment = self._status_segments[-1]
                    segment[1] = timestamp
                    segment[2] += 1
                else:
                    # [first timestamp, last timestamp, sample count, fields]
                    self._status_segments.append([timestamp, timestamp, 1, status])
                return
            for offset in MASTER_CONTROL_OFFSETS:
                joint_ids = self._joint_ids(offset)
                if can_id in joint_ids and len(data) >= 8:
                    first, second = struct.unpack(">ii", data[:8])
                    pair = joint_ids.index(can_id) * 2
                    self._positions[offset][pair] = first * MILLI_DEG_TO_RAD
                    self._positions[offset][pair + 1] = second * MILLI_DEG_TO_RAD
                    counts = self._counts[offset]
                    counts[can_id] = counts.get(can_id, 0) + 1
                    if all(frame_id in counts for frame_id in joint_ids):
                        complete_count = min(counts[frame_id] for frame_id in joint_ids)
                        if complete_count > self._group_count[offset]:
                            self._group_count[offset] = complete_count
                            self._group_times[offset].append(timestamp)
                            self._active_offset = offset
                            self._last_group_at = timestamp
                            self._last_pose = list(self._positions[offset])
                    return
                if can_id == self._gripper_id(offset) and len(data) >= 4:
                    gripper = struct.unpack(">i", data[:4])[0] / 1000000.0
                    self._positions[offset][6] = max(0.0, min(0.08, gripper))
                    self._gripper_times[offset].append(timestamp)
                    if self._active_offset == offset:
                        self._last_gripper_at = timestamp
                        if self._last_pose is not None:
                            self._last_pose[6] = self._positions[offset][6]
                    return

    @staticmethod
    def _rate(samples):
        if len(samples) < 2:
            return 0.0
        elapsed = samples[-1] - samples[0]
        return 0.0 if elapsed <= 0.0 else (len(samples) - 1) / elapsed

    def snapshot(self):
        with self._lock:
            if not self._capture_enabled or self._active_offset is None:
                return (0.0, 0.0, 0.0, 0.0)
            offset = self._active_offset
            return (
                self._last_group_at,
                self._last_gripper_at,
                self._rate(self._group_times[offset]),
                self._rate(self._gripper_times[offset]),
            )

    def pose(self):
        with self._lock:
            return None if self._last_pose is None else list(self._last_pose)

    def active_offset(self):
        with self._lock:
            return self._active_offset

    def rate_text(self):
        with self._lock:
            elapsed = max(0.001, time.monotonic() - self._capture_started_at)
            parts = []
            for offset in MASTER_CONTROL_OFFSETS:
                rates = [
                    self._counts[offset].get(frame_id, 0) / elapsed
                    for frame_id in self._joint_ids(offset)
                ]
                gripper_rate = len(self._gripper_times[offset]) / elapsed
                parts.append(
                    "0x{:03X}-0x{:03X}={:.1f}/{:.1f}/{:.1f}Hz "
                    "0x{:03X}={:.1f}Hz".format(
                        0x155 + offset,
                        0x157 + offset,
                        rates[0],
                        rates[1],
                        rates[2],
                        0x159 + offset,
                        gripper_rate,
                    )
                )
            return "; ".join(parts)

    @staticmethod
    def _relative_time(timestamp, since):
        return "+{:.3f}s".format(max(0.0, timestamp - since))

    def transition_marker(self):
        """Reset only handshake diagnostics and mark one 0xFA attempt."""
        with self._lock:
            self._role_config_tx.clear()
            self._motion_ctrl_2_tx.clear()
            self._restore_master_slave_tx.clear()
            self._status_segments.clear()
            self._can_error_frames.clear()
            return time.monotonic()

    def role_config_echo_seen(self, since, linkage_config=0xFA):
        """Whether a matching 0x470 TX echo appeared after ``since``."""
        with self._lock:
            return any(
                timestamp >= since
                and data
                and data[0] == linkage_config
                for timestamp, data in self._role_config_tx
            )

    def standby_echo_seen(self, since):
        """Whether MotionCtrl_2(STANDBY) was echoed after ``since``."""
        with self._lock:
            return any(
                timestamp >= since
                and len(data) >= 2
                and data[0] == 0x00
                and data[1] == 0x01
                for timestamp, data in self._motion_ctrl_2_tx
            )

    def restore_master_slave_echo_seen(self, since):
        """Whether 0x191 restore-master/slave mode=0 was echoed."""
        with self._lock:
            return any(
                timestamp >= since
                and len(data) >= 2
                and data[0] == 0x01
                and data[1] == 0x00
                for timestamp, data in self._restore_master_slave_tx
            )

    def ctrl_modes_since(self, since):
        """Return ordered, de-duplicated raw 0x2A1 ctrl_mode values."""
        with self._lock:
            modes = []
            for first, last, _count, status in self._status_segments:
                if last < since:
                    continue
                mode = status[0]
                if not modes or modes[-1] != mode:
                    modes.append(mode)
            return modes

    def can_errors_seen(self, since):
        with self._lock:
            return any(timestamp >= since for timestamp, _mask, _data in self._can_error_frames)

    def transition_report(self, since):
        """Compact bus-level 0x470/0x151/0x2A1 timeline for ROS logs."""
        with self._lock:
            now = time.monotonic()
            role_frames = [
                "{}:{}".format(
                    self._relative_time(timestamp, since), data.hex().upper()
                )
                for timestamp, data in self._role_config_tx
                if timestamp >= since
            ]
            standby_frames = [
                "{}:{}".format(
                    self._relative_time(timestamp, since), data.hex().upper()
                )
                for timestamp, data in self._motion_ctrl_2_tx
                if timestamp >= since
                and len(data) >= 2
                and data[0] == 0x00
                and data[1] == 0x01
            ]
            restore_frames = [
                "{}:{}".format(
                    self._relative_time(timestamp, since), data.hex().upper()
                )
                for timestamp, data in self._restore_master_slave_tx
                if timestamp >= since
                and len(data) >= 2
                and data[0] == 0x01
                and data[1] == 0x00
            ]
            status_parts = []
            for first, last, count, status in self._status_segments:
                if last < since:
                    continue
                ctrl_mode, arm_status, mode_feed, teach_status, motion_status, err_code = status
                status_parts.append(
                    "{}..{} x{} ctrl=0x{:02X} arm=0x{:02X} move=0x{:02X} "
                    "teach=0x{:02X} motion=0x{:02X} err=0x{:04X}".format(
                        self._relative_time(max(first, since), since),
                        self._relative_time(last, since),
                        count,
                        ctrl_mode,
                        arm_status,
                        mode_feed,
                        teach_status,
                        motion_status,
                        err_code,
                    )
                )
            error_parts = [
                "{}:class=0x{:08X}:{}".format(
                    self._relative_time(timestamp, since),
                    error_mask,
                    data.hex().upper(),
                )
                for timestamp, error_mask, data in self._can_error_frames
                if timestamp >= since
            ]
            return (
                "window={:.3f}s 0x470_TX_ECHO=[{}] "
                "0x151_STANDBY_TX_ECHO=[{}] "
                "0x191_RESTORE_TX_ECHO=[{}] 0x2A1=[{}] CAN_ERROR=[{}]"
            ).format(
                max(0.0, now - since),
                ", ".join(role_frames) if role_frames else "none",
                ", ".join(standby_frames) if standby_frames else "none",
                ", ".join(restore_frames) if restore_frames else "none",
                "; ".join(status_parts) if status_parts else "none",
                ", ".join(error_parts) if error_parts else "none",
            )

    def _reopen_socket(self):
        try:
            sock = self._open_socket()
        except OSError:
            return  # The netdev is not back yet; retry on the next error.
        old, self._socket = self._socket, sock
        try:
            old.close()
        except OSError:
            pass

    def close(self):
        self._running = False
        try:
            self._socket.close()
        except OSError:
            pass
        self._thread.join(timeout=0.3)


class PiperLeaderModeDriver:
    def __init__(self):
        self.can_port = rospy.get_param("~can_port")
        self.publish_hz = float(rospy.get_param("~publish_hz", 200.0))
        self.role_timeout = float(rospy.get_param("~role_timeout_s", 3.0))
        self.role_settle = float(rospy.get_param("~role_settle_s", 0.25))
        # Firmware S-V1.8-8 supports the SDK's runtime 0xFA/0xFC role command.
        # Before 0xFA, preload and hold the measured OUTPUT pose.  The hold is
        # deliberately short: it prevents a controller-stored target from
        # producing the observed upward kick while allowing firmware that only
        # accepts runtime 0xFA from enabled OUTPUT to enter teaching mode.
        self.master_prepare_hold = float(
            rospy.get_param("~master_prepare_hold_s", 0.10)
        )
        self.role_stream_stable = float(
            rospy.get_param("~role_stream_stable_s", 0.25)
        )
        self.role_stream_min_hz = float(
            rospy.get_param("~role_stream_min_hz", 10.0)
        )
        self.role_stream_freshness = float(
            rospy.get_param("~role_stream_freshness_s", 0.15)
        )
        # Some S-V1.8-8 leaders need several seconds to finish the seamless
        # 0xFA transition: the gripper can become compliant before the six
        # joints start publishing 0x155--0x157.  Do not interrupt that partial
        # transition after only one second with the standby fallback.
        self.master_first_no_stream_retry = float(
            rospy.get_param("~master_first_no_stream_retry_s", 4.0)
        )
        self.master_standby_activation = bool(
            rospy.get_param("~master_standby_activation", True)
        )
        # 0x191 mode=0 is retained as an opt-in diagnostic only. It is not part
        # of the official minimal set_leader_mode() sequence, and the bus traces
        # on this S-V1.8-8 pair showed it was echoed without activating MASTER.
        self.master_restore_linkage_activation = bool(
            rospy.get_param("~master_restore_linkage_activation", False)
        )
        self.role_config_retries = int(rospy.get_param("~role_config_retries", 2))
        self.follower_restore_movej_repeats = int(
            rospy.get_param("~follower_restore_movej_repeats", 20)
        )
        self.follower_restore_movej_period = float(
            rospy.get_param("~follower_restore_movej_period_s", 0.02)
        )
        self.drag_teach_timeout = float(
            rospy.get_param("~drag_teach_timeout_s", 2.0)
        )
        self.drag_teach_status_grace = float(
            rospy.get_param("~drag_teach_status_grace_s", 0.5)
        )
        self.master_enable_fallback = bool(
            rospy.get_param("~master_enable_fallback", True)
        )
        self.startup_role_timeout = float(
            rospy.get_param("~startup_role_timeout_s", 1.5)
        )
        self.move_speed_percent = int(rospy.get_param("~move_speed_percent", 100))
        self.max_joint_velocity = float(
            rospy.get_param("~max_joint_velocity_rad_s", 1.5)
        )
        self.max_gripper_velocity = float(
            rospy.get_param("~max_gripper_velocity_m_s", 0.15)
        )
        self.max_joint_acceleration = float(
            rospy.get_param("~max_joint_acceleration_rad_s2", 0.8)
        )
        self.max_gripper_acceleration = float(
            rospy.get_param("~max_gripper_acceleration_m_s2", 0.3)
        )
        self.alignment_max_joint_velocity = float(
            rospy.get_param("~alignment_max_joint_velocity_rad_s", 0.12)
        )
        self.alignment_max_gripper_velocity = float(
            rospy.get_param("~alignment_max_gripper_velocity_m_s", 0.03)
        )
        self.alignment_max_joint_acceleration = float(
            rospy.get_param("~alignment_max_joint_acceleration_rad_s2", 0.20)
        )
        self.alignment_max_gripper_acceleration = float(
            rospy.get_param("~alignment_max_gripper_acceleration_m_s2", 0.10)
        )
        self.home_max_joint_velocity = float(
            rospy.get_param("~home_max_joint_velocity_rad_s", 0.25)
        )
        self.home_max_gripper_velocity = float(
            rospy.get_param("~home_max_gripper_velocity_m_s", 0.03)
        )
        self.enable_timeout = float(rospy.get_param("~enable_timeout_s", 5.0))
        # HOME publishes /teleop/mode=IDLE and /teleop/home_active=True on two
        # topics. When IDLE arrived first, the IDLE branch disabled a powered
        # leader before home_active was seen; Piper has no brakes, so the arm
        # fell and HOME then re-enabled it at the fallen pose. Keep a powered
        # leader enabled for this grace period before an IDLE disable.
        self.idle_disable_grace = float(
            rospy.get_param("~idle_disable_grace_s", 1.0)
        )
        # MOVE J mode (MotionCtrl_2) is a mode switch, not a per-target frame.
        # Resending it with every 200 Hz target added a fifth frame per command
        # on a bus that already carries ~3150 Piper feedback frames/s. 0 keeps
        # the legacy every-command behavior.
        self.motion_ctrl_period = float(
            rospy.get_param("~motion_ctrl_period_s", 0.1)
        )
        self._last_motion_ctrl = 0.0
        self.enable_retry_period = float(
            rospy.get_param("~enable_retry_period_s", 0.1)
        )
        self.role_pose_tolerance = float(
            rospy.get_param("~role_pose_tolerance_rad", 0.08)
        )
        self.role_gripper_tolerance = float(
            rospy.get_param("~role_gripper_tolerance_m", 0.015)
        )
        self.role_pose_hard_tolerance = float(
            rospy.get_param("~role_pose_hard_tolerance_rad", 0.20)
        )
        self.role_gripper_hard_tolerance = float(
            rospy.get_param("~role_gripper_hard_tolerance_m", 0.030)
        )
        if (
            self.publish_hz <= 0.0
            or self.role_timeout <= 0.0
            or self.master_prepare_hold < 0.0
            or self.role_stream_stable <= 0.0
            or self.role_stream_min_hz <= 0.0
            or self.role_stream_freshness <= 0.0
            or self.master_first_no_stream_retry <= 0.0
            or self.role_config_retries <= 0
            or self.follower_restore_movej_repeats <= 0
            or self.follower_restore_movej_period <= 0.0
            or self.drag_teach_timeout <= 0.0
            or self.drag_teach_status_grace <= 0.0
            or self.startup_role_timeout <= 0.0
            or self.max_joint_velocity <= 0.0
            or self.max_gripper_velocity <= 0.0
            or self.max_joint_acceleration <= 0.0
            or self.max_gripper_acceleration <= 0.0
            or self.alignment_max_joint_velocity <= 0.0
            or self.alignment_max_gripper_velocity <= 0.0
            or self.alignment_max_joint_acceleration <= 0.0
            or self.alignment_max_gripper_acceleration <= 0.0
            or self.home_max_joint_velocity <= 0.0
            or self.home_max_gripper_velocity <= 0.0
            or self.role_pose_tolerance <= 0.0
            or self.role_gripper_tolerance <= 0.0
            or self.role_pose_hard_tolerance < self.role_pose_tolerance
            or self.role_gripper_hard_tolerance < self.role_gripper_tolerance
        ):
            raise ValueError("leader timing, retries and velocity limits must be positive")

        self._lock = threading.RLock()
        # Serializes the final OUTPUT JointCtrl/GripperCtrl call with the
        # takeover stop acknowledgement.  The mux may not send 0xFA until this
        # barrier reports quiet, so a cached position frame cannot overtake the
        # role-switch frame in SocketCAN.
        self._tx_lock = threading.Lock()
        self._desired_mode = "IDLE"
        self._idle_since = time.monotonic()
        self._role = "UNKNOWN"
        self._state = "STARTING"
        self._enable_requested = bool(rospy.get_param("~auto_enable", False))
        self._emergency_stop_requested = False
        self._home_active = False
        self._inference_active = False
        self._leader_tx_enabled = False
        self._output_enable_verified = False
        self._drag_teach_active = False
        self._drag_teach_status_lost_at = 0.0
        self._command = None
        self._last_feedback = None
        self._command_output = None
        self._command_velocity = [0.0] * 7
        self._last_command_time = time.monotonic()
        self._shutting_down = False
        self._last_safety_check = 0.0
        self._last_safety_ok = True
        # Latch one failed firmware-role request until the mux publishes a
        # different mode.  Without this gate, recovering UNKNOWN -> OUTPUT can
        # immediately start the same failed 0xFA sequence again, producing a
        # repeated enable/timeout/disable cycle on the physical leader.
        self._failed_role_request = None

        self._feedback_pub = rospy.Publisher(
            "/leader/joint_states", JointState, queue_size=1, tcp_nodelay=True
        )
        self._state_pub = rospy.Publisher(
            "/leader/control_state", String, queue_size=1, latch=True
        )
        self._leader_tx_quiet_pub = rospy.Publisher(
            "/leader/tx_quiet", Bool, queue_size=1, latch=True
        )
        self._command_sub = rospy.Subscriber(
            "/leader/joint_command",
            JointState,
            self._command_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._mode_sub = rospy.Subscriber(
            "/teleop/mode", String, self._mode_callback, queue_size=1
        )
        self._enable_sub = rospy.Subscriber(
            "/enable_flag", Bool, self._enable_callback, queue_size=1
        )
        self._home_sub = rospy.Subscriber(
            "/teleop/home_active", Bool, self._home_callback, queue_size=1
        )
        self._inference_active_sub = rospy.Subscriber(
            "/teleop/inference_active",
            Bool,
            self._inference_active_callback,
            queue_size=1,
        )
        self._leader_tx_enable_sub = rospy.Subscriber(
            "/teleop/leader_tx_enable",
            Bool,
            self._leader_tx_enable_callback,
            queue_size=1,
        )
        self._emergency_stop_sub = rospy.Subscriber(
            "/teleop/emergency_stop",
            Bool,
            self._emergency_stop_callback,
            queue_size=1,
        )

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("Leader mode driver opening %s", self.can_port)
        self.piper = C_PiperInterface_V2(can_name=self.can_port)
        self.piper.ConnectPort(piper_init=False)
        # A gs_usb adapter can stop transmitting while staying UP; see
        # piper_can_tx_guard. The mux enters HOLD on the report, and a
        # down/up of the interface recovers it without a relaunch.
        self._tx_guard = CanTxGuard(
            self.can_port,
            stall_timeout=float(rospy.get_param("~tx_stall_timeout_s", 0.5)),
        )
        self._link_watch = CanLinkWatch(self.can_port)
        self._tx_stall_pub = rospy.Publisher(
            "/teleop/can_tx_stalled", String, queue_size=10
        )
        self._raw_master_observer = RawMasterObserver(self.can_port)
        self._detect_initial_role()
        # The default gate is closed. Publish a deterministic initial ACK even
        # if the latched mux message arrived before startup completed.
        if not self._leader_tx_enabled:
            self._leader_tx_quiet_pub.publish(Bool(data=True))

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self._state_pub.publish(String(data=state))
            rospy.logwarn("%s leader control state: %s", self.can_port, state)

    def _mode_callback(self, message):
        mode = message.data.strip().upper()
        if mode in ("IDLE", "HOLD", "COLLECT", "INFERENCE", "TAKEOVER"):
            with self._lock:
                if mode != self._desired_mode:
                    self._failed_role_request = None
                    if mode == "IDLE":
                        self._idle_since = time.monotonic()
                self._desired_mode = mode

    def _enable_callback(self, message):
        with self._lock:
            self._enable_requested = bool(message.data)

    def _home_callback(self, message):
        with self._lock:
            self._home_active = bool(message.data)

    def _inference_active_callback(self, message):
        with self._lock:
            self._inference_active = bool(message.data)

    def _leader_tx_enable_callback(self, message):
        """Close OUTPUT publishing with a real post-send barrier ACK."""
        enabled = bool(message.data)
        with self._lock:
            previous = self._leader_tx_enabled
            self._leader_tx_enabled = enabled
        if enabled:
            self._leader_tx_quiet_pub.publish(Bool(data=False))
            if not previous:
                rospy.logwarn("%s leader OUTPUT TX gate opened", self.can_port)
            return

        # Set the Boolean gate before taking _tx_lock. If run() already owns
        # the lock, this waits for that last CAN call to return; if it does not,
        # run() sees the closed gate and cannot start another send.
        with self._tx_lock:
            with self._lock:
                self._command = None
            self._leader_tx_quiet_pub.publish(Bool(data=True))
        if previous:
            rospy.logwarn(
                "%s leader OUTPUT TX gate closed; in-flight command drained",
                self.can_port,
            )

    def _emergency_stop_callback(self, message):
        with self._lock:
            self._emergency_stop_requested = bool(message.data)

    def _command_callback(self, message):
        if len(message.position) < 7:
            rospy.logwarn_throttle(2.0, "Leader command has fewer than 7 joints")
            return
        values = [float(value) for value in message.position[:7]]
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(2.0, "Ignoring non-finite leader command")
            return
        with self._lock:
            # Never cache a frame received after the takeover drain barrier.
            # A fresh command will arrive immediately when the mux reopens the
            # gate for HOLD recovery, HOME, or INFERENCE.
            if self._leader_tx_enabled:
                self._command = values

    def _driver_enable_states(self):
        low_speed = self.piper.GetArmLowSpdInfoMsgs()
        return [
            bool(low_speed.motor_1.foc_status.driver_enable_status),
            bool(low_speed.motor_2.foc_status.driver_enable_status),
            bool(low_speed.motor_3.foc_status.driver_enable_status),
            bool(low_speed.motor_4.foc_status.driver_enable_status),
            bool(low_speed.motor_5.foc_status.driver_enable_status),
            bool(low_speed.motor_6.foc_status.driver_enable_status),
        ]

    def _motor_safety_faults(self):
        """Return active low-speed motor protections reported by firmware."""
        low_speed = self.piper.GetArmLowSpdInfoMsgs()
        faults = []
        for number in range(1, 7):
            motor = getattr(low_speed, "motor_{}".format(number))
            # can_id==0 means this SDK instance has not received that motor's
            # low-speed frame yet.  Enable verification handles missing data;
            # do not manufacture a fault from an uninitialised SDK object.
            if int(getattr(motor, "can_id", 0)) == 0:
                continue
            status = motor.foc_status
            for attribute, label in (
                ("voltage_too_low", "low voltage"),
                ("motor_overheating", "motor overtemperature"),
                ("driver_overcurrent", "driver overcurrent"),
                ("driver_overheating", "driver overtemperature"),
                ("collision_status", "collision protection"),
                ("driver_error_status", "driver error"),
                ("stall_status", "stall protection"),
            ):
                if bool(getattr(status, attribute, False)):
                    faults.append("J{} {}".format(number, label))
        arm_status_message = self.piper.GetArmStatus()
        if float(getattr(arm_status_message, "time_stamp", 0.0)) > 0.0:
            controller_status = int(arm_status_message.arm_status.arm_status)
            if controller_status == 0x0E:
                faults.append("main controller NTC overtemperature")
            elif controller_status == 0x0F:
                faults.append("release resistor NTC overtemperature")
        return faults

    def _safety_ok(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_safety_check < 0.10:
            return self._last_safety_ok
        self._last_safety_check = now
        faults = self._motor_safety_faults()
        if faults:
            rospy.logerr(
                "%s refuses enable/motion because firmware reports: %s",
                self.can_port,
                ", ".join(faults),
            )
            self._last_safety_ok = False
            return False
        self._last_safety_ok = True
        return True

    def _reset_command_trajectory(self):
        """Restart OUTPUT interpolation from measured pose with zero velocity."""
        self._command_output = (
            list(self._last_feedback) if self._last_feedback is not None else None
        )
        self._command_velocity = [0.0] * 7
        self._last_command_time = time.monotonic()
        # The first command after any reset re-asserts MOVE J mode.
        self._last_motion_ctrl = 0.0

    def _enable_and_wait(self, position_gripper=True):
        if not self._safety_ok(force=True):
            return False
        deadline = time.monotonic() + self.enable_timeout
        states = self._driver_enable_states()
        while not rospy.is_shutdown() and not self._shutting_down:
            states = self._driver_enable_states()
            if all(states):
                # Never invent a zero-angle gripper target during startup: it
                # would look like an unexpected close if joint feedback has
                # not arrived yet.
                if position_gripper and self._last_feedback is not None:
                    gripper = int(round(self._last_feedback[6] * 1000000.0))
                    self.piper.GripperCtrl(gripper, 1000, 0x01, 0)
                elif not position_gripper and self._last_feedback is not None:
                    # MASTER owns the gripper as an input.  Keep its measured
                    # opening but release the position loop before 0xFA.
                    gripper = int(round(self._last_feedback[6] * 1000000.0))
                    self.piper.GripperCtrl(gripper, 1000, 0x00, 0)
                self._output_enable_verified = True
                self._reset_command_trajectory()
                rospy.loginfo("%s leader output motors are enabled", self.can_port)
                return True
            if time.monotonic() >= deadline:
                break
            if not self._safety_ok(force=True):
                return False
            self.piper.EnableArm(7)
            time.sleep(self.enable_retry_period)
        rospy.logerr("%s leader enable timed out; motors=%s", self.can_port, states)
        return False

    def _disable_and_wait(self):
        """Physically disable all leader joints and verify motor feedback."""
        self._output_enable_verified = False
        self._reset_command_trajectory()
        gripper = None
        if self._last_feedback is not None:
            gripper = int(round(self._last_feedback[6] * 1000000.0))
        if self._role != "OUTPUT":
            # A 0xFA teaching-input arm suppresses the normal low-speed motor
            # feedback used for verification. Retrying DisableArm until that
            # unavailable feedback changes creates a visible torque oscillation.
            # Close the software command gate and send exactly one disable.
            self.piper.DisableArm(7)
            if gripper is not None:
                self.piper.GripperCtrl(gripper, 1000, 0x00, 0)
            self._set_state("DISABLED")
            rospy.logwarn(
                "%s leader disable sent once; feedback unavailable in %s role",
                self.can_port,
                self._role,
            )
            return True
        deadline = time.monotonic() + self.enable_timeout
        states = self._driver_enable_states()
        while not rospy.is_shutdown() and not self._shutting_down:
            states = self._driver_enable_states()
            if not any(states):
                self._output_enable_verified = False
                self._set_state("DISABLED")
                rospy.loginfo("%s leader motors are disabled", self.can_port)
                return True
            if time.monotonic() >= deadline:
                break
            self.piper.DisableArm(7)
            if gripper is not None:
                self.piper.GripperCtrl(gripper, 1000, 0x00, 0)
            time.sleep(self.enable_retry_period)
        rospy.logerr("%s leader disable timed out; motors=%s", self.can_port, states)
        self._set_state("ERROR")
        return False

    def _role_stream_snapshot(self, role):
        if role == "OUTPUT":
            joint = self.piper.GetArmJointMsgs()
            gripper = self.piper.GetArmGripperMsgs()
            return (
                float(joint.time_stamp),
                float(gripper.time_stamp),
                float(joint.Hz),
                float(gripper.Hz),
            )
        observer = getattr(self, "_raw_master_observer", None)
        if observer is not None:
            raw_snapshot = observer.snapshot()
            if raw_snapshot[0] > 0.0:
                return raw_snapshot
        joint = self.piper.GetArmJointCtrl()
        gripper = self.piper.GetArmGripperCtrl()
        return (
            float(joint.time_stamp),
            float(gripper.time_stamp),
            float(joint.Hz),
            float(gripper.Hz),
        )

    def _stream_timestamp(self, role):
        return self._role_stream_snapshot(role)[0]

    def _role_stream_active(self, role):
        joint_stamp, gripper_stamp, joint_hz, gripper_hz = (
            self._role_stream_snapshot(role)
        )
        if role == "MASTER":
            return joint_stamp > 0.0 and joint_hz >= self.role_stream_min_hz
        return (
            joint_stamp > 0.0
            and gripper_stamp > 0.0
            and joint_hz >= self.role_stream_min_hz
            and gripper_hz >= self.role_stream_min_hz
        )

    def _detect_initial_role(self):
        deadline = time.monotonic() + self.startup_role_timeout
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self._probe_current_role():
                return
            time.sleep(0.02)
        rospy.logerr(
            "%s initial leader role is not emitting a recognizable CAN stream; "
            "waiting for an explicit mode request",
            self.can_port,
        )
        self._role = "UNKNOWN"
        self._set_state("WAITING_FEEDBACK")

    def _probe_current_role(self):
        master_active = self._role_stream_active("MASTER")
        output_active = self._role_stream_active("OUTPUT")
        if master_active and not output_active:
            self._role = "MASTER"
            self._set_state("MASTER")
            return True
        if output_active and not master_active:
            self._role = "OUTPUT"
            self._set_state("OUTPUT")
            return True
        return False

    def _wait_for_new_stream(
        self, role, previous_snapshot, no_stream_retry_s=None
    ):
        """Require a complete, continuously updating role stream.

        One changed timestamp is insufficient: it can be a final queued frame
        from the old role.  Both joint and gripper streams must differ from the
        baseline, report complete SDK Hz, and continue changing for the stable
        interval before the role is exposed to the mux.
        """
        started_at = time.monotonic()
        deadline = started_at + self.role_timeout
        final_deadline = deadline + self.role_stream_stable
        previous_joint, previous_gripper = previous_snapshot[:2]
        observed_joint = previous_joint
        observed_gripper = previous_gripper
        joint_seen_at = 0.0
        gripper_seen_at = 0.0
        stable_since = None
        while not rospy.is_shutdown() and not self._shutting_down:
            now = time.monotonic()
            joint_stamp, gripper_stamp, joint_hz, gripper_hz = (
                self._role_stream_snapshot(role)
            )
            if joint_stamp > 0.0 and joint_stamp != observed_joint:
                observed_joint = joint_stamp
                joint_seen_at = now
            if gripper_stamp > 0.0 and gripper_stamp != observed_gripper:
                observed_gripper = gripper_stamp
                gripper_seen_at = now
            if role == "MASTER":
                # 0x155-0x157 carry the six teaching joints. Some firmware/SDK
                # combinations do not keep 0x159 gripper-control timestamps
                # fresh, so do not reject an otherwise valid teaching stream
                # only because the gripper frame is quiet.
                complete = (
                    joint_stamp != previous_joint
                    and joint_hz >= self.role_stream_min_hz
                    and joint_seen_at > 0.0
                    and now - joint_seen_at <= self.role_stream_freshness
                )
            else:
                complete = (
                    joint_stamp != previous_joint
                    and gripper_stamp != previous_gripper
                    and joint_hz >= self.role_stream_min_hz
                    and gripper_hz >= self.role_stream_min_hz
                    and joint_seen_at > 0.0
                    and gripper_seen_at > 0.0
                    and now - joint_seen_at <= self.role_stream_freshness
                    and now - gripper_seen_at <= self.role_stream_freshness
                )
            if complete:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= self.role_stream_stable:
                    return True
            else:
                stable_since = None
            # Keep the mux input alive while firmware changes its CAN role.
            # The arm is stationary at the verified pre-transition pose here.
            if self._last_feedback is not None:
                self._feedback_pub.publish(self._message(self._last_feedback))
            if (
                no_stream_retry_s is not None
                and observed_joint == previous_joint
                and now - started_at >= no_stream_retry_s
            ):
                rospy.logwarn(
                    "%s %s produced no new joint frames after %.2fs; "
                    "retrying the role command without waiting %.2fs",
                    getattr(self, "can_port", "leader"),
                    role,
                    no_stream_retry_s,
                    self.role_timeout,
                )
                return False
            if now >= deadline and stable_since is None:
                rospy.logwarn(
                    "%s %s stream timeout snapshot: joint_stamp=%.3f "
                    "gripper_stamp=%.3f joint_hz=%.1f gripper_hz=%.1f",
                    getattr(self, "can_port", "leader"),
                    role,
                    joint_stamp,
                    gripper_stamp,
                    joint_hz,
                    gripper_hz,
                )
                return False
            if now >= final_deadline:
                rospy.logwarn(
                    "%s %s stream timeout snapshot: joint_stamp=%.3f "
                    "gripper_stamp=%.3f joint_hz=%.1f gripper_hz=%.1f",
                    getattr(self, "can_port", "leader"),
                    role,
                    joint_stamp,
                    gripper_stamp,
                    joint_hz,
                    gripper_hz,
                )
                return False
            time.sleep(0.02)
        return False

    def _settle_stream(self):
        deadline = time.monotonic() + self.role_settle
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self._last_feedback is not None:
                self._feedback_pub.publish(self._message(self._last_feedback))
            time.sleep(0.02)

    def _read_role_pose(self, role):
        """Read J1..J6/G from the CAN stream owned by one physical role."""
        if role == "OUTPUT":
            joint_message = self.piper.GetArmJointMsgs()
            joints = joint_message.joint_state
            gripper = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        else:
            observer = getattr(self, "_raw_master_observer", None)
            if observer is not None:
                raw_pose = observer.pose()
                if raw_pose is not None:
                    return raw_pose
            joint_message = self.piper.GetArmJointCtrl()
            joints = joint_message.joint_ctrl
            gripper_message = self.piper.GetArmGripperCtrl()
            if float(gripper_message.time_stamp) > 0.0:
                gripper = gripper_message.gripper_ctrl.grippers_angle
            elif self._last_feedback is not None:
                gripper = int(round(self._last_feedback[6] * 1000000.0))
            else:
                gripper = 0
        if float(joint_message.time_stamp) <= 0.0:
            return None
        values = [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
        ]
        pose = [value * MILLI_DEG_TO_RAD for value in values]
        pose.append(max(0.0, min(0.08, gripper / 1000000.0)))
        return pose

    def _validate_role_pose_continuity(
        self, reference, observed, role, allow_operator_motion=False
    ):
        """Check role-stream pose continuity without mistaking motion for mapping.

        A strict comparison is useful while the arm is stationary.  During the
        runtime 0xFA/0xFC hand-off, however, the controller releases the leader
        before the SDK's stream-rate estimate has settled.  The operator can
        therefore move a real joint by more than the hard tolerance while this
        function is waiting.  That is motion over time, not a changed CAN joint
        order.  Runtime hand-offs use ``allow_operator_motion`` only after a
        complete, continuously updating role stream has already been verified;
        the mux continues to slew-limit every follower command.
        """
        if reference is None or observed is None:
            rospy.logerr(
                "%s cannot validate %s mapping continuity without both poses",
                self.can_port,
                role,
            )
            return False
        joint_deltas = [
            observed[index] - reference[index] for index in range(6)
        ]
        gripper_delta = observed[6] - reference[6]
        rospy.loginfo(
            "%s %s mapping check delta [J1..J6 rad]=%s G=%.5fm",
            self.can_port,
            role,
            "[{}]".format(
                ", ".join("{:+.5f}".format(value) for value in joint_deltas)
            ),
            gripper_delta,
        )
        max_joint_delta = max(abs(value) for value in joint_deltas)
        max_gripper_delta = abs(gripper_delta)
        if (
            max_joint_delta > self.role_pose_hard_tolerance
            or max_gripper_delta > self.role_gripper_hard_tolerance
        ):
            if allow_operator_motion:
                rospy.logwarn(
                    "%s %s pose changed beyond the stationary mapping gate "
                    "during the live role hand-off; treating it as operator "
                    "motion because the complete role stream is verified "
                    "(follower output remains velocity-limited)",
                    self.can_port,
                    role,
                )
                return True
            rospy.logerr(
                "%s %s mapping continuity failed; refusing role activation",
                self.can_port,
                role,
            )
            return False
        if (
            max_joint_delta > self.role_pose_tolerance
            or max_gripper_delta > self.role_gripper_tolerance
        ):
            rospy.logwarn(
                "%s %s mapping continuity exceeded nominal tolerance but is "
                "within hard tolerance; allowing role activation",
                self.can_port,
                role,
            )
        return True

    def _seed_output_target(self, pose, gripper_code=0x00):
        """Write the measured pose as target before CAN MOVE J or enable."""
        if pose is None or len(pose) < 7:
            return False
        joints = [int(round(value * RAD_TO_MILLI_DEG)) for value in pose[:6]]
        gripper = int(round(max(0.0, min(0.08, pose[6])) * 1000000.0))
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(gripper, 1000, gripper_code, 0)
        self._last_feedback = list(pose)
        self._reset_command_trajectory()
        return True

    def _arm_output_at_current_pose(self):
        """Enter enabled CAN MOVE J and clear S-V1.8-8 torque-mode residue.

        The arm can already be publishing normal 0x2Ax OUTPUT feedback when a
        previous fixed-MASTER session left the firmware's torque-control state
        partially active.  Apply the documented current-pose MOVE J recovery
        on every first OUTPUT enable, not only after an observed runtime 0xFC.
        """
        observer = getattr(self, "_raw_master_observer", None)
        if observer is not None:
            observer.stop_capture()
        pose = self._read_role_pose("OUTPUT")
        if pose is None:
            pose = self._last_feedback
        if pose is None:
            rospy.logerr("%s has no OUTPUT pose to seed before enable", self.can_port)
            return False
        # Standby first, preload the measured target while unpowered, enter
        # CAN MOVE J, then repeat the same target before enabling the motors.
        self.piper.MotionCtrl_2(0x00, 0x01, 0)
        self._settle_stream()
        if not self._seed_output_target(pose, gripper_code=0x00):
            return False
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_speed_percent)
        self._seed_output_target(pose, gripper_code=0x00)
        if not self._enable_and_wait():
            return False
        # Repeat the measured target after the enable feedback changes so no
        # controller-stored target can become the first powered command.  This
        # also covers a fixed-MASTER -> supervised-launch hand-off where normal
        # OUTPUT feedback appeared before this process could observe the 0xFC.
        if not self._seed_output_target(pose, gripper_code=0x01):
            return False
        if not self._repeat_follower_restore_target(pose):
            return False
        rospy.logwarn(
            "%s OUTPUT enable recovery applied %d measured-pose MOVE J commands",
            self.can_port,
            self.follower_restore_movej_repeats,
        )
        return True

    def _repeat_follower_restore_target(self, pose):
        """Apply the v1.8-5--v1.8-8 Leader->Follower recovery workaround.

        AgileX documents that these firmware versions may retain the leader's
        torque-control state after 0xFC.  Several MOVE J commands at the
        measured current pose make the transition deterministic without an
        enable/disable cycle or an old stored-position jump.
        """
        if pose is None or len(pose) < 7:
            return False
        for _ in range(self.follower_restore_movej_repeats):
            self.piper.MotionCtrl_2(0x01, 0x01, self.move_speed_percent)
            if not self._seed_output_target(pose, gripper_code=0x01):
                return False
            time.sleep(self.follower_restore_movej_period)
        return True

    def _prepare_disabled_output_for_master(self):
        """Leave OUTPUT unpowered and capture its pose immediately before 0xFA."""
        if not self._safety_ok(force=True):
            return None
        # The official piper_set_master demo sends 0xFA without enabling the
        # arm.  This is retained only for a legacy COLLECT role request; T
        # takeover has a separate drag-teach path that never disables joints.
        if not self._disable_and_wait():
            return None
        if self.master_prepare_hold > 0.0:
            time.sleep(self.master_prepare_hold)
        reference = self._read_role_pose("OUTPUT")
        if reference is None:
            reference = self._last_feedback
        if reference is None:
            rospy.logerr(
                "%s has no disabled OUTPUT pose before MASTER transition",
                self.can_port,
            )
            return None
        rospy.logwarn(
            "%s disabled OUTPUT pose ready for 0xFA (%.2fs settle)",
            self.can_port,
            self.master_prepare_hold,
        )
        return reference

    def _prepare_enabled_output_for_master(self):
        """Fallback: hold the measured OUTPUT pose before 0xFA."""
        if not self._arm_output_at_current_pose():
            return None
        if self.master_prepare_hold > 0.0:
            time.sleep(self.master_prepare_hold)
        reference = self._read_role_pose("OUTPUT")
        if reference is None:
            reference = self._last_feedback
        if reference is None:
            rospy.logerr(
                "%s has no enabled OUTPUT pose before MASTER transition",
                self.can_port,
            )
            return None
        rospy.logwarn(
            "%s enabled OUTPUT fallback pose ready for 0xFA (%.2fs settle)",
            self.can_port,
            self.master_prepare_hold,
        )
        return reference

    def _linkage_teaching_mode_active(self):
        """Whether 0x2A1 confirms linkage teaching-input mode (0x06)."""
        status_message = self.piper.GetArmStatus()
        if float(getattr(status_message, "time_stamp", 0.0)) <= 0.0:
            return False
        ctrl_mode = int(status_message.arm_status.ctrl_mode)
        if ctrl_mode == 0x06:
            rospy.logwarn(
                "%s 0x2A1 confirms linkage teaching-input mode (ctrl_mode=0x06); "
                "waiting for the first physical MASTER movement frame",
                self.can_port,
            )
            return True
        return False

    def _control_mode(self):
        """Return the latest 0x2A1 control mode, or ``None`` before feedback."""
        status_message = self.piper.GetArmStatus()
        if float(getattr(status_message, "time_stamp", 0.0)) <= 0.0:
            return None
        return int(status_message.arm_status.ctrl_mode)

    def _drag_teach_mode_active(self):
        """The official 0x150 drag-teach acknowledgement is ctrl_mode 0x02."""
        return self._control_mode() == 0x02

    def _wait_for_drag_teach_mode(self, expected):
        """Wait for 0x2A1 to enter or leave drag-teach without losing feedback."""
        deadline = time.monotonic() + self.drag_teach_timeout
        last_mode = None
        while not rospy.is_shutdown() and not self._shutting_down:
            last_mode = self._control_mode()
            active = last_mode == 0x02
            if active == bool(expected):
                return True
            pose = self._read_role_pose("OUTPUT")
            if pose is not None:
                self._last_feedback = list(pose)
                self._feedback_pub.publish(self._message(pose))
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        rospy.logerr(
            "%s drag-teach acknowledgement timed out; expected=%s ctrl_mode=%s",
            self.can_port,
            "0x02" if expected else "not 0x02",
            "none" if last_mode is None else "0x{:02X}".format(last_mode),
        )
        return False

    def _restore_output_hold(self, reference=None):
        """Leave drag teaching and immediately hold the measured OUTPUT pose.

        This path deliberately sends neither 0xFA/0xFC nor DisableArm.  The
        position target is seeded before leaving teach mode, then repeated as
        CAN MOVE J takes ownership so a stored controller target cannot appear.
        """
        pose = self._read_role_pose("OUTPUT")
        if pose is None:
            pose = reference if reference is not None else self._last_feedback
        if pose is None:
            rospy.logerr("%s has no measured pose for drag-teach exit", self.can_port)
            return False
        pose = list(pose)
        if not self._seed_output_target(pose, gripper_code=0x00):
            return False
        self.piper.MotionCtrl_1(0x00, 0x00, 0x02)
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_speed_percent)
        self._seed_output_target(pose, gripper_code=0x01)
        if not all(self._driver_enable_states()) and not self._enable_and_wait():
            return False
        self._drag_teach_active = False
        self._drag_teach_status_lost_at = 0.0
        self._role = "OUTPUT"
        self._output_enable_verified = True
        self._last_feedback = pose
        self._reset_command_trajectory()
        return True

    def _enter_drag_teach(self):
        """Enter runtime gravity-compensated takeover without changing 0xFC.

        ``MotionCtrl_1(..., grag_teach_ctrl=0x01)`` is the SDK's documented
        drag-teach command.  The gripper position loop is released at its
        measured opening, while joint motor power is retained.  Success is
        exposed only after 0x2A1 confirms control mode 0x02.
        """
        if not self._enable_requested:
            self._set_state("WAITING_ENABLE")
            return False
        if self._role != "OUTPUT":
            rospy.logerr(
                "%s cannot enter runtime drag teaching from role %s",
                self.can_port,
                self._role,
            )
            self._set_state("TEACH_FAILED")
            return False
        reference = self._read_role_pose("OUTPUT")
        if reference is None:
            reference = self._last_feedback
        if reference is None:
            rospy.logerr("%s has no OUTPUT pose before drag teaching", self.can_port)
            self._set_state("TEACH_FAILED")
            return False
        reference = list(reference)
        if not self._output_enable_verified and not self._arm_output_at_current_pose():
            self._set_state("TEACH_FAILED")
            return False

        self._set_state("SWITCHING_TEACH")
        gripper = int(round(max(0.0, min(0.08, reference[6])) * 1000000.0))
        # Release only the leader gripper position loop.  Never disable joints
        # here: the operator is relying on the two-second powered hand-off.
        self.piper.GripperCtrl(gripper, 1000, 0x00, 0)
        for attempt in range(1, self.role_config_retries + 1):
            self.piper.MotionCtrl_1(0x00, 0x00, 0x01)
            if self._wait_for_drag_teach_mode(True):
                observed = self._read_role_pose("OUTPUT")
                if observed is not None:
                    self._last_feedback = list(observed)
                self._drag_teach_active = True
                self._drag_teach_status_lost_at = 0.0
                self._output_enable_verified = True
                self._command_output = None
                with self._lock:
                    self._failed_role_request = None
                self._set_state("DRAG_TEACH")
                rospy.logwarn(
                    "%s drag teaching active: 0x2A1 ctrl_mode=0x02; "
                    "0x2Ax feedback now owns follower commands",
                    self.can_port,
                )
                return True
            rospy.logwarn(
                "%s drag-teach request attempt %d/%d was not acknowledged",
                self.can_port,
                attempt,
                self.role_config_retries,
            )

        rospy.logerr(
            "%s cannot enter drag teaching; restoring powered OUTPUT hold "
            "without sending 0xFA or disabling the leader",
            self.can_port,
        )
        if not self._restore_output_hold(reference):
            self._set_state("ERROR")
            return False
        self._set_state("TEACH_FAILED")
        return False

    def _exit_drag_teach(self):
        """Return from manual takeover to a powered current-pose OUTPUT hold."""
        self._set_state("SWITCHING_OUTPUT")
        reference = self._read_role_pose("OUTPUT")
        if reference is None:
            reference = self._last_feedback
        if not self._restore_output_hold(reference):
            self._set_state("ERROR")
            return False
        self._set_state("OUTPUT")
        rospy.logwarn(
            "%s drag teaching ended; powered OUTPUT hold restored at measured pose",
            self.can_port,
        )
        return True

    def _drag_teach_is_healthy(self):
        """Fail closed if firmware unexpectedly leaves drag teaching."""
        if self._drag_teach_mode_active():
            self._drag_teach_status_lost_at = 0.0
            return True
        now = time.monotonic()
        if self._drag_teach_status_lost_at == 0.0:
            self._drag_teach_status_lost_at = now
            return True
        if now - self._drag_teach_status_lost_at <= self.drag_teach_status_grace:
            return True
        rospy.logerr(
            "%s left 0x2A1 drag-teach mode during TAKEOVER; restoring powered hold",
            self.can_port,
        )
        reference = self._read_role_pose("OUTPUT")
        if not self._restore_output_hold(reference):
            self._set_state("ERROR")
        else:
            self._set_state("TEACH_FAILED")
        return False

    def _accept_master_armed_without_stream(self, reference, source_label):
        """Accept a status-confirmed stationary MASTER without fabricating input.

        Some controllers only emit 0x155-0x157 when the operator first moves
        in teaching-input mode.  Their 0x2A1 ctrl_mode=0x06 is still an
        authoritative confirmation that gravity-compensated linkage teaching
        has entered; retain the measured OUTPUT pose until real input arrives.
        """
        self._role = "MASTER"
        self._last_feedback = list(reference)
        self._output_enable_verified = True
        self._command_output = None
        with self._lock:
            self._failed_role_request = None
        rospy.logwarn(
            "%s MASTER armed via %s; 0x155-0x157 will be consumed after "
            "the operator starts moving the arm",
            self.can_port,
            source_label,
        )
        self._set_state("MASTER_ARMED")
        return True

    def _log_master_handshake_diagnostic(
        self,
        observer,
        since,
        attempt,
        success,
        final_attempt,
        master_stream=False,
    ):
        """Classify one runtime 0xFA attempt from independent raw CAN data."""
        if observer is None or since is None:
            rospy.logwarn(
                "%s MASTER handshake attempt %d has no raw observer",
                self.can_port,
                attempt,
            )
            return

        report = observer.transition_report(since)
        rospy.logwarn(
            "%s MASTER handshake attempt %d bus timeline: %s",
            self.can_port,
            attempt,
            report,
        )
        role_echo = observer.role_config_echo_seen(since, 0xFA)
        standby_echo = observer.standby_echo_seen(since)
        restore_echo = observer.restore_master_slave_echo_seen(since)
        modes = observer.ctrl_modes_since(since)
        can_errors = observer.can_errors_seen(since)
        modes_text = "->".join("0x{:02X}".format(mode) for mode in modes)
        if success:
            rospy.logwarn(
                "%s MASTER handshake VERIFIED: 0x470_TX_ECHO=%s "
                "ctrl_mode_sequence=%s master_stream=%s",
                self.can_port,
                role_echo,
                modes_text if modes_text else "none",
                master_stream,
            )
            return

        log = rospy.logerr if final_attempt else rospy.logwarn
        if not role_echo:
            log(
                "%s MASTER handshake diagnosis: no 0x470/FA TX echo was "
                "observed on %s; investigate the host SocketCAN/gs_usb TX "
                "path before attributing the failure to arm firmware",
                self.can_port,
                self.can_port,
            )
        elif can_errors:
            log(
                "%s MASTER handshake diagnosis: 0x470/FA was echoed but CAN "
                "error frames occurred; inspect interface TX/RX error counters",
                self.can_port,
            )
        elif 0x06 in modes:
            log(
                "%s MASTER handshake diagnosis: controller reported 0x2A1 "
                "ctrl_mode=0x06 after an echoed 0x470/FA, but no complete "
                "0x155-0x157 stream followed; role activation succeeded but "
                "the MASTER control stream is missing",
                self.can_port,
            )
        elif not modes:
            log(
                "%s MASTER handshake diagnosis: 0x470/FA TX echo is present "
                "without CAN errors, but 0x2A1 disappeared and no MASTER "
                "stream followed; controller entered an unreported firmware state",
                self.can_port,
            )
        elif standby_echo and not restore_echo and final_attempt:
            log(
                "%s MASTER handshake diagnosis: 0xFA and STANDBY were echoed "
                "but 0x191 mode=0 restore was not echoed; investigate the "
                "host TX path for the explicit linkage restore command",
                self.can_port,
            )
        else:
            log(
                "%s MASTER handshake diagnosis: 0x470/FA TX echo is present "
                "without CAN errors, but ctrl_mode remained %s and never "
                "became 0x06%s%s; controller ignored/rejected the role activation",
                self.can_port,
                modes_text,
                " after an echoed STANDBY" if standby_echo else "",
                " and echoed 0x191 restore" if restore_echo else "",
            )

    def _activate_master_stream(self, reference, source_label):
        observer = getattr(self, "_raw_master_observer", None)
        if observer is not None:
            observer.start_capture(reference)
        before = self._role_stream_snapshot("MASTER")
        for attempt in range(1, self.role_config_retries + 1):
            # Try the official minimal 0x470/0xFA write first. The protocol
            # appendix additionally requires restart *or standby* to activate
            # a saved teaching-input role. Use standby only on the final
            # attempt, after the seamless path was proven silent and after the
            # mux has stopped policy and frozen both followers.
            attempt_started = (
                observer.transition_marker() if observer is not None else None
            )
            self.piper.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
            self._settle_stream()
            standby_activation = (
                getattr(self, "master_standby_activation", True)
                and attempt == self.role_config_retries
            )
            if standby_activation:
                rospy.logwarn(
                    "%s activating saved 0xFA through protocol-required "
                    "STANDBY; operator support and follower HOLD remain active",
                    self.can_port,
                )
                self.piper.MotionCtrl_2(0x00, 0x01, 0)
                self._output_enable_verified = False
                self._settle_stream()
                if (
                    getattr(self, "master_restore_linkage_activation", True)
                    and self._control_mode() == 0x00
                ):
                    rospy.logwarn(
                        "%s controller is in protocol STANDBY; sending 0x191 "
                        "mode=0 to restore zero-force master/slave linkage "
                        "without HOME or motor disable",
                        self.can_port,
                    )
                    self.piper.ReqMasterArmMoveToHome(0)
                    self._settle_stream()
            # A valid 0xFA transition can be silent until the first manual
            # movement.  Do not restore OUTPUT and remove gravity compensation
            # when the controller itself confirms linkage teaching input.
            if self._linkage_teaching_mode_active():
                self._log_master_handshake_diagnostic(
                    observer,
                    attempt_started,
                    attempt,
                    success=True,
                    final_attempt=attempt == self.role_config_retries,
                    master_stream=False,
                )
                return self._accept_master_armed_without_stream(
                    reference, source_label
                )
            stream_active = self._wait_for_new_stream(
                "MASTER",
                before,
                no_stream_retry_s=(
                    getattr(self, "master_first_no_stream_retry", 1.0)
                    if attempt < self.role_config_retries
                    else None
                ),
            )
            self._log_master_handshake_diagnostic(
                observer,
                attempt_started,
                attempt,
                success=stream_active,
                final_attempt=attempt == self.role_config_retries,
                master_stream=stream_active,
            )
            if stream_active:
                observed = self._read_role_pose("MASTER")
                self._role = "MASTER"
                if not self._validate_role_pose_continuity(
                    reference,
                    observed,
                    "MASTER",
                    allow_operator_motion=True,
                ):
                    self._set_state("ERROR")
                    return False
                self._last_feedback = list(observed)
                self._output_enable_verified = True
                self._command_output = None
                with self._lock:
                    self._failed_role_request = None
                rospy.logwarn(
                    "%s MASTER teaching stream active via %s",
                    self.can_port,
                    source_label,
                )
                if observer is not None and observer.active_offset() is not None:
                    rospy.logwarn(
                        "%s MASTER input verified by read-only raw SocketCAN "
                        "at control offset 0x%02X",
                        self.can_port,
                        observer.active_offset(),
                    )
                self._set_state("MASTER")
                return True
            rospy.logwarn(
                "%s master-role transition attempt %d/%d via %s produced no "
                "fresh 0x155-0x157 joint-control feedback",
                self.can_port,
                attempt,
                self.role_config_retries,
                source_label,
            )
            if observer is not None:
                rospy.logwarn(
                    "%s read-only raw MASTER rates: %s",
                    self.can_port,
                    observer.rate_text(),
                )
            before = self._role_stream_snapshot("MASTER")
        return False

    def _enter_output_role(self):
        """Change a teaching-input leader into a commandable output arm."""
        observer = getattr(self, "_raw_master_observer", None)
        if observer is not None:
            observer.stop_capture()
        if not self._enable_requested:
            self._set_state("WAITING_ENABLE")
            return False
        self._set_state("SWITCHING_OUTPUT")
        reference = (
            self._read_role_pose(self._role)
            if self._role in ("MASTER", "OUTPUT")
            else self._last_feedback
        )
        before = self._role_stream_snapshot("OUTPUT")
        for attempt in range(1, self.role_config_retries + 1):
            # Match pyAgxArm set_follower_mode(): 0xFC is the role switch.
            # MotionCtrl_2 and current-pose targets are sent only after normal
            # 0x2Ax feedback proves the output role is live.
            self.piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
            self._settle_stream()
            if self._wait_for_new_stream("OUTPUT", before):
                observed = self._read_role_pose("OUTPUT")
                self._role = "OUTPUT"
                if not self._validate_role_pose_continuity(
                    reference,
                    observed,
                    "OUTPUT",
                    allow_operator_motion=True,
                ):
                    self._set_state("ERROR")
                    return False
                self._last_feedback = list(observed)
                if not self._arm_output_at_current_pose():
                    self._set_state("ERROR")
                    return False
                self._set_state("OUTPUT")
                rospy.logwarn(
                    "%s OUTPUT active at measured pose; applied %d current-pose "
                    "MOVE J recovery commands and used no stale target",
                    self.can_port,
                    self.follower_restore_movej_repeats,
                )
                return True
            rospy.logwarn(
                "%s output-role transition attempt %d/%d produced no feedback",
                self.can_port,
                attempt,
                self.role_config_retries,
            )
            before = self._role_stream_snapshot("OUTPUT")
        rospy.logerr(
            "%s did not start standard 0x2A5-0x2A8 feedback after 0xFC; "
            "this firmware may require an arm power cycle",
            self.can_port,
        )
        self._role = "UNKNOWN"
        self._set_state("ERROR")
        return False

    def _recover_output_feedback_role(self):
        """Recover an unidentifiable startup role without powering the arm.

        A previous 0xFA request can be retained by the controller while neither
        the normal 0x2Ax feedback nor the 0x15x teaching stream is visible at
        the next ROS startup.  Waiting for an active-mode request deadlocks the
        mux because it requires fresh leader feedback before accepting C/T.
        Restore 0xFC in standby, verify the normal joint stream, and leave the
        arm physically unpowered in IDLE. No MOVE J target is activated here.
        """
        observer = getattr(self, "_raw_master_observer", None)
        if observer is not None:
            observer.stop_capture()
        self._set_state("RECOVERING_OUTPUT_FEEDBACK")
        reference = self._last_feedback
        before = self._role_stream_snapshot("OUTPUT")
        for attempt in range(1, self.role_config_retries + 1):
            self.piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
            self._settle_stream()
            self.piper.MotionCtrl_2(0x00, 0x01, 0)
            self._settle_stream()
            if self._wait_for_new_stream("OUTPUT", before):
                observed = self._read_role_pose("OUTPUT")
                self._role = "OUTPUT"
                if reference is not None and not self._validate_role_pose_continuity(
                    reference, observed, "OUTPUT_RECOVERY"
                ):
                    self._set_state("ERROR")
                    return False
                self._last_feedback = list(observed)
                self._output_enable_verified = False
                self._reset_command_trajectory()
                if not self._disable_and_wait():
                    return False
                rospy.logwarn(
                    "%s startup role recovered to OUTPUT feedback with motors disabled",
                    self.can_port,
                )
                return True
            rospy.logwarn(
                "%s startup OUTPUT-feedback recovery attempt %d/%d produced no feedback",
                self.can_port,
                attempt,
                self.role_config_retries,
            )
            before = self._role_stream_snapshot("OUTPUT")
        rospy.logerr(
            "%s could not recover standard 0x2A5-0x2A8 feedback while disabled",
            self.can_port,
        )
        self._role = "UNKNOWN"
        self._set_state("ERROR")
        return False

    def _enter_master_role(self, allow_disable_fallback=True):
        """Change a motion-output leader into 0xFA teaching-input mode.

        TAKEOVER passes ``allow_disable_fallback=False``: a failed seamless
        hand-off must preserve powered support and must never drop the arm by
        trying the historical disable/re-enable fallback.
        """
        if not self._enable_requested:
            self._set_state("WAITING_ENABLE")
            return False
        self._set_state("SWITCHING_MASTER")
        reference = (
            self._read_role_pose(self._role)
            if self._role in ("OUTPUT", "MASTER")
            else self._last_feedback
        )
        if reference is None:
            rospy.logerr("%s has no pose before MASTER transition", self.can_port)
            self._set_state("ERROR")
            return False
        # The leader is already enabled and holding the same physical pose as
        # its follower.  Keep that powered hand-off until the single official
        # 0xFA role write takes ownership; no DisableArm is permitted here.
        if self.master_enable_fallback and self._role == "OUTPUT":
            reference = self._prepare_enabled_output_for_master()
            if reference is None:
                self._set_state("ERROR")
                return False
            if self._activate_master_stream(
                reference, "measured-pose OUTPUT hold"
            ):
                return True
            if not allow_disable_fallback:
                rospy.logerr(
                    "%s seamless 0xFA takeover was not confirmed; preserving "
                    "powered support and restoring 0xFC at the measured pose",
                    self.can_port,
                )
                self._role = (
                    "OUTPUT" if self._role_stream_active("OUTPUT") else "UNKNOWN"
                )
                if not self._enter_output_role():
                    self._set_state("ERROR")
                else:
                    # Report the failed hand-off for one mux cycle, then let
                    # HOLD/HOME continue from the verified powered OUTPUT
                    # state.  A persistent ERROR here previously made H and a
                    # later I appear dead even though 0xFC recovery succeeded.
                    self._set_state("TEACH_FAILED")
                return False
            rospy.logwarn(
                "%s measured-pose OUTPUT hold did not start MASTER; retrying "
                "the SDK-minimal disabled-output 0xFA sequence",
                self.can_port,
            )
            if not self._recover_output_feedback_role():
                self._set_state("ERROR")
                return False
            self._set_state("SWITCHING_MASTER")
        if self._role == "OUTPUT":
            reference = self._prepare_disabled_output_for_master()
            if reference is None:
                self._set_state("ERROR")
                return False
        if self._activate_master_stream(reference, "disabled-output 0xFA"):
            return True
        rospy.logerr(
            "%s did not start 0x155-0x157 control feedback after 0xFA",
            self.can_port,
        )
        self._role = "OUTPUT" if self._role_stream_active("OUTPUT") else "UNKNOWN"
        self._set_state("ERROR")
        return False

    def _enter_emergency_stop(self):
        self.piper.MotionCtrl_1(0x01, 0x00, 0x00)
        self._drag_teach_active = False
        self._drag_teach_status_lost_at = 0.0
        self._output_enable_verified = False
        self._reset_command_trajectory()
        self._set_state("ESTOP")
        rospy.logerr("%s QUICK EMERGENCY STOP active", self.can_port)

    def _recover_emergency_stop(self):
        self._set_state("RECOVERING")
        self.piper.MotionCtrl_1(0x02, 0x00, 0x00)
        # A fresh enable is required, but it is intentionally deferred to
        # _arm_output_at_current_pose(), which preloads measured joint/gripper
        # targets and checks firmware protection bits before applying power.
        self._output_enable_verified = False
        self._set_state("WAITING_ENABLE")
        rospy.logwarn(
            "%s emergency stop reset; safe measured-pose re-enable required",
            self.can_port,
        )

    def _read_pose(self):
        if self._role == "UNKNOWN":
            return self._last_feedback
        pose = self._read_role_pose(self._role)
        if pose is None:
            return self._last_feedback
        self._last_feedback = pose
        return pose

    def _send_command(self, position, home_active=False, inference_active=False):
        if not self._safety_ok():
            raise RuntimeError("firmware motor protection is active")
        now = time.monotonic()
        # Cap a delayed loop iteration so it cannot turn scheduler latency into
        # one large position jump.
        dt = min(0.05, max(1.0 / self.publish_hz, now - self._last_command_time))
        self._last_command_time = now
        if self._command_output is None:
            self._command_output = (
                list(self._last_feedback)
                if self._last_feedback is not None
                else list(position)
            )
            self._command_velocity = [0.0] * 7
        bounded = []
        for index, (old, target) in enumerate(zip(self._command_output, position)):
            velocity_limit = (
                self.home_max_gripper_velocity
                if home_active and index == 6
                else self.home_max_joint_velocity
                if home_active
                else self.alignment_max_gripper_velocity
                if not inference_active and index == 6
                else self.alignment_max_joint_velocity
                if not inference_active
                else self.max_gripper_velocity
                if index == 6
                else self.max_joint_velocity
            )
            acceleration_limit = (
                self.alignment_max_gripper_acceleration
                if not home_active and not inference_active and index == 6
                else self.alignment_max_joint_acceleration
                if not home_active and not inference_active
                else self.max_gripper_acceleration
                if index == 6
                else self.max_joint_acceleration
            )
            delta = target - old
            desired_velocity = max(
                -velocity_limit, min(velocity_limit, delta / dt)
            )
            previous_velocity = self._command_velocity[index]
            velocity_step = acceleration_limit * dt
            next_velocity = previous_velocity + max(
                -velocity_step,
                min(velocity_step, desired_velocity - previous_velocity),
            )
            position_step = next_velocity * dt
            if (
                delta == 0.0
                or position_step * delta <= 0.0
                or abs(position_step) >= abs(delta)
            ):
                bounded.append(target)
                self._command_velocity[index] = 0.0
            else:
                bounded.append(old + position_step)
                self._command_velocity[index] = next_velocity
        self._command_output = bounded
        joints = [int(round(value * RAD_TO_MILLI_DEG)) for value in bounded[:6]]
        gripper = int(round(max(0.0, min(0.08, bounded[6])) * 1000000.0))
        now = time.monotonic()
        if now - self._last_motion_ctrl >= self.motion_ctrl_period:
            self.piper.MotionCtrl_2(0x01, 0x01, self.move_speed_percent)
            self._last_motion_ctrl = now
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(gripper, 1000, 0x01, 0)
        self._tx_guard.note_command_sent()

    def _check_can(self):
        """Report a stalled adapter; reopen the SDK port after its reset."""
        event = self._tx_guard.poll()
        if event == "stalled":
            rospy.logerr(
                "%s CAN TX STALLED: leader commands are sent but the adapter "
                "has transmitted nothing for %.1fs (tx_packets=%s; dmesg: "
                "gs_usb 'echo id'). Keep this launch running and reset the "
                "adapter: `%s`",
                self.can_port,
                self._tx_guard.stalled_for(),
                self._tx_guard.last_value,
                reset_command(self.can_port),
            )
            self._tx_stall_pub.publish(String(data=self.can_port))
        elif event == "resumed":
            rospy.logwarn("%s CAN transmission resumed", self.can_port)
        if not self._link_watch.poll():
            return
        # piper_sdk's read thread exits on the ENETDOWN of a down/up.
        with self._tx_lock:
            self.piper.DisconnectPort()
            self.piper.ConnectPort(can_init=True, piper_init=False)
            self._reset_command_trajectory()
        rospy.logwarn("%s CAN link was reset; SDK port reopened", self.can_port)

    @staticmethod
    def _message(position):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = position
        return message

    def _on_shutdown(self):
        self._shutting_down = True
        observer = getattr(self, "_raw_master_observer", None)
        if observer is not None:
            observer.close()
        if hasattr(self, "piper"):
            try:
                self.piper.DisconnectPort(thread_timeout=0.2)
            except Exception as exc:
                rospy.logwarn("Leader CAN close failed: %s", exc)

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            if self._role == "UNKNOWN" and self._state == "WAITING_FEEDBACK":
                self._probe_current_role()
            with self._lock:
                desired_mode = self._desired_mode
                home_active = self._home_active
                inference_active = self._inference_active
                leader_tx_enabled = self._leader_tx_enabled
                enable_requested = self._enable_requested
                emergency_stop_requested = self._emergency_stop_requested
                command = None if self._command is None else list(self._command)
                failed_role_request = self._failed_role_request
                idle_since = self._idle_since

            # A powered leader stays enabled briefly after IDLE so that a HOME
            # request whose home_active message lags the IDLE mode message
            # cannot drop the arm (see idle_disable_grace).
            idle_grace_pending = (
                desired_mode == "IDLE"
                and not home_active
                and self._output_enable_verified
                and time.monotonic() - idle_since < self.idle_disable_grace
            )

            # The inference client publishes one global enable flag for all
            # four arms. Do not let that flag turn a disabled leader on merely
            # because the mux is in HOLD: HOLD can be entered from IDLE when a
            # supervised preflight rejects the initial pose mismatch. Enabling
            # a loaded leader in that state caused the observed J5
            # overcurrent/stall protection. HOLD may only *retain* a leader
            # that was already safely enabled by INFERENCE/TAKEOVER.
            leader_enable_requested = enable_requested and (
                desired_mode in ("INFERENCE", "TAKEOVER")
                or home_active
                or (
                    desired_mode == "HOLD"
                    and self._output_enable_verified
                )
            )

            # IDLE is the neutral recovery role.  Always restore 0xFC there so
            # a previous interrupted/saved 0xFA cannot leave the arm silent and
            # permanently block the next inference request.
            # Both fixed collection and supervised TAKEOVER use the Piper
            # leader role.  The fixed-collection launch has separate read-only
            # readers; this dynamic path is used only by supervised TAKEOVER.
            desired_role = (
                "MASTER"
                if desired_mode in ("COLLECT", "TAKEOVER")
                else "OUTPUT"
            )
            if emergency_stop_requested:
                if self._state != "ESTOP":
                    try:
                        self._enter_emergency_stop()
                    except Exception as exc:
                        rospy.logerr("%s emergency stop send failed: %s", self.can_port, exc)
                        self._set_state("ERROR")
            elif self._state == "ESTOP":
                try:
                    self._recover_emergency_stop()
                except Exception as exc:
                    rospy.logerr("%s emergency stop recovery failed: %s", self.can_port, exc)
                    self._set_state("ERROR")
            elif idle_grace_pending:
                # Keep holding the last powered target; re-evaluated each loop.
                pass
            elif desired_mode == "IDLE" and not home_active:
                # IDLE always converges to 0xFC standby with motors disabled.
                # Role recovery must happen before the enable gate; otherwise
                # a persisted silent 0xFA state can never publish the feedback
                # that the mux requires before accepting C or I.
                try:
                    if self._drag_teach_active:
                        self._exit_drag_teach()
                    if self._role != "OUTPUT":
                        self._recover_output_feedback_role()
                    elif self._state != "DISABLED":
                        self._disable_and_wait()
                except Exception as exc:
                    rospy.logerr(
                        "%s IDLE recovery failed: %s",
                        self.can_port,
                        exc,
                    )
                    self._set_state("ERROR")
            elif not leader_enable_requested:
                if self._state != "DISABLED":
                    try:
                        if self._drag_teach_active:
                            self.piper.MotionCtrl_1(0x00, 0x00, 0x02)
                            self._drag_teach_active = False
                        self._disable_and_wait()
                    except Exception as exc:
                        rospy.logerr("%s disable failed: %s", self.can_port, exc)
                        self._set_state("ERROR")
            elif self._role == "UNKNOWN":
                # Recover a known OUTPUT feedback baseline first; the next
                # loop iteration performs the requested active-role switch.
                try:
                    self._recover_output_feedback_role()
                except Exception as exc:
                    rospy.logerr(
                        "%s active-mode feedback recovery failed: %s",
                        self.can_port,
                        exc,
                    )
                    self._set_state("ERROR")
            elif self._role != desired_role:
                if failed_role_request == (desired_mode, desired_role):
                    # The failed 0xFA request is latched until the mux moves
                    # out of TAKEOVER.  TEACH_FAILED makes the mux fail closed
                    # once without poisoning the already-restored OUTPUT role;
                    # HOLD/HOME can therefore recover on the next mode update.
                    self._set_state("TEACH_FAILED")
                elif self._state in (
                    "MASTER",
                    "OUTPUT",
                    "WAITING_FEEDBACK",
                    "WAITING_ENABLE",
                    "DISABLED",
                ) or (self._state == "ERROR" and desired_role == "OUTPUT"):
                    try:
                        if desired_role == "OUTPUT":
                            transition_ok = self._enter_output_role()
                        else:
                            transition_ok = self._enter_master_role(
                                allow_disable_fallback=desired_mode != "TAKEOVER"
                            )
                        if not transition_ok:
                            with self._lock:
                                self._failed_role_request = (
                                    desired_mode,
                                    desired_role,
                                )
                    except Exception as exc:
                        rospy.logerr("%s %s-role transition failed: %s", self.can_port, desired_role.lower(), exc)
                        with self._lock:
                            self._failed_role_request = (
                                desired_mode,
                                desired_role,
                            )
                        self._set_state("ERROR")
            elif self._drag_teach_active:
                try:
                    self._exit_drag_teach()
                except Exception as exc:
                    rospy.logerr(
                        "%s drag-teach exit failed: %s", self.can_port, exc
                    )
                    self._set_state("ERROR")
            elif self._state == "TEACH_FAILED":
                # The mux changes TAKEOVER to HOLD after a failed hand-off.
                # Resume publishing powered OUTPUT state so HOLD commands can
                # continue at the measured pose without another enable cycle.
                self._set_state("OUTPUT")
            elif (
                self._state != "ERROR"
                and leader_enable_requested
                and not self._output_enable_verified
            ):
                try:
                    if self._role == "MASTER":
                        # MASTER suppresses the low-speed verification stream;
                        # one enable retains the stream-verified teaching role.
                        self.piper.EnableArm(7)
                        self._output_enable_verified = True
                        self._set_state("MASTER")
                    elif self._arm_output_at_current_pose():
                        self._set_state(self._role)
                    else:
                        self._set_state("ERROR")
                except Exception as exc:
                    rospy.logerr(
                        "%s existing %s enable verification failed: %s",
                        self.can_port,
                        self._role,
                        exc,
                    )
                    self._set_state("ERROR")
            if (
                not emergency_stop_requested
                and desired_role == "OUTPUT"
                and self._role == "OUTPUT"
                and self._state == "OUTPUT"
                and command is not None
                and leader_tx_enabled
            ):
                try:
                    with self._tx_lock:
                        # Recheck after entering the send barrier. The stop
                        # callback sets this flag before waiting on _tx_lock.
                        with self._lock:
                            tx_still_enabled = self._leader_tx_enabled
                            latest_command = (
                                None
                                if self._command is None
                                else list(self._command)
                            )
                        if tx_still_enabled and latest_command is not None:
                            self._send_command(
                                latest_command,
                                home_active=home_active,
                                inference_active=inference_active,
                            )
                except Exception as exc:
                    rospy.logerr_throttle(
                        1.0, "%s leader command failed: %s", self.can_port, exc
                    )
                    self._set_state("ERROR")

            self._check_can()
            pose = self._read_pose()
            if pose is not None:
                self._feedback_pub.publish(self._message(pose))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("piper_leader_mode_driver")
    PiperLeaderModeDriver().run()
