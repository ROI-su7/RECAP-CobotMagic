#!/usr/bin/env python3
"""Read one fixed Piper teaching-input leader without writing to CAN.

Protocol 0x470 permits the linkage-control target to be 0x15x, 0x16x or
0x17x. This reader detects all three legal groups directly from SocketCAN.
That avoids a false ``0 Hz`` result when a controller retained a non-zero
target offset, and avoids the SDK parser's native-0x15x-only ID map.

The three joint frames are authoritative. The gripper frame is optional for
liveness because some stationary-gripper firmware variants do not repeat it.
"""

import math
import socket
import struct
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


MILLI_DEG_TO_RAD = math.pi / 180000.0
CAN_FRAME_FORMAT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)
CAN_ID_MASK = 0x1FFFFFFF
CONTROL_OFFSETS = (0x00, 0x10, 0x20)
JOINT_SUFFIXES = (0x05, 0x06, 0x07)
GRIPPER_SUFFIX = 0x09


class FixedMasterReader:
    def __init__(self):
        self.can_port = rospy.get_param("~can_port")
        self.publish_hz = float(rospy.get_param("~publish_hz", 200.0))
        self.startup_timeout = float(rospy.get_param("~startup_timeout_s", 5.0))
        self.stream_timeout = float(rospy.get_param("~stream_timeout_s", 0.15))
        # Bound on how long a silent bus is held at the last pose before the
        # link is declared dead.  Disabled by default (0.0) and it must stay
        # that way for normal collection: measured on this hardware, a
        # stationary leader emits nothing at all -- not even a heartbeat --
        # so an idle leader and an unplugged one are indistinguishable from
        # the bus alone, and any finite bound turns every operator pause into
        # a spurious command-withholding fault.  Set it only while actively
        # diagnosing a leader that is known to be moving.
        self.hold_timeout = float(rospy.get_param("~hold_timeout_s", 0.0))
        if (
            self.publish_hz <= 0.0
            or self.startup_timeout <= 0.0
            or self.stream_timeout <= 0.0
            or self.hold_timeout < 0.0
            or (0.0 < self.hold_timeout <= self.stream_timeout)
        ):
            raise ValueError(
                "reader rates and timeouts must be positive and a non-zero "
                "hold_timeout_s must exceed stream_timeout_s"
            )

        self._joint_pub = rospy.Publisher(
            "/master/joint_states", JointState, queue_size=1, tcp_nodelay=True
        )
        self._state_pub = rospy.Publisher(
            "/master/control_state", String, queue_size=1, latch=True
        )
        self._state = "STARTING"
        self._lock = threading.Lock()
        self._positions = {offset: [0.0] * 7 for offset in CONTROL_OFFSETS}
        self._seen_at = {offset: {} for offset in CONTROL_OFFSETS}
        self._counts = {offset: {} for offset in CONTROL_OFFSETS}
        self._active_offset = None
        self._last_pose = None
        self._last_frame_at = 0.0
        self._last_any_frame_at = 0.0
        self._opened_at = time.monotonic()
        self._running = True

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("Fixed MASTER raw reader opening %s", self.can_port)
        self._socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self._socket.settimeout(0.1)
        self._socket.bind((self.can_port,))
        self._thread = threading.Thread(
            target=self._read_loop,
            name="fixed-master-{}".format(self.can_port),
            daemon=True,
        )
        self._thread.start()
        self._wait_for_master_stream()

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self._state_pub.publish(String(data=state))
            rospy.logwarn("%s fixed leader state: %s", self.can_port, state)

    @staticmethod
    def _frame_ids(offset):
        return tuple(0x150 + offset + suffix for suffix in JOINT_SUFFIXES)

    @staticmethod
    def _gripper_id(offset):
        return 0x150 + offset + GRIPPER_SUFFIX

    def _read_loop(self):
        while self._running and not rospy.is_shutdown():
            try:
                frame = self._socket.recv(CAN_FRAME_SIZE)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running:
                    rospy.logerr("%s MASTER raw receive failed: %s", self.can_port, exc)
                return
            if len(frame) != CAN_FRAME_SIZE:
                continue
            can_id, dlc, data = struct.unpack(CAN_FRAME_FORMAT, frame)
            now = time.monotonic()
            # Liveness is tracked over every arbitration ID, not just the
            # control group: a stationary leader legitimately stops emitting
            # 0x15x targets while its other periodic frames keep coming, so
            # only a fully silent bus means the link itself is gone.
            with self._lock:
                self._last_any_frame_at = now
            self._process_frame(can_id & CAN_ID_MASK, bytes(data[:dlc]), now)

    def _process_frame(self, can_id, data, timestamp):
        for offset in CONTROL_OFFSETS:
            joint_ids = self._frame_ids(offset)
            if can_id in joint_ids and len(data) >= 8:
                first, second = struct.unpack(">ii", data[:8])
                pair = joint_ids.index(can_id) * 2
                with self._lock:
                    self._positions[offset][pair] = first * MILLI_DEG_TO_RAD
                    self._positions[offset][pair + 1] = second * MILLI_DEG_TO_RAD
                    self._seen_at[offset][can_id] = timestamp
                    self._counts[offset][can_id] = self._counts[offset].get(can_id, 0) + 1
                return
            if can_id == self._gripper_id(offset) and len(data) >= 4:
                gripper = struct.unpack(">i", data[:4])[0] / 1000000.0
                with self._lock:
                    self._positions[offset][6] = max(0.0, min(0.08, gripper))
                    self._seen_at[offset][can_id] = timestamp
                    self._counts[offset][can_id] = self._counts[offset].get(can_id, 0) + 1
                return

    def _sample(self, now, require_fresh=True):
        with self._lock:
            candidates = []
            for offset in CONTROL_OFFSETS:
                ids = self._frame_ids(offset)
                if not all(frame_id in self._seen_at[offset] for frame_id in ids):
                    continue
                oldest = min(self._seen_at[offset][frame_id] for frame_id in ids)
                if 0.0 <= now - oldest <= self.stream_timeout:
                    candidates.append((oldest, offset))
            if not candidates:
                # Piper teaching-input firmware may only emit 0x155-0x159
                # while a leader is being moved. Once a complete group has
                # proved the role, a stationary arm has no new target; retain
                # and republish its last measured pose instead of treating
                # normal quiescence as a CAN fault.
                return (
                    not require_fresh
                    and self._active_offset is not None
                    and self._last_pose is not None
                )
            _, offset = max(candidates)
            self._active_offset = offset
            self._last_pose = list(self._positions[offset])
            self._last_frame_at = now
            return True

    def _rate_text(self):
        elapsed = max(0.001, time.monotonic() - self._opened_at)
        with self._lock:
            parts = []
            for offset in CONTROL_OFFSETS:
                rates = [
                    self._counts[offset].get(frame_id, 0) / elapsed
                    for frame_id in self._frame_ids(offset)
                ]
                base = 0x150 + offset
                parts.append(
                    "0x{:03X}-0x{:03X}={:.1f}/{:.1f}/{:.1f}Hz".format(
                        base + 5, base + 7, rates[0], rates[1], rates[2]
                    )
                )
            return "; ".join(parts)

    def _wait_for_master_stream(self):
        deadline = time.monotonic() + self.startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._sample(time.monotonic(), require_fresh=True):
                self._set_state("MASTER")
                rospy.logwarn(
                    "%s fixed MASTER stream verified at control offset 0x%02X; "
                    "raw reader transmitted no CAN frame",
                    self.can_port,
                    self._active_offset,
                )
                return
            time.sleep(0.02)
        self._set_state("ERROR")
        raise RuntimeError(
            "{} has no complete fresh MASTER joint group; {}. "
            "Protocol allows only 0x155-0x157, 0x165-0x167 or 0x175-0x177.".format(
                self.can_port, self._rate_text()
            )
        )

    def _message(self):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = list(self._last_pose)
        return message

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            now = time.monotonic()
            with self._lock:
                last_any = self._last_any_frame_at
            hold_age = (
                now - self._last_frame_at if self._last_frame_at > 0.0 else 0.0
            )
            bus_age = now - last_any if last_any > 0.0 else 0.0
            if self.hold_timeout > 0.0 and bus_age > self.hold_timeout:
                # Only reachable when the operator opted in, because a
                # stationary leader is silent on this hardware.  Holding the
                # last pose indefinitely is the correct default: it keeps a
                # paused operator working, at the cost of not being able to
                # tell a paused leader from an unplugged one.
                self._set_state("STALE")
                rospy.logerr_throttle(
                    1.0,
                    "%s CAN bus fully silent for %.1fs (limit %.1fs); leader "
                    "command withheld -- check the leader's power and CAN "
                    "wiring, the USB adapter is still present (%s)",
                    self.can_port,
                    bus_age,
                    self.hold_timeout,
                    self._rate_text(),
                )
            elif self._sample(now, require_fresh=False):
                self._set_state(
                    "MASTER_HOLD" if hold_age > self.stream_timeout else "MASTER"
                )
                self._joint_pub.publish(self._message())
                gripper_id = self._gripper_id(self._active_offset)
                with self._lock:
                    gripper_seen = gripper_id in self._seen_at[self._active_offset]
                if not gripper_seen:
                    rospy.logwarn_throttle(
                        5.0,
                        "%s has valid MASTER joints but no gripper-control frame; "
                        "publishing gripper value 0 until one arrives",
                        self.can_port,
                    )
            else:
                self._set_state("STALE")
                rospy.logerr_throttle(
                    1.0,
                    "%s fixed MASTER stream is stale; no leader command published (%s)",
                    self.can_port,
                    self._rate_text(),
                )
            rate.sleep()

    def _shutdown(self):
        self._running = False
        if hasattr(self, "_socket"):
            try:
                self._socket.close()
            except OSError:
                pass
        if hasattr(self, "_thread"):
            self._thread.join(timeout=0.3)


if __name__ == "__main__":
    rospy.init_node("piper_fixed_master_reader")
    FixedMasterReader().run()
