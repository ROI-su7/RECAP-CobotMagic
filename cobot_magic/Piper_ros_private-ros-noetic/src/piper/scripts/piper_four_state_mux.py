#!/usr/bin/env python3
"""Inference/takeover command arbiter for four independent Piper arms.

This node is the only publisher to the two follower command topics. During
INFERENCE it also sends the same bounded policy target to both leader drivers.
TAKEOVER freezes policy/followers, waits for the operator hand-ready delay, then
uses each 0xFA leader's gravity-compensated MASTER feedback as the absolute
follower command source. Persistent 0xFA/0xFC roles are not changed at runtime.
"""

import math
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String


VALID_MODES = ("IDLE", "HOLD", "COLLECT", "INFERENCE", "TAKEOVER")
MODE_ALIASES = {
    "0": "IDLE",
    "1": "COLLECT",
    "2": "INFERENCE",
    "3": "TAKEOVER",
    "I": "INFERENCE",
    "T": "TAKEOVER",
}


class TimedJointState:
    def __init__(self):
        self.position = None
        self.received_at = 0.0

    def update(self, message):
        if len(message.position) < 7:
            rospy.logwarn_throttle(2.0, "Ignoring JointState with fewer than 7 positions")
            return
        position = [float(value) for value in message.position[:7]]
        if not all(math.isfinite(value) for value in position):
            rospy.logwarn_throttle(2.0, "Ignoring non-finite JointState")
            return
        self.position = position
        self.received_at = time.monotonic()

    def sample(self, now, timeout):
        if self.position is None:
            return None, False
        return list(self.position), 0.0 <= now - self.received_at <= timeout


class FourStateMux:
    def __init__(self):
        self.publish_hz = float(rospy.get_param("~publish_hz", 100.0))
        self.auto_start_inference = bool(
            rospy.get_param("~auto_start_inference", False)
        )
        self.auto_start_collect = bool(
            rospy.get_param("~auto_start_collect", False)
        )
        self.session_profile = str(
            rospy.get_param("~session_profile", "supervised")
        ).strip().lower()
        if self.session_profile not in ("supervised", "collect"):
            raise ValueError("~session_profile must be 'supervised' or 'collect'")
        if self.auto_start_inference and self.auto_start_collect:
            raise ValueError("only one automatic four-CAN mode may be selected")
        if self.session_profile == "collect" and self.auto_start_inference:
            raise ValueError("collect profile cannot auto-start INFERENCE")
        if self.session_profile == "supervised" and self.auto_start_collect:
            raise ValueError("supervised profile cannot auto-start COLLECT")
        self.input_timeout = float(rospy.get_param("~input_timeout_s", 0.15))
        # Policy generation is asynchronous and can pause at action-chunk
        # boundaries longer than 150 ms. Keep the stricter timeout for live
        # leader/follower feedback, but give policy commands their own bound.
        self.policy_input_timeout = float(
            rospy.get_param("~policy_input_timeout_s", 1.0)
        )
        self.max_joint_velocity = float(
            rospy.get_param("~max_joint_velocity_rad_s", 1.5)
        )
        self.max_gripper_velocity = float(
            rospy.get_param("~max_gripper_velocity_m_s", 0.15)
        )
        self.enforce_position_limits = bool(
            rospy.get_param("~enforce_position_limits", False)
        )
        self.joint_min = list(
            rospy.get_param(
                "~joint_min_rad",
                [-2.6179, 0.0, -2.9670, -1.7450, -1.2200, -2.09439],
            )
        )
        self.joint_max = list(
            rospy.get_param(
                "~joint_max_rad",
                [2.6179, 3.1400, 0.0, 1.7450, 1.2200, 2.09439],
            )
        )
        self.gripper_min = float(rospy.get_param("~gripper_min_m", 0.0))
        self.gripper_max = float(rospy.get_param("~gripper_max_m", 0.08))
        self.takeover_joint_tolerance = float(
            rospy.get_param("~takeover_joint_tolerance_rad", 0.04)
        )
        self.takeover_gripper_tolerance = float(
            rospy.get_param("~takeover_gripper_tolerance_m", 0.005)
        )
        self.takeover_stable_samples = int(
            rospy.get_param("~takeover_stable_samples", 20)
        )
        self.takeover_alignment_timeout = float(
            rospy.get_param("~takeover_alignment_timeout_s", 1.0)
        )
        self.takeover_release_delay = float(
            rospy.get_param("~takeover_release_delay_s", 1.0)
        )
        # After T, OpenPI receives /teleop/hold_active and clears its pending
        # action chunk.  Require an observable quiet interval on both policy
        # topics before requesting 0xFA; late ROS frames are still ignored by
        # HOLD, but this gate makes the ownership ordering explicit.
        self.takeover_policy_quiet = float(
            rospy.get_param("~takeover_policy_quiet_s", 0.20)
        )
        # The OpenPI executor publishes an explicit acknowledgement only after
        # its action epoch has been stopped and both ROS command publishers are
        # outside an in-flight publish call.  Do not infer that solely from a
        # quiet input topic: a callback or TCPROS frame can still be queued.
        self.takeover_policy_ack_timeout = float(
            rospy.get_param("~takeover_policy_ack_timeout_s", 2.0)
        )
        # After the policy ACK, close the OUTPUT-command gate in both leader
        # drivers.  Each driver ACKs after any in-flight CAN command completes;
        # retain this additional interval for the SocketCAN TX queue to drain
        # before sending the 0xFA leader-role frame.
        self.takeover_leader_tx_quiet = float(
            rospy.get_param("~takeover_leader_tx_quiet_s", 0.20)
        )
        self.takeover_role_timeout = float(
            rospy.get_param("~takeover_role_timeout_s", 15.0)
        )
        # COLLECT also changes both leaders from OUTPUT into the 0xFA MASTER
        # role.  Do not expose leader commands while that blocking CAN role
        # transition temporarily pauses joint feedback.
        self.collect_role_timeout = float(
            rospy.get_param("~collect_role_timeout_s", 15.0)
        )
        self.inference_start_timeout = float(
            rospy.get_param("~inference_start_timeout_s", 10.0)
        )
        self.home_timeout = float(rospy.get_param("~home_timeout_s", 15.0))
        self.home_stable_samples = int(rospy.get_param("~home_stable_samples", 20))
        self.home_max_joint_velocity = float(
            rospy.get_param("~home_max_joint_velocity_rad_s", 0.25)
        )
        self.home_max_gripper_velocity = float(
            rospy.get_param("~home_max_gripper_velocity_m_s", 0.03)
        )
        # Optional HOME release: open both follower grippers in place for
        # home_release_s so a held object is let go, keep them open while the
        # arms move HOME, then close to the captured HOME gripper value.
        # home_release_gripper_m = 0 keeps the original HOME behaviour.
        self.home_release_gripper = float(
            rospy.get_param("~home_release_gripper_m", 0.0)
        )
        self.home_release_duration = float(rospy.get_param("~home_release_s", 1.5))
        # Supervised inference must not use motor torque to correct a large
        # launch-time leader/follower mismatch.  Large deltas previously made
        # a loaded leader J5 enter overcurrent/stall protection before policy
        # commands were released.
        self.supervised_start_joint_tolerance = float(
            rospy.get_param("~supervised_start_joint_tolerance_rad", 0.08)
        )
        self.supervised_start_gripper_tolerance = float(
            rospy.get_param("~supervised_start_gripper_tolerance_m", 0.015)
        )
        if (
            self.publish_hz <= 0.0
            or self.input_timeout <= 0.0
            or self.policy_input_timeout <= 0.0
        ):
            raise ValueError("publish_hz and input timeouts must be positive")
        if (
            self.takeover_joint_tolerance <= 0.0
            or self.takeover_gripper_tolerance <= 0.0
            or self.takeover_stable_samples <= 0
            or self.takeover_alignment_timeout <= 0.0
            or self.takeover_release_delay < 0.0
            or self.takeover_policy_quiet < 0.0
            or self.takeover_policy_ack_timeout <= 0.0
            or self.takeover_leader_tx_quiet < 0.0
            or self.takeover_role_timeout <= 0.0
            or self.collect_role_timeout <= 0.0
            or self.inference_start_timeout <= 0.0
            or self.home_timeout <= 0.0
            or self.home_stable_samples <= 0
            or self.home_max_joint_velocity <= 0.0
            or self.home_max_gripper_velocity <= 0.0
            or self.home_release_gripper < 0.0
            or self.home_release_duration < 0.0
            or self.supervised_start_joint_tolerance <= 0.0
            or self.supervised_start_gripper_tolerance <= 0.0
        ):
            raise ValueError("takeover tolerances and stable samples must be positive")
        if len(self.joint_min) != 6 or len(self.joint_max) != 6:
            raise ValueError("joint_min_rad and joint_max_rad must each contain 6 values")

        self._lock = threading.RLock()
        self._mode = "IDLE"
        self._left = {
            "leader": TimedJointState(),
            "policy": TimedJointState(),
            "follower": TimedJointState(),
        }
        self._right = {
            "leader": TimedJointState(),
            "policy": TimedJointState(),
            "follower": TimedJointState(),
        }
        self._last_output = {"left": None, "right": None}
        self._hold_target = {"left": None, "right": None}
        self._leader_state = {"left": "UNKNOWN", "right": "UNKNOWN"}
        self._takeover_alignment_count = 0
        self._takeover_ready = False
        self._takeover_pending = False
        self._takeover_pending_since = 0.0
        self._takeover_phase = "IDLE"
        self._takeover_role_started_at = 0.0
        self._takeover_alignment_latched = False
        self._policy_hold_ack = False
        self._policy_hold_ack_received_at = 0.0
        self._leader_tx_enabled = False
        self._leader_tx_stop_requested_at = 0.0
        self._leader_tx_quiet_since = 0.0
        self._leader_tx_quiet_ack = {"left": False, "right": False}
        self._leader_tx_quiet_ack_at = {"left": 0.0, "right": 0.0}
        self._collect_phase = "IDLE"
        self._collect_role_started_at = 0.0
        self._inference_phase = "IDLE"
        self._inference_started_at = 0.0
        self._inference_active = False
        # A HOLD -> I resume is not active until both policy messages are
        # newer than the I request.  This prevents replay of the action chunk
        # that was current when Space was pressed.
        self._policy_resume_after = 0.0
        self._enable_requested = False
        self._auto_start_consumed = False
        self._emergency_stop_active = False
        self._hold_active = False
        self._home_active = False
        self._home_phase = "IDLE"
        self._home_release_until = 0.0
        self._home_gripper_open = False
        self._home_started_at = 0.0
        self._home_stable_count = 0
        self._home_follower_target = {"left": None, "right": None}
        self._home_leader_target = {"left": None, "right": None}
        self._last_publish_time = time.monotonic()

        self._left_command_pub = rospy.Publisher(
            "/teleop/command/joint_left", JointState, queue_size=1, tcp_nodelay=True
        )
        self._right_command_pub = rospy.Publisher(
            "/teleop/command/joint_right", JointState, queue_size=1, tcp_nodelay=True
        )
        self._left_leader_command_pub = rospy.Publisher(
            "/teleop/leader_command/joint_left",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._right_leader_command_pub = rospy.Publisher(
            "/teleop/leader_command/joint_right",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        # Compatibility output for the existing collection tools.  Follower
        # drivers subscribe to /teleop/command/*, so this cannot form a loop.
        self._left_action_pub = rospy.Publisher(
            "/master/joint_left", JointState, queue_size=1, tcp_nodelay=True
        )
        self._right_action_pub = rospy.Publisher(
            "/master/joint_right", JointState, queue_size=1, tcp_nodelay=True
        )
        self._mode_pub = rospy.Publisher(
            "/teleop/mode", String, queue_size=1, latch=True
        )
        self._takeover_ready_pub = rospy.Publisher(
            "/teleop/takeover_ready", Bool, queue_size=1, latch=True
        )
        self._takeover_phase_pub = rospy.Publisher(
            "/teleop/takeover_phase", String, queue_size=1, latch=True
        )
        self._inference_active_pub = rospy.Publisher(
            "/teleop/inference_active", Bool, queue_size=1, latch=True
        )
        self._home_active_pub = rospy.Publisher(
            "/teleop/home_active", Bool, queue_size=1, latch=True
        )
        self._emergency_stop_pub = rospy.Publisher(
            "/teleop/emergency_stop", Bool, queue_size=1, latch=True
        )
        self._hold_active_pub = rospy.Publisher(
            "/teleop/hold_active", Bool, queue_size=1, latch=True
        )
        self._leader_tx_enable_pub = rospy.Publisher(
            "/teleop/leader_tx_enable", Bool, queue_size=1, latch=True
        )

        subscriptions = (
            ("/teleop/leader/joint_left", self._left["leader"]),
            ("/teleop/leader/joint_right", self._right["leader"]),
            ("/teleop/policy/joint_left", self._left["policy"]),
            ("/teleop/policy/joint_right", self._right["policy"]),
            ("/puppet/joint_left", self._left["follower"]),
            ("/puppet/joint_right", self._right["follower"]),
        )
        self._subscribers = [
            rospy.Subscriber(
                topic,
                JointState,
                self._joint_callback,
                callback_args=storage,
                queue_size=1,
                tcp_nodelay=True,
            )
            for topic, storage in subscriptions
        ]
        self._mode_subscriber = rospy.Subscriber(
            "/teleop/mode_request", String, self._mode_callback, queue_size=10
        )
        self._enable_subscriber = rospy.Subscriber(
            "/enable_flag", Bool, self._enable_callback, queue_size=1
        )
        self._leader_state_subscribers = [
            rospy.Subscriber(
                "/teleop/leader/control_state_left",
                String,
                self._leader_state_callback,
                callback_args="left",
                queue_size=1,
            ),
            rospy.Subscriber(
                "/teleop/leader/control_state_right",
                String,
                self._leader_state_callback,
                callback_args="right",
                queue_size=1,
            ),
        ]
        self._home_subscriber = rospy.Subscriber(
            "/teleop/home_request", Empty, self._home_callback, queue_size=1
        )
        self._emergency_stop_subscriber = rospy.Subscriber(
            "/teleop/emergency_stop_request",
            Empty,
            self._emergency_stop_callback,
            queue_size=1,
        )
        self._hold_subscriber = rospy.Subscriber(
            "/teleop/hold_request", Empty, self._hold_callback, queue_size=1
        )
        # Published by a Piper driver whose CAN adapter stopped transmitting.
        self._can_tx_stalled_subscriber = rospy.Subscriber(
            "/teleop/can_tx_stalled",
            String,
            self._can_tx_stalled_callback,
            queue_size=10,
        )
        self._policy_hold_ack_subscriber = rospy.Subscriber(
            "/teleop/policy_hold_ack",
            Bool,
            self._policy_hold_ack_callback,
            queue_size=1,
        )
        self._leader_tx_quiet_subscribers = [
            rospy.Subscriber(
                "/teleop/leader/tx_quiet_left",
                Bool,
                self._leader_tx_quiet_callback,
                callback_args="left",
                queue_size=1,
            ),
            rospy.Subscriber(
                "/teleop/leader/tx_quiet_right",
                Bool,
                self._leader_tx_quiet_callback,
                callback_args="right",
                queue_size=1,
            ),
        ]
        self._mode_pub.publish(String(data=self._mode))
        self._takeover_ready_pub.publish(Bool(data=False))
        self._takeover_phase_pub.publish(String(data=self._takeover_phase))
        self._inference_active_pub.publish(Bool(data=False))
        self._home_active_pub.publish(Bool(data=False))
        self._emergency_stop_pub.publish(Bool(data=False))
        self._hold_active_pub.publish(Bool(data=False))
        self._leader_tx_enable_pub.publish(Bool(data=False))
        if self.auto_start_inference:
            rospy.loginfo(
                "Supervised inference mux armed for one automatic INFERENCE start"
            )
        elif self.auto_start_collect:
            rospy.loginfo(
                "Four-CAN collection mux armed for one automatic COLLECT start"
            )
        else:
            rospy.loginfo("Inference/takeover Piper mux started in IDLE")

    def _joint_callback(self, message, storage):
        with self._lock:
            storage.update(message)

    def _leader_state_callback(self, message, side):
        with self._lock:
            self._leader_state[side] = message.data.strip().upper()

    def _policy_hold_ack_callback(self, message):
        """Record the executor-side proof that no old policy action can publish."""
        with self._lock:
            self._policy_hold_ack = bool(message.data)
            self._policy_hold_ack_received_at = time.monotonic()

    def _leader_tx_quiet_callback(self, message, side):
        """Record a leader driver's post-send-barrier acknowledgement."""
        with self._lock:
            self._leader_tx_quiet_ack[side] = bool(message.data)
            self._leader_tx_quiet_ack_at[side] = time.monotonic()

    def _set_leader_tx_enabled(self, enabled, force=False):
        """Open/close cached OUTPUT command transmission in both leaders."""
        enabled = bool(enabled)
        if enabled == self._leader_tx_enabled and not force:
            return
        self._leader_tx_enabled = enabled
        if not enabled:
            self._leader_tx_stop_requested_at = time.monotonic()
            self._leader_tx_quiet_since = 0.0
            self._leader_tx_quiet_ack = {"left": False, "right": False}
            self._leader_tx_quiet_ack_at = {"left": 0.0, "right": 0.0}
        else:
            self._leader_tx_stop_requested_at = 0.0
            self._leader_tx_quiet_since = 0.0
        self._leader_tx_enable_pub.publish(Bool(data=enabled))

    def _set_takeover_phase(self, phase):
        phase = str(phase).strip().upper()
        if phase == self._takeover_phase:
            return
        self._takeover_phase = phase
        self._takeover_phase_pub.publish(String(data=phase))

    def _emergency_stop_callback(self, _message):
        now = time.monotonic()
        with self._lock:
            if self._emergency_stop_active:
                rospy.logwarn("QUICK EMERGENCY STOP is already active")
                return
            previous = self._mode
            self._auto_start_consumed = True
            self._hold_target["left"] = self._current_anchor("left", now)
            self._hold_target["right"] = self._current_anchor("right", now)
            self._mode = "IDLE"
            self._takeover_pending = False
            self._set_takeover_phase("IDLE")
            self._collect_phase = "IDLE"
            self._inference_phase = "IDLE"
            self._set_inference_active(False)
            self._set_takeover_ready(False)
            if self._home_active:
                self._finish_home(False, "HOME interrupted by emergency stop")
            self._emergency_stop_active = True
            self._hold_active = False
            self._set_leader_tx_enabled(False)
            self._mode_pub.publish(String(data=self._mode))
            self._emergency_stop_pub.publish(Bool(data=True))
            self._hold_active_pub.publish(Bool(data=False))
            rospy.logerr(
                "QUICK EMERGENCY STOP: %s -> IDLE; all arm commands inhibited. "
                "Press h to reset and return HOME",
                previous,
            )

    def _can_tx_stalled_callback(self, message):
        """Stop every arm when one arm's CAN adapter no longer transmits.

        That arm silently holds its last target, so continuing would move the
        other arms alone (e.g. pull the cable) and HOME would never finish.
        """
        port = message.data.strip()
        if not port:
            return
        now = time.monotonic()
        reason = (
            "{0} stopped transmitting CAN frames; that arm only holds its last "
            "target. Keep the launch running, run `sudo ip link set {0} down "
            "&& sudo ip link set {0} up` (the drivers reopen the port), then "
            "resume with I or H".format(port)
        )
        with self._lock:
            if self._home_active:
                self._finish_home(False, "HOME aborted: " + reason)
            elif self._mode in ("IDLE", "HOLD"):
                rospy.logerr("%s", reason)
            else:
                self._fail_closed(reason, now)

    def _hold_callback(self, _message):
        now = time.monotonic()
        with self._lock:
            self._enter_hold("operator software HOLD", now)

    def _enter_hold(self, reason, now):
        """Freeze measured follower poses without sending Piper 0x150.

        HOLD intentionally keeps the global enable request intact. Followers
        and OUTPUT leaders continue receiving bounded position targets, so an
        operator pause cannot turn an unbraked Piper into damping mode.
        """
        if self._emergency_stop_active:
            rospy.logerr("Cannot enter software HOLD while hardware E-STOP is active")
            return
        previous = self._mode
        for side in ("left", "right"):
            anchor = self._current_anchor(side, now)
            if anchor is not None:
                self._hold_target[side] = anchor
        self._mode = "HOLD"
        self._hold_active = True
        self._takeover_pending = False
        self._takeover_alignment_latched = False
        self._set_takeover_phase("HOLD")
        self._collect_phase = "IDLE"
        self._inference_phase = "IDLE"
        self._policy_resume_after = 0.0
        self._set_inference_active(False)
        self._set_takeover_ready(False)
        self._set_leader_tx_enabled(True)
        if self._home_active:
            self._finish_home(False, "HOME interrupted by software HOLD")
        self._mode_pub.publish(String(data=self._mode))
        self._hold_active_pub.publish(Bool(data=True))
        rospy.logwarn(
            "%s: %s -> HOLD; all available arms remain enabled and position-held",
            reason,
            previous,
        )

    def _home_callback(self, _message):
        now = time.monotonic()
        with self._lock:
            self._auto_start_consumed = True
            if any(
                self._home_follower_target[side] is None
                or self._home_leader_target[side] is None
                for side in ("left", "right")
            ):
                rospy.logerr("Cannot return HOME: startup poses are not captured yet")
                return
            self._hold_target["left"] = self._current_anchor("left", now)
            self._hold_target["right"] = self._current_anchor("right", now)
            if self._hold_target["left"] is None or self._hold_target["right"] is None:
                rospy.logerr("Cannot return HOME: follower feedback is unavailable")
                return
            previous = self._mode
            self._mode = "IDLE"
            self._hold_active = False
            self._takeover_pending = False
            self._set_takeover_phase("IDLE")
            self._collect_phase = "IDLE"
            self._inference_phase = "IDLE"
            self._set_inference_active(False)
            if self._emergency_stop_active:
                self._emergency_stop_active = False
                self._emergency_stop_pub.publish(Bool(data=False))
                rospy.logwarn("Emergency stop reset requested by HOME")
            self._home_active = True
            self._home_phase = "SWITCHING"
            self._home_started_at = now
            self._home_stable_count = 0
            self._home_gripper_open = False
            # Announce HOME before IDLE: a leader driver that sees IDLE without
            # home_active disables its motors, and Piper has no brakes.
            self._home_active_pub.publish(Bool(data=True))
            self._mode_pub.publish(String(data=self._mode))
            self._set_leader_tx_enabled(True)
            self._hold_active_pub.publish(Bool(data=False))
            rospy.logwarn("HOME requested: %s -> IDLE; preparing all four arms", previous)

    def _mode_callback(self, message):
        requested = message.data.strip().upper()
        requested = MODE_ALIASES.get(requested, requested)
        rospy.logwarn(
            "Mode request received: %s (current=%s, profile=%s)",
            requested,
            self._mode,
            self.session_profile,
        )
        if requested not in VALID_MODES:
            rospy.logwarn("Rejected unknown teleop mode %r", message.data)
            return
        allowed = (
            ("IDLE", "COLLECT")
            if self.session_profile == "collect"
            else ("IDLE", "HOLD", "INFERENCE", "TAKEOVER")
        )
        if requested not in allowed:
            rospy.logwarn(
                "Rejected %s in %s command-line profile",
                requested,
                self.session_profile,
            )
            return
        if requested == "IDLE":
            with self._lock:
                self._auto_start_consumed = True
        self._set_mode(requested)

    def _enable_callback(self, message):
        requested = bool(message.data)
        with self._lock:
            if requested == self._enable_requested:
                return
            self._enable_requested = requested
            if not requested:
                self._auto_start_consumed = True
                if self._mode != "IDLE":
                    self._set_mode("IDLE")
                rospy.logwarn(
                    "%s command output disarmed by enable=false",
                    self.session_profile,
                )
            elif not self._auto_start_consumed:
                if self.auto_start_inference:
                    rospy.logwarn(
                        "Supervised inference enable received; waiting for policy and feedback"
                    )
                elif self.auto_start_collect:
                    rospy.logwarn(
                        "Four-CAN collection enable received; waiting for leader/follower feedback"
                    )

    def _process_auto_start(self, now):
        if (
            not (self.auto_start_inference or self.auto_start_collect)
            or self._auto_start_consumed
            or not self._enable_requested
            or self._mode != "IDLE"
            or self._emergency_stop_active
            or self._home_active
        ):
            return
        _, _, policy_fresh = self._fresh_pair("policy", now)
        _, _, leaders_fresh = self._fresh_pair("leader", now)
        _, _, followers_fresh = self._fresh_pair("follower", now)
        if not followers_fresh:
            return
        if self.auto_start_inference and not policy_fresh:
            return
        if self.auto_start_inference and not self._supervised_start_preflight(now):
            return
        if self.auto_start_collect and not leaders_fresh:
            return
        self._auto_start_consumed = True
        if self.auto_start_inference:
            rospy.logwarn(
                "Command-line supervised profile has fresh inputs; requesting INFERENCE"
            )
            self._set_mode("INFERENCE")
        else:
            rospy.logwarn(
                "Command-line collection profile has fresh inputs; requesting COLLECT"
            )
            self._set_mode("COLLECT")

    def _fresh_pair(self, source, now):
        timeout = (
            self.policy_input_timeout if source == "policy" else self.input_timeout
        )
        left, left_fresh = self._left[source].sample(now, timeout)
        right, right_fresh = self._right[source].sample(now, timeout)
        return left, right, left_fresh and right_fresh

    def _source_age_text(self, source, now):
        ages = []
        for side, storage in (("left", self._left), ("right", self._right)):
            received_at = storage[source].received_at
            age = float("inf") if received_at <= 0.0 else max(0.0, now - received_at)
            ages.append("{}={:.3f}s".format(side, age))
        return ", ".join(ages)

    def _current_anchor(self, side, now):
        storage = self._left if side == "left" else self._right
        follower, follower_fresh = storage["follower"].sample(now, self.input_timeout)
        if follower_fresh:
            return follower
        if self._last_output[side] is not None:
            return list(self._last_output[side])
        return None

    def _set_mode(self, requested):
        now = time.monotonic()
        with self._lock:
            if requested == "TAKEOVER" and self._takeover_pending:
                rospy.logwarn(
                    "TAKEOVER is already pending; followers remain in powered "
                    "HOLD while the %.2fs hand-ready delay completes",
                    self.takeover_release_delay,
                )
                return
            if requested == "HOLD":
                self._enter_hold("mode request", now)
                return
            if self._emergency_stop_active:
                rospy.logerr(
                    "Cannot enter %s while emergency stop is active; press h to reset and return HOME",
                    requested,
                )
                return
            if self._home_active:
                self._finish_home(False, "HOME cancelled by mode request")
            if requested == self._mode:
                if requested == "INFERENCE" and self._takeover_pending:
                    self._takeover_pending = False
                    rospy.logwarn("Pending TAKEOVER cancelled; continuing INFERENCE")
                return
            if requested in ("COLLECT", "TAKEOVER"):
                left_leader, right_leader, fresh = self._fresh_pair("leader", now)
                if not fresh:
                    rospy.logerr("Cannot enter %s: leader input is incomplete or stale", requested)
                    return
            if requested == "INFERENCE":
                _, _, fresh = self._fresh_pair("policy", now)
                resuming_from_hold = self._mode == "HOLD"
                if not fresh and not resuming_from_hold:
                    rospy.logerr("Cannot enter INFERENCE: policy input is incomplete or stale")
                    return
                if not resuming_from_hold and not self._supervised_start_preflight(now):
                    return

            left_anchor = self._current_anchor("left", now)
            right_anchor = self._current_anchor("right", now)
            if requested != "IDLE" and (left_anchor is None or right_anchor is None):
                rospy.logerr("Cannot enter %s: follower feedback is unavailable", requested)
                return

            if requested == "TAKEOVER":
                if self._mode != "INFERENCE" or self._inference_phase != "ACTIVE":
                    rospy.logerr(
                        "TAKEOVER is only valid while supervised INFERENCE is active "
                        "(mode=%s phase=%s)",
                        self._mode,
                        self._inference_phase,
                    )
                    return
                allowed_states = ("OUTPUT", "SWITCHING_OUTPUT")
                if any(
                    state not in allowed_states for state in self._leader_state.values()
                ):
                    rospy.logerr(
                        "Cannot enter TAKEOVER: leader roles are left=%s right=%s",
                        self._leader_state["left"],
                        self._leader_state["right"],
                    )
                    return
                # T is a real policy stop, not merely a delayed ownership
                # request. Freeze the measured follower poses, announce HOLD
                # to the policy client so its queued action chunk is discarded,
                # and keep both leaders in powered OUTPUT position hold while
                # the solo operator puts both hands on them. After the delay,
                # the leader drivers request the firmware's seamless 0xFA
                # zero-force leader role without disabling either arm.
                self._takeover_pending = True
                self._takeover_pending_since = now
                self._takeover_alignment_latched = False
                # A Boolean ACK is accepted only when it is received after this
                # T request.  Resetting the local latch prevents a stale ACK
                # from an earlier HOLD from opening the role-switch gate.
                self._policy_hold_ack = False
                self._policy_hold_ack_received_at = 0.0
                self._hold_target["left"] = left_anchor
                self._hold_target["right"] = right_anchor
                self._mode = "HOLD"
                self._hold_active = True
                self._inference_phase = "IDLE"
                self._set_takeover_phase("POLICY_STOP_REQUEST")
                self._set_inference_active(False)
                self._set_leader_tx_enabled(True)
                self._mode_pub.publish(String(data=self._mode))
                self._hold_active_pub.publish(Bool(data=True))
                rospy.logwarn(
                    "TAKEOVER accepted: INFERENCE -> powered HOLD immediately; "
                    "policy stop requested; waiting for executor ACK, leader "
                    "TX drain, and %.2fs hand-ready delay before 0xFA MASTER",
                    self.takeover_release_delay,
                )
                return
            if requested == "IDLE":
                self._hold_target["left"] = left_anchor
                self._hold_target["right"] = right_anchor

            previous = self._mode
            self._mode = requested
            self._hold_active = False
            self._hold_active_pub.publish(Bool(data=False))
            self._takeover_pending = False
            self._takeover_alignment_latched = False
            self._set_takeover_phase("IDLE")
            if requested == "COLLECT":
                self._set_leader_tx_enabled(False)
                self._collect_phase = "SWITCHING_MASTER"
                self._collect_role_started_at = now
                self._hold_target["left"] = left_anchor
                self._hold_target["right"] = right_anchor
            else:
                self._collect_phase = "IDLE"
            if requested == "INFERENCE":
                self._set_leader_tx_enabled(True)
                self._inference_phase = "SWITCHING_OUTPUT"
                self._inference_started_at = now
                self._policy_resume_after = now if previous == "HOLD" else 0.0
                self._set_inference_active(False)
                self._hold_target["left"] = left_anchor
                self._hold_target["right"] = right_anchor
            else:
                self._inference_phase = "IDLE"
                self._policy_resume_after = 0.0
                self._set_inference_active(False)
                if requested == "IDLE":
                    self._set_leader_tx_enabled(False)
            if requested != "INFERENCE":
                self._takeover_alignment_count = 0
            self._mode_pub.publish(String(data=self._mode))
            if requested == "COLLECT":
                rospy.logwarn(
                    "COLLECT requested: %s -> COLLECT; holding followers "
                    "until both leaders confirm MASTER",
                    previous,
                )
            elif requested == "INFERENCE":
                rospy.logwarn(
                    "INFERENCE requested: %s -> INFERENCE; holding followers "
                    "until both leaders confirm OUTPUT alignment",
                    previous,
                )
            else:
                rospy.logwarn("Teleop ownership changed: %s -> %s", previous, requested)

    def _fail_closed(self, reason, now):
        if self._mode in ("IDLE", "HOLD"):
            return
        self._enter_hold(reason, now)

    def _target_for_side(self, side, now):
        storage = self._left if side == "left" else self._right
        if self._mode in ("IDLE", "HOLD"):
            if self._home_active and self._home_phase == "RELEASING":
                return self._with_gripper(
                    self._hold_target[side], self.home_release_gripper
                )
            if self._home_active and self._home_phase == "MOVING":
                if self._home_gripper_open:
                    return self._with_gripper(
                        self._home_follower_target[side], self.home_release_gripper
                    )
                return self._home_follower_target[side]
            if self._hold_target[side] is None:
                self._hold_target[side] = self._current_anchor(side, now)
            return self._hold_target[side]

        if self._mode == "INFERENCE" and (
            self._takeover_pending or self._inference_phase != "ACTIVE"
        ):
            return self._hold_target[side]

        if self._mode == "COLLECT" and self._collect_phase != "ACTIVE":
            return self._hold_target[side]

        if self._mode == "TAKEOVER" and self._takeover_phase != "ACTIVE":
            return self._hold_target[side]

        source = "policy" if self._mode == "INFERENCE" else "leader"
        timeout = (
            self.policy_input_timeout if source == "policy" else self.input_timeout
        )
        position, fresh = storage[source].sample(now, timeout)
        if not fresh:
            return None
        return position

    def _takeover_alignment_is_safe(self, now):
        """Return whether both live leader/follower pairs are aligned."""
        left_leader, right_leader, leaders_fresh = self._fresh_pair("leader", now)
        left_follower, right_follower, followers_fresh = self._fresh_pair(
            "follower", now
        )
        roles_ready = all(state == "OUTPUT" for state in self._leader_state.values())
        if not leaders_fresh or not followers_fresh or not roles_ready:
            return False
        for leader, follower in (
            (left_leader, left_follower),
            (right_leader, right_follower),
        ):
            if (
                max(abs(a - b) for a, b in zip(leader[:6], follower[:6]))
                > self.takeover_joint_tolerance
            ):
                return False
            if abs(leader[6] - follower[6]) > self.takeover_gripper_tolerance:
                return False
        return True

    def _update_takeover_alignment(self, now):
        alignment_phase = self._mode == "INFERENCE" or (
            self._mode == "HOLD" and self._takeover_pending
        )
        if not alignment_phase:
            self._takeover_alignment_count = 0
            self._set_takeover_ready(False)
            return
        if self._takeover_alignment_is_safe(now):
            self._takeover_alignment_count = min(
                self.takeover_stable_samples, self._takeover_alignment_count + 1
            )
        else:
            self._takeover_alignment_count = 0
        self._set_takeover_ready(
            self._takeover_alignment_count >= self.takeover_stable_samples
        )

    def _set_takeover_ready(self, ready):
        ready = bool(ready)
        if ready == self._takeover_ready:
            return
        self._takeover_ready = ready
        self._takeover_ready_pub.publish(Bool(data=ready))
        if ready:
            rospy.loginfo("Leader/follower poses aligned for immediate TAKEOVER")

    def _set_inference_active(self, active):
        active = bool(active)
        if active == self._inference_active:
            return
        self._inference_active = active
        self._inference_active_pub.publish(Bool(data=active))

    def _supervised_start_preflight(self, now):
        """Require near-aligned physical poses before leader motors energize."""
        left_leader, right_leader, leaders_fresh = self._fresh_pair("leader", now)
        left_follower, right_follower, followers_fresh = self._fresh_pair(
            "follower", now
        )
        if not leaders_fresh or not followers_fresh:
            rospy.logwarn_throttle(
                2.0,
                "Supervised inference waiting for fresh leader/follower preflight streams",
            )
            return False
        deltas = {}
        safe = True
        for side, leader, follower in (
            ("left", left_leader, left_follower),
            ("right", right_leader, right_follower),
        ):
            joint_deltas = [follower[i] - leader[i] for i in range(6)]
            gripper_delta = follower[6] - leader[6]
            deltas[side] = (joint_deltas, gripper_delta)
            if (
                max(abs(value) for value in joint_deltas)
                > self.supervised_start_joint_tolerance
                or abs(gripper_delta) > self.supervised_start_gripper_tolerance
            ):
                safe = False
        if not safe:
            rospy.logerr_throttle(
                2.0,
                "Supervised inference NOT enabled: launch-time leader->follower "
                "delta exceeds the safe pre-alignment gate (left J=%s G=%+.4f; "
                "right J=%s G=%+.4f). First use the working collection profile "
                "to place each follower at its leader pose, then stop it without "
                "moving the arms and start supervised inference.",
                "[{}]".format(
                    ", ".join("{:+.4f}".format(v) for v in deltas["left"][0])
                ),
                deltas["left"][1],
                "[{}]".format(
                    ", ".join("{:+.4f}".format(v) for v in deltas["right"][0])
                ),
                deltas["right"][1],
            )
            return False
        return True

    def _capture_home(self, now):
        if all(
            self._home_follower_target[side] is not None
            and self._home_leader_target[side] is not None
            for side in ("left", "right")
        ):
            return
        left_follower, right_follower, followers_fresh = self._fresh_pair(
            "follower", now
        )
        left_leader, right_leader, leaders_fresh = self._fresh_pair("leader", now)
        if not followers_fresh or not leaders_fresh:
            return
        self._home_follower_target["left"] = left_follower
        self._home_follower_target["right"] = right_follower
        self._home_leader_target["left"] = left_leader
        self._home_leader_target["right"] = right_leader
        rospy.loginfo("Captured startup poses for HOME")

    @staticmethod
    def _with_gripper(position, gripper):
        """Copy a 7-D joint target with a different gripper opening."""
        if position is None:
            return None
        return list(position[:6]) + [gripper]

    def _finish_home(self, success, message):
        release_pending = self._home_active and self.home_release_gripper > 0.0
        self._home_active = False
        self._home_phase = "IDLE"
        self._home_gripper_open = False
        self._home_stable_count = 0
        self._hold_target["left"] = self._current_anchor("left", time.monotonic())
        self._hold_target["right"] = self._current_anchor("right", time.monotonic())
        if release_pending:
            # Even when HOME times out, never leave released grippers open.
            for side in ("left", "right"):
                home = self._home_follower_target[side]
                if self._hold_target[side] is not None and home is not None:
                    self._hold_target[side] = self._with_gripper(
                        self._hold_target[side], home[6]
                    )
        # HOME completion is a powered position hold, not IDLE. Piper has no
        # joint brakes, so disabling here would undo the safety guarantee.
        if not self._emergency_stop_active:
            self._mode = "HOLD"
            self._hold_active = True
            self._mode_pub.publish(String(data=self._mode))
            self._hold_active_pub.publish(Bool(data=True))
        self._home_active_pub.publish(Bool(data=False))
        if success:
            rospy.logwarn("HOME complete; remaining enabled in HOLD")
        else:
            rospy.logerr("%s", message)

    def _process_home(self, now):
        if not self._home_active:
            return
        releasing = self.home_release_gripper > 0.0
        timeout = self.home_timeout + (self.home_release_duration if releasing else 0.0)
        if now - self._home_started_at > timeout:
            self._finish_home(False, "HOME timed out; holding the current pose")
            return
        if self._home_phase == "SWITCHING":
            if all(state == "OUTPUT" for state in self._leader_state.values()):
                if releasing:
                    self._home_phase = "RELEASING"
                    self._home_release_until = now + self.home_release_duration
                    self._home_gripper_open = True
                    rospy.logwarn(
                        "All leader arms are OUTPUT; opening follower grippers to "
                        "%.3f m for %.1fs before moving HOME",
                        self.home_release_gripper,
                        self.home_release_duration,
                    )
                else:
                    self._home_phase = "MOVING"
                    rospy.logwarn("All leader arms are OUTPUT; moving four arms HOME")
            return
        if self._home_phase == "RELEASING":
            if now >= self._home_release_until:
                self._home_phase = "MOVING"
                rospy.logwarn("Grippers released; moving four arms HOME (grippers open)")
            return
        aligned = True
        followers_home = True
        for side, storage in (("left", self._left), ("right", self._right)):
            follower, follower_fresh = storage["follower"].sample(
                now, self.input_timeout
            )
            leader, leader_fresh = storage["leader"].sample(now, self.input_timeout)
            if not follower_fresh or not leader_fresh:
                aligned = False
                followers_home = False
                continue
            follower_target = self._home_follower_target[side]
            if (
                max(abs(a - b) for a, b in zip(follower[:6], follower_target[:6]))
                > self.takeover_joint_tolerance
            ):
                followers_home = False
            if self._home_gripper_open:
                follower_target = self._with_gripper(
                    follower_target, self.home_release_gripper
                )
            for actual, target in (
                (follower, follower_target),
                (leader, self._home_leader_target[side]),
            ):
                if max(abs(a - b) for a, b in zip(actual[:6], target[:6])) > self.takeover_joint_tolerance:
                    aligned = False
                if abs(actual[6] - target[6]) > self.takeover_gripper_tolerance:
                    aligned = False
        if self._home_gripper_open and followers_home:
            # Follower joints are HOME with open fingers; close to the HOME
            # value. Leader alignment is not required for this step.
            self._home_gripper_open = False
            self._home_stable_count = 0
            rospy.logwarn("Follower arms at HOME; closing follower grippers")
            return
        self._home_stable_count = self._home_stable_count + 1 if aligned else 0
        if self._home_stable_count >= self.home_stable_samples:
            self._finish_home(True, "")

    def _process_pending_takeover(self, now):
        if not self._takeover_pending:
            return
        elapsed = now - self._takeover_pending_since
        policy_acknowledged = (
            self._policy_hold_ack
            and self._policy_hold_ack_received_at >= self._takeover_pending_since
        )
        if not policy_acknowledged:
            if elapsed > self.takeover_policy_ack_timeout:
                self._abort_pending_takeover(
                    "OpenPI did not acknowledge that policy command publication stopped"
                )
            else:
                rospy.logwarn_throttle(
                    1.0,
                    "TAKEOVER remains in powered HOLD: waiting for "
                    "/teleop/policy_hold_ack=true from OpenPI",
                )
            return

        policy_quiet = all(
            now - storage["policy"].received_at >= self.takeover_policy_quiet
            for storage in (self._left, self._right)
        )

        # Keep OUTPUT position commands flowing throughout the operator's
        # one-second hand-ready period. Closing this gate immediately after the
        # policy ACK caused the physical leaders to drift past the 0.04-rad
        # alignment gate before the delay elapsed, so 0xFA was never sent.
        if self._leader_tx_stop_requested_at <= 0.0:
            pre_drain_deadline = (
                max(self.takeover_release_delay, self.takeover_policy_ack_timeout)
                + self.takeover_alignment_timeout
            )
            ready_to_drain = (
                elapsed >= self.takeover_release_delay
                and policy_quiet
                and self._takeover_ready
            )
            if ready_to_drain:
                # Alignment is verified and latched at the exact ownership
                # boundary. During the following short TX drain the leader is
                # intentionally no longer driven as OUTPUT, so requiring live
                # OUTPUT alignment again would recreate the deadlock.
                self._takeover_alignment_latched = True
                self._set_takeover_phase("DRAINING_OUTPUT")
                self._set_leader_tx_enabled(False)
                rospy.logwarn(
                    "OpenPI stopped and one-second powered HOLD completed with "
                    "leader/follower alignment verified; closing both leader "
                    "OUTPUT TX gates"
                )
                return
            if elapsed > pre_drain_deadline:
                details = []
                if not policy_quiet:
                    details.append("policy topic still active")
                if not self._takeover_ready:
                    details.append("leader/follower alignment lost")
                self._abort_pending_takeover(
                    "TAKEOVER pre-drain gate timed out: {}".format(
                        ", ".join(details) if details else "unknown gate"
                    )
                )
            elif elapsed >= self.takeover_release_delay:
                rospy.logwarn_throttle(
                    1.0,
                    "TAKEOVER remains in powered OUTPUT HOLD: waiting for "
                    "policy quiet and aligned leader/follower feedback before "
                    "the short TX drain",
                )
            return

        # Stop the drivers from replaying their cached final OUTPUT target.
        # The driver ACK is protected by its send mutex, so ACK=true proves
        # that any JointCtrl/GripperCtrl call already in progress has returned.
        leader_tx_acknowledged = all(
            self._leader_tx_quiet_ack[side]
            and self._leader_tx_quiet_ack_at[side]
            >= self._leader_tx_stop_requested_at
            for side in ("left", "right")
        )
        if leader_tx_acknowledged and self._leader_tx_quiet_since <= 0.0:
            self._leader_tx_quiet_since = max(
                self._leader_tx_quiet_ack_at["left"],
                self._leader_tx_quiet_ack_at["right"],
            )
        leader_tx_quiet = (
            leader_tx_acknowledged
            and now - self._leader_tx_quiet_since
            >= self.takeover_leader_tx_quiet
        )
        if leader_tx_quiet:
            self._set_takeover_phase("LEADER_TX_QUIET")
        if (
            self._takeover_alignment_latched
            and leader_tx_quiet
        ):
            self._takeover_pending = False
            self._begin_takeover_role_switch(now)
            return
        if (
            now - self._leader_tx_stop_requested_at
            > self.takeover_leader_tx_quiet + self.takeover_alignment_timeout
        ):
            details = []
            if not leader_tx_acknowledged:
                details.append(
                    "leader TX ACK left={} right={}".format(
                        self._leader_tx_quiet_ack["left"],
                        self._leader_tx_quiet_ack["right"],
                    )
                )
            if not leader_tx_quiet:
                details.append("leader SocketCAN quiet interval incomplete")
            if not self._takeover_alignment_latched:
                details.append("alignment was not latched before TX drain")
            self._abort_pending_takeover(
                "TAKEOVER TX-drain gate timed out: {}".format(
                    ", ".join(details) if details else "unknown gate"
                )
            )

    def _abort_pending_takeover(self, reason):
        """Fail closed in powered HOLD and restore OUTPUT hold transmission."""
        self._takeover_pending = False
        self._takeover_alignment_latched = False
        self._set_takeover_phase("HOLD")
        self._set_takeover_ready(False)
        self._set_leader_tx_enabled(True)
        rospy.logerr(
            "%s; remaining in powered HOLD (policy remains stopped; press I "
            "explicitly to resume)",
            reason,
        )

    def _begin_takeover_role_switch(self, now):
        """Freeze followers, then request runtime 0xFA on both leaders."""
        self._hold_target["left"] = self._current_anchor("left", now)
        self._hold_target["right"] = self._current_anchor("right", now)
        self._mode = "TAKEOVER"
        self._takeover_pending = False
        self._set_takeover_phase("SWITCHING_MASTER")
        self._inference_phase = "IDLE"
        self._set_inference_active(False)
        # Keep this gate closed throughout MASTER/TAKEOVER.  It is reopened
        # only by HOLD recovery, HOME, or a fresh INFERENCE request.
        self._set_leader_tx_enabled(False)
        self._takeover_role_started_at = now
        self._takeover_alignment_count = 0
        self._mode_pub.publish(String(data=self._mode))
        rospy.logwarn(
            "Teleop ownership switching: INFERENCE -> TAKEOVER; "
            "followers held until both leaders publish fresh 0x155-0x159 MASTER input"
        )

    def _process_takeover_role_switch(self, now):
        if self._mode != "TAKEOVER":
            return
        if self._takeover_phase != "SWITCHING_MASTER":
            return
        _, _, leaders_fresh = self._fresh_pair("leader", now)
        if leaders_fresh and all(
            state in ("MASTER", "MASTER_ARMED")
            for state in self._leader_state.values()
        ):
            self._set_takeover_phase("ACTIVE")
            rospy.logwarn(
                "TAKEOVER active: policy remains stopped; follower command source "
                "changed to gravity-compensated leader 0x155-0x159 input"
            )
            return
        if now - self._takeover_role_started_at > self.takeover_role_timeout:
            self._fail_closed(
                "TAKEOVER 0xFA MASTER transition timed out (left={} right={})".format(
                    self._leader_state["left"], self._leader_state["right"]
                ),
                now,
            )

    def _process_collect_start(self, now):
        """Keep followers fixed until both leaders are usable 0xFA sources."""
        if self._mode != "COLLECT" or self._collect_phase != "SWITCHING_MASTER":
            return
        _, _, leaders_fresh = self._fresh_pair("leader", now)
        if leaders_fresh and all(
            state == "MASTER" for state in self._leader_state.values()
        ):
            self._collect_phase = "ACTIVE"
            rospy.logwarn(
                "COLLECT active: both leaders are MASTER; leader commands released"
            )
            return
        if now - self._collect_role_started_at > self.collect_role_timeout:
            self._fail_closed(
                "COLLECT master-role transition timed out (left={} right={})".format(
                    self._leader_state["left"], self._leader_state["right"]
                ),
                now,
            )

    def _process_inference_start(self, now):
        if self._mode != "INFERENCE" or self._inference_phase != "SWITCHING_OUTPUT":
            return
        _, _, policy_fresh = self._fresh_pair("policy", now)
        policy_resynced = policy_fresh and all(
            storage["policy"].received_at > self._policy_resume_after
            for storage in (self._left, self._right)
        )
        if self._takeover_ready and policy_resynced and all(
            state == "OUTPUT" for state in self._leader_state.values()
        ):
            self._inference_phase = "ACTIVE"
            self._set_inference_active(True)
            rospy.logwarn(
                "INFERENCE active: all four arms aligned; policy commands released"
            )
            return
        if now - self._inference_started_at > self.inference_start_timeout:
            self._fail_closed(
                "INFERENCE startup timed out before both leaders reached OUTPUT alignment",
                now,
            )

    def _clamp(self, position):
        # Position limits are optional because calibrated Piper installations
        # can report small offsets outside the nominal SDK limits.  Velocity
        # limiting remains active regardless of this setting.
        if not self.enforce_position_limits:
            return list(position)
        result = []
        for index in range(6):
            result.append(
                max(self.joint_min[index], min(self.joint_max[index], position[index]))
            )
        result.append(max(self.gripper_min, min(self.gripper_max, position[6])))
        return result

    def _rate_limit(
        self, side, target, dt, joint_velocity=None, gripper_velocity=None
    ):
        target = self._clamp(target)
        previous = self._last_output[side]
        if previous is None:
            return target
        joint_step = (
            self.max_joint_velocity if joint_velocity is None else joint_velocity
        ) * dt
        gripper_step = (
            self.max_gripper_velocity
            if gripper_velocity is None
            else gripper_velocity
        ) * dt
        result = []
        for index, (old, new) in enumerate(zip(previous, target)):
            maximum_step = gripper_step if index == 6 else joint_step
            delta = max(-maximum_step, min(maximum_step, new - old))
            result.append(old + delta)
        return result

    @staticmethod
    def _message(position):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = position
        return message

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            now = time.monotonic()
            with self._lock:
                self._capture_home(now)
                self._process_auto_start(now)
                self._update_takeover_alignment(now)
                self._process_inference_start(now)
                self._process_pending_takeover(now)
                self._process_takeover_role_switch(now)
                self._process_collect_start(now)
                self._process_home(now)
                if any(
                    state in ("ERROR", "TEACH_FAILED")
                    for state in self._leader_state.values()
                ):
                    self._fail_closed("A leader role transition failed", now)
                left_target = self._target_for_side("left", now)
                right_target = self._target_for_side("right", now)
                if (left_target is None or right_target is None) and self._mode != "IDLE":
                    source = "policy" if self._mode == "INFERENCE" else "leader"
                    self._fail_closed(
                        "Active {} source became stale ({})".format(
                            source, self._source_age_text(source, now)
                        ),
                        now,
                    )
                    left_target = self._target_for_side("left", now)
                    right_target = self._target_for_side("right", now)

                dt = max(1.0 / self.publish_hz, now - self._last_publish_time)
                self._last_publish_time = now
                if left_target is not None and right_target is not None:
                    joint_velocity = (
                        self.home_max_joint_velocity
                        if self._home_active and self._home_phase == "MOVING"
                        else None
                    )
                    gripper_velocity = (
                        self.home_max_gripper_velocity
                        if self._home_active and self._home_phase == "MOVING"
                        else None
                    )
                    left_output = self._rate_limit(
                        "left", left_target, dt, joint_velocity, gripper_velocity
                    )
                    right_output = self._rate_limit(
                        "right", right_target, dt, joint_velocity, gripper_velocity
                    )
                    self._last_output["left"] = left_output
                    self._last_output["right"] = right_output
                else:
                    left_output = None
                    right_output = None
                output_mode = self._mode
                leader_tx_enabled = self._leader_tx_enabled
                home_active = self._home_active
                home_phase = self._home_phase
                home_leader_left = self._home_leader_target["left"]
                home_leader_right = self._home_leader_target["right"]
                emergency_stop_active = self._emergency_stop_active

            if (
                not emergency_stop_active
                and left_output is not None
                and right_output is not None
            ):
                left_message = self._message(left_output)
                right_message = self._message(right_output)
                self._left_command_pub.publish(left_message)
                self._right_command_pub.publish(right_message)
                self._left_action_pub.publish(left_message)
                self._right_action_pub.publish(right_message)
                if leader_tx_enabled and (
                    output_mode in ("INFERENCE", "HOLD") or home_active
                ):
                    if home_active and home_phase == "MOVING":
                        left_message = self._message(home_leader_left)
                        right_message = self._message(home_leader_right)
                    self._left_leader_command_pub.publish(left_message)
                    self._right_leader_command_pub.publish(right_message)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("piper_four_state_mux")
    FourStateMux().run()
