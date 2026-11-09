#!/usr/bin/env python3
"""Fail-closed command mux for a fixed-MASTER four-CAN collection session."""

import math
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String


class TimedJointState:
    def __init__(self):
        self.position = None
        self.received_at = 0.0

    def update(self, message):
        if len(message.position) < 7:
            rospy.logwarn_throttle(2.0, "Ignoring JointState with fewer than 7 positions")
            return
        values = [float(value) for value in message.position[:7]]
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(2.0, "Ignoring non-finite JointState")
            return
        self.position = values
        self.received_at = time.monotonic()

    def sample(self, now, timeout):
        if self.position is None:
            return None, False
        return list(self.position), 0.0 <= now - self.received_at <= timeout


class FixedCollectMux:
    def __init__(self):
        self.publish_hz = float(rospy.get_param("~publish_hz", 200.0))
        self.input_timeout = float(rospy.get_param("~input_timeout_s", 0.15))
        self.startup_timeout = float(rospy.get_param("~startup_timeout_s", 8.0))
        self.recovery_stable_time = float(
            rospy.get_param("~recovery_stable_time_s", 0.10)
        )
        # Measured follower ceiling on this cell is ~3.0 rad/s per joint and
        # ~0.22 m/s at the gripper. A limit below that throttles the arm in
        # software before it ever reaches its hardware limit.
        self.max_joint_velocity = float(
            rospy.get_param("~max_joint_velocity_rad_s", 3.0)
        )
        self.max_gripper_velocity = float(
            rospy.get_param("~max_gripper_velocity_m_s", 0.8)
        )
        # Anti-windup bound, see _bounded().
        self.max_joint_lag = float(
            rospy.get_param("~max_joint_tracking_lag_rad", 0.10)
        )
        self.max_gripper_lag = float(
            rospy.get_param("~max_gripper_tracking_lag_m", 0.01)
        )
        if min(
            self.publish_hz,
            self.input_timeout,
            self.startup_timeout,
            self.recovery_stable_time,
            self.max_joint_velocity,
            self.max_gripper_velocity,
            self.max_joint_lag,
            self.max_gripper_lag,
        ) <= 0.0:
            raise ValueError("fixed collection rates, timeouts and limits must be positive")

        self._lock = threading.RLock()
        self._leader = {"left": TimedJointState(), "right": TimedJointState()}
        self._follower = {"left": TimedJointState(), "right": TimedJointState()}
        self._follower_enabled = {"left": False, "right": False}
        self._last_output = {"left": None, "right": None}
        self._last_publish_time = time.monotonic()
        self._operator_active = True
        self._enable_requested = False
        self._enable_requested_at = 0.0
        self._collect_active = False
        self._recovery_started_at = 0.0
        self._hold_reason = None
        self._emergency_stop = False
        self._published_mode = None

        self._command_pub = {
            side: rospy.Publisher(
                "/teleop/command/joint_{}".format(side),
                JointState,
                queue_size=1,
                tcp_nodelay=True,
            )
            for side in ("left", "right")
        }
        self._action_pub = {
            side: rospy.Publisher(
                "/master/joint_{}".format(side),
                JointState,
                queue_size=1,
                tcp_nodelay=True,
            )
            for side in ("left", "right")
        }
        self._mode_pub = rospy.Publisher(
            "/teleop/mode", String, queue_size=1, latch=True
        )
        self._emergency_pub = rospy.Publisher(
            "/teleop/emergency_stop", Bool, queue_size=1, latch=True
        )
        self._inference_pub = rospy.Publisher(
            "/teleop/inference_active", Bool, queue_size=1, latch=True
        )
        self._takeover_pub = rospy.Publisher(
            "/teleop/takeover_ready", Bool, queue_size=1, latch=True
        )
        self._follower_enable_pub = rospy.Publisher(
            "/teleop/follower_enable_request", Bool, queue_size=1, latch=True
        )
        for side in ("left", "right"):
            rospy.Subscriber(
                "/teleop/leader/joint_{}".format(side),
                JointState,
                self._joint_callback,
                callback_args=self._leader[side],
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                "/puppet/joint_{}".format(side),
                JointState,
                self._joint_callback,
                callback_args=self._follower[side],
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                "/teleop/follower_enabled_{}".format(side),
                Bool,
                self._follower_enabled_callback,
                callback_args=side,
                queue_size=1,
                tcp_nodelay=True,
            )
        rospy.Subscriber(
            "/teleop/emergency_stop_request",
            Empty,
            self._emergency_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/teleop/mode_request", String, self._mode_callback, queue_size=10
        )
        rospy.Subscriber(
            "/teleop/home_request", Empty, self._home_callback, queue_size=1
        )
        rospy.Subscriber(
            "/enable_flag", Bool, self._enable_callback, queue_size=1
        )

        self._set_mode("IDLE")
        self._emergency_pub.publish(Bool(data=False))
        self._inference_pub.publish(Bool(data=False))
        self._takeover_pub.publish(Bool(data=False))
        self._follower_enable_pub.publish(Bool(data=False))
        rospy.loginfo("Fixed-MASTER collection mux waiting for four fresh arm streams")

    def _joint_callback(self, message, storage):
        with self._lock:
            storage.update(message)

    def _follower_enabled_callback(self, message, side):
        with self._lock:
            enabled = bool(message.data)
            if enabled != self._follower_enabled[side]:
                self._follower_enabled[side] = enabled
                rospy.logwarn(
                    "Fixed collection %s follower enable confirmed=%s", side, enabled
                )

    def _set_mode(self, mode):
        if mode != self._published_mode:
            self._published_mode = mode
            self._mode_pub.publish(String(data=mode))

    def _emergency_callback(self, _message):
        with self._lock:
            self._latch_emergency("operator QUICK EMERGENCY STOP")

    def _enable_callback(self, message):
        with self._lock:
            requested = bool(message.data)
            if requested and self._emergency_stop:
                rospy.logerr("Fixed collection enable rejected: E-STOP is latched")
                return
            if requested == self._enable_requested:
                return
            self._enable_requested = requested
            if not requested:
                self._collect_active = False
                self._hold_reason = None
                self._recovery_started_at = 0.0
                self._last_output = {"left": None, "right": None}
                self._follower_enable_pub.publish(Bool(data=False))
                if self._operator_active:
                    self._set_mode("IDLE")
                rospy.logwarn("Fixed collection command output disabled")
            else:
                self._enable_requested_at = time.monotonic()
                self._recovery_started_at = 0.0
                self._follower_enable_pub.publish(Bool(data=True))
                rospy.logwarn(
                    "Fixed collection enable received; waiting for both followers to "
                    "confirm all six motor drivers"
                )

    def _mode_callback(self, message):
        requested = message.data.strip().upper()
        with self._lock:
            if requested == "IDLE":
                self._operator_active = False
                self._collect_active = False
                self._set_mode("IDLE")
                rospy.logwarn("Fixed collection stopped by IDLE request")
            else:
                rospy.logwarn(
                    "Fixed collection ignores %s; stop this launch before inference/takeover",
                    requested,
                )

    @staticmethod
    def _home_callback(_message):
        rospy.logwarn("HOME is unavailable in fixed collection; stop and restart safely")

    def _latch_emergency(self, reason):
        if self._emergency_stop:
            return
        self._emergency_stop = True
        self._enable_requested = False
        self._collect_active = False
        self._operator_active = False
        self._follower_enable_pub.publish(Bool(data=False))
        self._emergency_pub.publish(Bool(data=True))
        self._set_mode("ESTOP")
        rospy.logerr("Fixed collection stopped: %s", reason)

    def _fresh_inputs(self, now):
        values = {}
        stale = []
        for group_name, group in (("leader", self._leader), ("follower", self._follower)):
            values[group_name] = {}
            for side in ("left", "right"):
                value, fresh = group[side].sample(now, self.input_timeout)
                values[group_name][side] = value
                if not fresh:
                    stale.append("{}_{}".format(group_name, side))
        return values, stale

    @staticmethod
    def _message(position):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = list(position)
        return message

    def _bounded(self, side, target, dt):
        old = self._last_output[side]
        if old is None:
            return list(target)
        output = []
        for index, (previous, requested) in enumerate(zip(old, target)):
            if index == 6:
                velocity, max_lag = self.max_gripper_velocity, self.max_gripper_lag
            else:
                velocity, max_lag = self.max_joint_velocity, self.max_joint_lag
            step = velocity * dt
            delta = requested - previous
            value = previous + max(-step, min(step, delta))
            # Without this clamp the slew limit is an unbounded integrator:
            # a leader segment faster than `velocity` accumulates error for
            # its whole duration, so the follower is still travelling after
            # the operator has already stopped.
            output.append(max(requested - max_lag, min(requested + max_lag, value)))
        return output

    def _publish_last_output(self):
        for side in ("left", "right"):
            if self._last_output[side] is not None:
                self._command_pub[side].publish(
                    self._message(self._last_output[side])
                )

    def _enter_powered_hold(self, reason):
        self._recovery_started_at = 0.0
        self._publish_last_output()
        self._set_mode("HOLD")
        if reason != self._hold_reason:
            self._hold_reason = reason
            rospy.logerr(
                "Fixed collection powered HOLD: %s; no E-STOP was sent and "
                "followers retain their last position targets",
                reason,
            )

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            now = time.monotonic()
            with self._lock:
                values, stale = self._fresh_inputs(now)
                followers_enabled = all(self._follower_enabled.values())
                if self._operator_active and not self._emergency_stop:
                    if stale:
                        if self._collect_active:
                            self._enter_powered_hold(
                                "stale streams={}".format(",".join(stale))
                            )
                        elif (
                            self._enable_requested
                            and now - self._enable_requested_at > self.startup_timeout
                        ):
                            # No command source has been released yet, so a
                            # missing startup stream is not an in-motion
                            # fault.  Latching Piper 0x150 here made a brief
                            # follower boot/feedback delay turn into a hard
                            # recovery cycle: the operator could no longer
                            # enable the arm once feedback arrived. Remain
                            # command-inhibited and report exactly what is
                            # still missing; an active collection still
                            # fail-closes above if feedback disappears.
                            rospy.logwarn_throttle(
                                5.0,
                                "Fixed collection waiting; followers remain command-inhibited "
                                "until complete MASTER/follower streams arrive",
                            )
                    elif not followers_enabled:
                        missing = [
                            side
                            for side in ("left", "right")
                            if not self._follower_enabled[side]
                        ]
                        if self._collect_active:
                            self._enter_powered_hold(
                                "unconfirmed follower enable={}".format(
                                    ",".join(missing)
                                )
                            )
                        elif self._enable_requested:
                            rospy.logwarn_throttle(
                                2.0,
                                "Fixed collection command-inhibited; waiting for "
                                "follower enable confirmation: %s",
                                ",".join(missing),
                            )
                    else:
                        for side in ("left", "right"):
                            self._action_pub[side].publish(
                                self._message(values["leader"][side])
                            )
                        if not self._enable_requested:
                            if self._collect_active:
                                self._collect_active = False
                            self._set_mode("IDLE")
                        elif not self._collect_active:
                            if self._recovery_started_at <= 0.0:
                                self._recovery_started_at = now
                            if now - self._recovery_started_at < self.recovery_stable_time:
                                rate.sleep()
                                continue
                            for side in ("left", "right"):
                                self._last_output[side] = list(values["follower"][side])
                            self._last_publish_time = now
                            self._collect_active = True
                            self._set_mode("COLLECT")
                            rospy.logwarn(
                                "Fixed collection active: fresh MASTER/follower inputs and "
                                "both follower enables are confirmed"
                            )
                        if self._collect_active:
                            if self._hold_reason is not None:
                                if self._recovery_started_at <= 0.0:
                                    self._recovery_started_at = now
                                if now - self._recovery_started_at < self.recovery_stable_time:
                                    self._publish_last_output()
                                    rate.sleep()
                                    continue
                                rospy.logwarn(
                                    "Fixed collection recovered from powered HOLD: %s",
                                    self._hold_reason,
                                )
                                self._hold_reason = None
                            self._set_mode("COLLECT")
                            dt = min(
                                0.05,
                                max(1.0 / self.publish_hz, now - self._last_publish_time),
                            )
                            self._last_publish_time = now
                            for side in ("left", "right"):
                                command = self._bounded(
                                    side, values["leader"][side], dt
                                )
                                self._last_output[side] = command
                                self._command_pub[side].publish(self._message(command))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("piper_fixed_collect_mux")
    FixedCollectMux().run()
