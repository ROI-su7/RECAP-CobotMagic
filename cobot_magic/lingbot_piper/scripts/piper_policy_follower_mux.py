#!/usr/bin/env python3
"""Fail-closed policy mux for the common two-follower inference session.

This node never opens or commands either leader CAN. /enable_flag only arms
the session; a private /teleop/follower_enable is sent to the two drivers only
after fresh left/right policy plus follower feedback are all present. The first
command is seeded from measured follower poses.

LingBot copy of cobot_magic's piper_policy_follower_mux.py. Differences:
- Freshness is sampled under the lock, and a feedback sample newer than the
  loop's clock (negative age) counts as fresh. The reference version could
  read -0.001 s as stale.
- Follower feedback gaps never disable the motors (Piper has no brakes; 0x150
  and disable both drop the arms). Every mode freezes the last validated
  targets with torque on, as the reference already did for rlt_stage2.
- HOME pauses at its last output during a feedback gap and resumes when
  feedback returns; a gap longer than home_pause_timeout_s converts to HOLD.
  With rlt_stage2 the gap converts to HOLD at once (no automatic motion).
- I during HOME is queued until HOME completes. The reference cancelled HOME
  and drove the arms back to the pre-HOME HOLD pose at 1.5 rad/s, so the
  policy restarted from an unseen mid-way pose (2026-10-21). Every resume now
  holds the current output while waiting for the fresh policy chunk.
- A HOLD pressed before the followers were enabled no longer deadlocks the
  start: arming clears it and I releases it. The reference latched
  hold_active=true, the HOLD-aware client never published, activation needs
  policy input, and I was rejected until then (2026-10-21 15:06-15:10, Pi05).
- Follower firmware faults (/puppet/arm_status_*) block enabling and stop
  inference in HOLD. A quick stop latched in the controller survives a ROS
  relaunch; the drivers then report "enabled" while every motion command is
  ignored and the command drifts away from the pose (2026-10-21 15:10-15:18).
- Optional grip squeeze (gripper_squeeze_margin_m, default off) for clients
  that send raw predicted openings, e.g. Pi05; see _squeeze.
Only the operator E-stop request latches E-STOP.
"""

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

    def age(self, now):
        return float("inf") if self.position is None else now - self.received_at

    def sample(self, now, timeout):
        if self.position is None:
            return None, False
        # A callback may stamp received_at after `now` was read; negative age
        # is the freshest possible sample, not a stale one.
        return list(self.position), now - self.received_at <= timeout


class PolicyFollowerMux:
    # Checked by deploy/piper_real/mux_guard.py; keep both strings equal.
    MUX_IMPL = "lingbot_piper/piper_policy_follower_mux powered-hold v1"
    # PiperStatusMsg.arm_status values after which motion commands are ignored.
    ARM_FAULTS = {
        1: "EMERGENCY STOP latched",
        5: "joint communication error",
        6: "joint brake not released",
        7: "collision protection",
        9: "joint status error",
    }
    STATUS_TIMEOUT = 1.0

    def __init__(self):
        self.rlt_stage2 = bool(rospy.get_param("~rlt_stage2", False))
        self.publish_hz = float(rospy.get_param("~publish_hz", 200.0))
        self.input_timeout = float(rospy.get_param("~input_timeout_s", 0.15))
        self.policy_timeout = float(
            rospy.get_param("~policy_input_timeout_s", 1.0)
        )
        self.startup_timeout = float(
            rospy.get_param("~startup_timeout_s", 10.0)
        )
        self.max_joint_velocity = float(
            rospy.get_param("~max_joint_velocity_rad_s", 1.5)
        )
        self.max_gripper_velocity = float(
            rospy.get_param("~max_gripper_velocity_m_s", 0.15)
        )
        self.home_max_joint_velocity = float(
            rospy.get_param("~home_max_joint_velocity_rad_s", 0.25)
        )
        self.home_max_gripper_velocity = float(
            rospy.get_param("~home_max_gripper_velocity_m_s", 0.03)
        )
        self.home_joint_tolerance = float(
            rospy.get_param("~home_joint_tolerance_rad", 0.04)
        )
        self.home_gripper_tolerance = float(
            rospy.get_param("~home_gripper_tolerance_m", 0.005)
        )
        self.home_stable_samples = int(
            rospy.get_param("~home_stable_samples", 20)
        )
        self.home_pause_timeout = float(
            rospy.get_param("~home_pause_timeout_s", 1.0)
        )
        self.squeeze_margin = float(rospy.get_param("~gripper_squeeze_margin_m", 0.0))
        self.squeeze_below = float(rospy.get_param("~gripper_hold_below_m", 0.03))
        self.squeeze_taper = float(rospy.get_param("~gripper_hold_taper_m", 0.04))
        if not (0.0 <= self.squeeze_margin < 0.02 and 0.0 < self.squeeze_below < self.squeeze_taper):
            raise ValueError("gripper squeeze needs 0 <= margin < 0.02 m and 0 < hold_below < hold_taper")
        if min(
            self.publish_hz,
            self.input_timeout,
            self.policy_timeout,
            self.startup_timeout,
            self.max_joint_velocity,
            self.max_gripper_velocity,
            self.home_max_joint_velocity,
            self.home_max_gripper_velocity,
            self.home_joint_tolerance,
            self.home_gripper_tolerance,
            self.home_stable_samples,
            self.home_pause_timeout,
        ) <= 0.0:
            raise ValueError("follower inference rates, timeouts and limits must be positive")

        self._lock = threading.RLock()
        self._policy = {"left": TimedJointState(), "right": TimedJointState()}
        self._follower = {"left": TimedJointState(), "right": TimedJointState()}
        self._last_output = {"left": None, "right": None}
        self._last_publish_time = time.monotonic()
        self._enable_requested = False
        self._enable_requested_at = 0.0
        self._active = False
        self._operator_active = True
        self._emergency_stop = False
        self._hold_active = self.rlt_stage2
        # HOLD -> I is a two-phase handshake.  The ROS controller keeps the
        # measured pose while the OpenPI client discards its old action chunk;
        # policy commands received before the I request are never replayed.
        self._resume_pending = False
        self._resume_requested_at = 0.0
        self._hold_target = {"left": None, "right": None}
        self._home_target = {"left": None, "right": None}
        self._home_active = False
        self._home_stable_count = 0
        self._home_paused_at = None
        self._resume_after_home = False
        self._arm_status = {"left": None, "right": None}
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
        # Preserve the existing collector's executed-action topics.  These are
        # the bounded commands actually selected for the followers, not raw
        # policy messages.
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
        self._inference_pub = rospy.Publisher(
            "/teleop/inference_active", Bool, queue_size=1, latch=True
        )
        self._emergency_pub = rospy.Publisher(
            "/teleop/emergency_stop", Bool, queue_size=1, latch=True
        )
        self._takeover_pub = rospy.Publisher(
            "/teleop/takeover_ready", Bool, queue_size=1, latch=True
        )
        self._hold_pub = rospy.Publisher(
            "/teleop/hold_active", Bool, queue_size=1, latch=True
        )
        self._home_pub = rospy.Publisher(
            "/teleop/home_active", Bool, queue_size=1, latch=True
        )
        # Follower drivers in start_inference_followers.launch subscribe to
        # this gated topic instead of the public /enable_flag. This prevents a
        # slow first model inference from enabling stationary arms and then
        # tripping the old 10-second startup E-STOP.
        self._follower_enable_pub = rospy.Publisher(
            "/teleop/follower_enable", Bool, queue_size=1, latch=True
        )
        # The LingBot client refuses --execute unless the node publishing
        # /teleop/follower_enable also publishes this marker: the reference
        # mux removes torque on HOME after a lock race.
        self._impl_pub = rospy.Publisher(
            "/teleop/follower_mux_impl", String, queue_size=1, latch=True
        )
        self._impl_pub.publish(String(data=self.MUX_IMPL))

        for side in ("left", "right"):
            rospy.Subscriber(
                "/teleop/policy/joint_{}".format(side),
                JointState,
                self._joint_callback,
                callback_args=self._policy[side],
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
        try:
            from piper_msgs.msg import PiperStatusMsg
        except ImportError:  # offline tests without the piper workspace
            PiperStatusMsg = None
        if PiperStatusMsg is not None:
            for side in ("left", "right"):
                rospy.Subscriber(
                    "/puppet/arm_status_{}".format(side),
                    PiperStatusMsg,
                    self._status_callback,
                    callback_args=side,
                    queue_size=1,
                )
        rospy.Subscriber("/enable_flag", Bool, self._enable_callback, queue_size=1)
        rospy.Subscriber(
            "/teleop/emergency_stop_request",
            Empty,
            self._emergency_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/teleop/hold_request", Empty, self._hold_callback, queue_size=1
        )
        rospy.Subscriber(
            "/teleop/mode_request", String, self._mode_callback, queue_size=1
        )
        rospy.Subscriber(
            "/teleop/home_request", Empty, self._home_callback, queue_size=1
        )

        self._set_mode("WAITING_ENABLE")
        self._inference_pub.publish(Bool(data=False))
        self._emergency_pub.publish(Bool(data=False))
        self._takeover_pub.publish(Bool(data=False))
        self._hold_pub.publish(Bool(data=self._hold_active))
        self._home_pub.publish(Bool(data=False))
        self._follower_enable_pub.publish(Bool(data=False))
        rospy.loginfo(
            "Follower-only inference waiting for explicit enable, policy and feedback"
        )

    def _set_mode(self, mode):
        if self.rlt_stage2 and mode in ("INFERENCE_FOLLOWERS", "WAITING_POLICY_RESYNC"):
            mode = "INFERENCE"
        if mode != self._published_mode:
            self._published_mode = mode
            self._mode_pub.publish(String(data=mode))

    def _status_callback(self, message, side):
        with self._lock:
            self._arm_status[side] = (int(message.arm_status), time.monotonic())

    def _arm_faults(self, now):
        faults = []
        for side in ("left", "right"):
            status = self._arm_status[side]
            if status is not None and now - status[1] <= self.STATUS_TIMEOUT:
                if status[0] in self.ARM_FAULTS:
                    faults.append("{}: {}".format(side, self.ARM_FAULTS[status[0]]))
        return faults

    def _joint_callback(self, message, storage):
        with self._lock:
            storage.update(message)

    def _enable_callback(self, message):
        with self._lock:
            requested = bool(message.data)
            if requested and self._emergency_stop:
                rospy.logerr("Follower inference enable rejected: E-STOP is latched")
                return
            if requested and not self._operator_active:
                rospy.logerr("Follower inference enable rejected: session is stopped")
                return
            if requested == self._enable_requested:
                return
            self._enable_requested = requested
            self._active = False
            self._resume_pending = False
            self._resume_requested_at = 0.0
            self._inference_pub.publish(Bool(data=False))
            self._last_output = {"left": None, "right": None}
            if requested:
                self._enable_requested_at = time.monotonic()
                if self.rlt_stage2:
                    self._hold_active = True
                    self._hold_pub.publish(Bool(data=True))
                    self._set_mode("HOLD")
                else:
                    if self._hold_active:
                        self._clear_prestart_hold("arming")
                    self._set_mode("WAITING_POLICY")
                rospy.logwarn(
                    "Follower inference armed; waiting for two policies and two feedback streams"
                )
            else:
                self._hold_active = False
                self._home_active = False
                self._resume_after_home = False
                self._hold_pub.publish(Bool(data=False))
                self._home_pub.publish(Bool(data=False))
                self._follower_enable_pub.publish(Bool(data=False))
                self._set_mode("WAITING_ENABLE")
                rospy.logwarn("Follower inference command output disabled")

    def _mode_callback(self, message):
        requested = message.data.strip().upper()
        with self._lock:
            if requested == "IDLE":
                self._enter_hold("IDLE request converted to safe HOLD")
            elif requested == "INFERENCE":
                if self._emergency_stop:
                    rospy.logerr("Cannot resume INFERENCE: hardware E-STOP is latched")
                elif self._active and any(
                    not self._follower[side].sample(time.monotonic(), self.input_timeout)[1]
                    for side in ("left", "right")
                ):
                    self._feedback_timeout_hold("I rejected: follower feedback is stale")
                elif self._arm_faults(time.monotonic()):
                    rospy.logerr(
                        "I rejected: follower firmware reports %s; motion commands would be ignored",
                        ", ".join(self._arm_faults(time.monotonic())),
                    )
                elif (self._enable_requested and not self._active and not self.rlt_stage2
                      and self._hold_active):
                    self._clear_prestart_hold("I")
                elif not self._enable_requested or (not self._active and not self.rlt_stage2):
                    rospy.logerr("Cannot resume INFERENCE before followers are active")
                elif self._active and not self._hold_active and not self._resume_pending:
                    # The reference mux entered resync here and then tracked a
                    # stale (or missing, i.e. crash) HOLD target.
                    rospy.logwarn("I ignored: follower inference is already running")
                elif self._home_active:
                    self._resume_after_home = True
                    rospy.logwarn("I queued: inference resumes from the HOME pose once HOME completes")
                else:
                    self._resume_inference()
            else:
                rospy.logwarn(
                    "Follower-only inference ignores %s; TAKEOVER requires the supervised launch",
                    requested,
                )

    def _clear_prestart_hold(self, source):
        # Nothing is enabled yet, so there is no pose to hold; a latched
        # hold_active=true would only stop the client from ever publishing.
        self._hold_active = False
        self._home_active = False
        self._resume_after_home = False
        self._hold_pub.publish(Bool(data=False))
        self._home_pub.publish(Bool(data=False))
        rospy.logwarn(
            "%s cleared a HOLD requested before the followers were enabled; "
            "waiting for two policies and two feedback streams", source
        )

    def _resume_inference(self):
        # Release the OpenPI pause gate, but continue publishing the current
        # output until both policy topics contain a command produced after
        # this resume request. Never fall back to an older HOLD pose.
        for side in ("left", "right"):
            if self._last_output[side] is not None:
                self._hold_target[side] = list(self._last_output[side])
        self._hold_active = False
        self._resume_pending = True
        self._resume_requested_at = time.monotonic()
        self._home_active = False
        self._resume_after_home = False
        self._hold_pub.publish(Bool(data=False))
        self._home_pub.publish(Bool(data=False))
        self._set_mode("WAITING_POLICY_RESYNC")
        self._inference_pub.publish(Bool(data=False))
        rospy.logwarn(
            "Follower-only I received; holding until a new post-HOLD policy chunk arrives"
        )

    def _home_callback(self, _message):
        with self._lock:
            if not self._enable_requested or not self._active:
                rospy.logerr("Cannot start HOME before followers are active")
                return
            if any(self._home_target[side] is None for side in ("left", "right")):
                rospy.logerr("Cannot start HOME: startup follower poses were not captured")
                return
            now = time.monotonic()
            if any(
                not self._follower[side].sample(now, self.input_timeout)[1]
                for side in ("left", "right")
            ):
                self._feedback_timeout_hold("HOME rejected: follower feedback is stale")
                return
            self._enter_hold("HOME requested")
            self._home_active = True
            self._home_stable_count = 0
            self._home_paused_at = None
            self._home_pub.publish(Bool(data=True))
            self._set_mode("HOME")
            rospy.logwarn("Follower HOME active; moving to startup poses at limited speed")

    def _hold_callback(self, _message):
        with self._lock:
            self._enter_hold("operator software HOLD")

    def _enter_hold(self, reason):
        now = time.monotonic()
        for side in ("left", "right"):
            measured, fresh = self._follower[side].sample(now, self.input_timeout)
            if fresh:
                self._hold_target[side] = measured
                if self._last_output[side] is not None:
                    # A gripper closed on an object measures the object's
                    # width; holding that width drops the grip force. Keep
                    # the tighter command so HOLD does not release the wire.
                    measured[6] = min(measured[6], self._last_output[side][6])
            elif self._last_output[side] is not None:
                self._hold_target[side] = list(self._last_output[side])
        self._hold_active = True
        self._resume_pending = False
        self._resume_requested_at = 0.0
        self._home_active = False
        self._home_stable_count = 0
        self._home_paused_at = None
        self._resume_after_home = False
        self._inference_pub.publish(Bool(data=False))
        self._hold_pub.publish(Bool(data=True))
        self._home_pub.publish(Bool(data=False))
        self._set_mode("HOLD")
        if self._enable_requested and self._active:
            rospy.logwarn("%s; followers remain enabled and position-held", reason)
        else:
            rospy.logwarn("%s; no enabled follower session exists yet", reason)

    def _emergency_callback(self, _message):
        with self._lock:
            self._latch_emergency("operator QUICK EMERGENCY STOP")

    def _feedback_ages(self, now):
        return ", ".join(
            "{}={:.3f}s".format(side, self._follower[side].age(now))
            for side in ("left", "right")
        )

    def _feedback_timeout_hold(self, reason):
        """Freeze targets without removing support torque on RX gaps.

        Disabling or 0x150 quick stop lets an unbraked Piper fall, so a stale
        feedback stream never latches E-STOP in any mode.
        """
        now = time.monotonic()
        ages = self._feedback_ages(now)
        if not self._hold_active or self._resume_pending or self._home_active:
            self._enter_hold(reason)
        rospy.logerr_throttle(
            1.0, "%s; feedback age [%s], limit=%.3fs; powered HOLD retained. "
            "Restore feedback, then press I; no automatic resume.",
            reason, ages, self.input_timeout,
        )

    def _latch_emergency(self, reason):
        if self._emergency_stop:
            return
        self._emergency_stop = True
        self._enable_requested = False
        self._active = False
        self._resume_after_home = False
        self._resume_pending = False
        self._resume_requested_at = 0.0
        self._operator_active = False
        self._inference_pub.publish(Bool(data=False))
        self._follower_enable_pub.publish(Bool(data=False))
        self._emergency_pub.publish(Bool(data=True))
        self._hold_pub.publish(Bool(data=False))
        self._home_pub.publish(Bool(data=False))
        self._set_mode("ESTOP")
        rospy.logerr("Follower-only inference stopped: %s", reason)

    def _samples(self, now):
        values = {"policy": {}, "follower": {}}
        freshness = {"policy": {}, "follower": {}}
        for side in ("left", "right"):
            values["policy"][side], fresh = self._policy[side].sample(
                now, self.policy_timeout
            )
            freshness["policy"][side] = fresh
            values["follower"][side], fresh = self._follower[side].sample(
                now, self.input_timeout
            )
            freshness["follower"][side] = fresh
        return values, freshness

    @staticmethod
    def _missing_streams(freshness):
        return [
            "{}_{}".format(group, side)
            for group in ("policy", "follower")
            for side in ("left", "right")
            if not freshness[group][side]
        ]

    def _squeeze(self, target):
        """Command a held object tighter than the predicted opening.

        Both datasets record action[t] == state[t+1]: a grasp is the object's
        measured width, so commanding it back gives almost no grip force and
        objects slip (LingBot fourth wire, Pi05 on 2026-10-21). Below
        squeeze_below close by squeeze_margin (not below 0), tapering to no
        change at squeeze_taper. LingBot squeezes in its client instead.
        """
        if self.squeeze_margin <= 0.0:
            return target
        weight = min(1.0, max(0.0, (self.squeeze_taper - target[6])
                                   / (self.squeeze_taper - self.squeeze_below)))
        squeezed = list(target)
        squeezed[6] = max(0.0, target[6] - self.squeeze_margin * weight)
        return squeezed

    def _bounded(self, side, target, dt, joint_velocity=None, gripper_velocity=None):
        previous = self._last_output[side]
        if previous is None:
            return list(target)
        result = []
        for index, (old, new) in enumerate(zip(previous, target)):
            velocity = (
                self.max_gripper_velocity if gripper_velocity is None else gripper_velocity
            ) if index == 6 else (
                self.max_joint_velocity if joint_velocity is None else joint_velocity
            )
            step = velocity * dt
            result.append(old + max(-step, min(step, new - old)))
        return result

    @staticmethod
    def _message(position):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = list(position)
        return message

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            messages = None
            activate_followers = False
            with self._lock:
                # Read the clock only after any callback holding the lock has
                # finished; a pre-lock timestamp can predate fresh feedback.
                now = time.monotonic()
                # Capture the launch-time follower poses before any model
                # command is released. H always returns to these positions.
                for side in ("left", "right"):
                    if self._home_target[side] is None:
                        position, fresh = self._follower[side].sample(
                            now, self.input_timeout
                        )
                        if fresh:
                            self._home_target[side] = position
                if (
                    self._operator_active
                    and self._enable_requested
                    and not self._emergency_stop
                ):
                    values, freshness = self._samples(now)
                    missing = self._missing_streams(freshness)
                    faults = self._arm_faults(now)
                    if not self._active:
                        if self.rlt_stage2 and not self._resume_pending:
                            # No policy motion or motor enable before I.
                            pass
                        elif faults:
                            rospy.logerr_throttle(
                                5.0,
                                "Followers stay disabled: firmware reports %s; motion "
                                "commands would be ignored. Stop the client and launch, "
                                "power-cycle the follower arms, then relaunch",
                                ", ".join(faults),
                            )
                        elif not missing:
                            for side in ("left", "right"):
                                self._last_output[side] = list(
                                    values["follower"][side]
                                )
                            # Publish an exact measured-pose seed before the
                            # gated enable flag. Policy motion begins on the
                            # next loop iteration.
                            messages = {
                                side: self._message(values["follower"][side])
                                for side in ("left", "right")
                            }
                            self._last_publish_time = now
                            self._active = True
                            self._hold_active = False
                            self._hold_pub.publish(Bool(data=False))
                            self._resume_pending = False
                            activate_followers = True
                            self._set_mode("INFERENCE_FOLLOWERS")
                            self._inference_pub.publish(Bool(data=True))
                            rospy.logwarn(
                                "Follower-only inference active; leader CANs are not opened"
                            )
                        elif now - self._enable_requested_at > self.startup_timeout:
                            rospy.logwarn_throttle(
                                5.0,
                                "Follower inference still waiting; followers remain "
                                "disabled; missing streams: %s",
                                ", ".join(missing),
                            )
                    elif not self._hold_active and faults:
                        self._enter_hold("follower firmware reports " + ", ".join(faults))
                        rospy.logerr(
                            "Inference stopped in HOLD: %s. Commands are ignored until the "
                            "arm is reset; the HOLD pose prevents a jump afterwards",
                            ", ".join(faults),
                        )
                    elif not self._hold_active and not self._resume_pending and any(
                        not freshness["follower"][side] for side in ("left", "right")
                    ):
                        self._feedback_timeout_hold("follower feedback became stale")
                    elif not self._hold_active and not self._resume_pending and any(
                        not freshness["policy"][side] for side in ("left", "right")
                    ):
                        # A model/network pause does not require Piper 0x150.
                        # Freeze the last measured pose and keep torque enabled.
                        self._enter_hold("policy input became stale")

                    if self._active and self._resume_pending:
                        follower_missing = any(
                            not freshness["follower"][side] for side in ("left", "right")
                        )
                        if follower_missing:
                            self._feedback_timeout_hold(
                                "follower feedback became stale while waiting for policy resync"
                            )
                        else:
                            policy_resynced = all(
                                freshness["policy"][side]
                                and self._policy[side].received_at
                                > self._resume_requested_at
                                for side in ("left", "right")
                            )
                            if policy_resynced:
                                self._resume_pending = False
                                self._set_mode("INFERENCE_FOLLOWERS")
                                self._inference_pub.publish(Bool(data=True))
                                rospy.logwarn(
                                    "Follower-only inference resumed with a fresh post-HOLD policy chunk"
                                )

                    if self._active and not activate_followers and (
                        self._hold_active or self._resume_pending
                    ):
                        follower_missing = any(
                            not freshness["follower"][side] for side in ("left", "right")
                        )
                        if follower_missing and self._home_active and self.rlt_stage2:
                            # Stage 2 never resumes motion by itself after a
                            # feedback gap: convert HOME to a powered HOLD.
                            self._feedback_timeout_hold("follower feedback became stale during HOME")
                        if follower_missing and self._home_active:
                            # Pause HOME at its last verified output. The
                            # tolerance check below needs fresh feedback.
                            if self._home_paused_at is None:
                                self._home_paused_at = now
                                rospy.logerr(
                                    "Follower feedback stale during HOME [%s], limit=%.3fs; "
                                    "HOME paused, torque kept",
                                    self._feedback_ages(now), self.input_timeout,
                                )
                            elif now - self._home_paused_at > self.home_pause_timeout:
                                self._feedback_timeout_hold(
                                    "follower feedback stale for over {:.1f}s during HOME".format(
                                        self.home_pause_timeout
                                    )
                                )
                            self._home_stable_count = 0
                            self._last_publish_time = now
                            messages = {
                                side: self._message(self._last_output[side])
                                for side in ("left", "right")
                            }
                        elif follower_missing:
                            self._feedback_timeout_hold("follower feedback became stale during HOLD")
                        elif self._home_paused_at is not None:
                            rospy.logwarn(
                                "Follower feedback restored after %.3fs; HOME resumed",
                                now - self._home_paused_at,
                            )
                            self._home_paused_at = None
                        if messages is None and not self._emergency_stop:
                            # Keep sending the frozen last valid targets even
                            # during an RX gap. Never advance HOME or policy.
                            dt = min(
                                0.05,
                                max(1.0 / self.publish_hz, now - self._last_publish_time),
                            )
                            self._last_publish_time = now
                            messages = {}
                            aligned = True
                            for side in ("left", "right"):
                                if self._home_active:
                                    target = self._home_target[side]
                                elif self._hold_target[side] is not None:
                                    target = self._hold_target[side]
                                else:
                                    # Never crash the required node; hold in place.
                                    target = self._last_output[side]
                                output = self._bounded(
                                    side,
                                    target,
                                    dt,
                                    self.home_max_joint_velocity if self._home_active else None,
                                    self.home_max_gripper_velocity if self._home_active else None,
                                )
                                self._last_output[side] = output
                                messages[side] = self._message(output)
                                if self._home_active:
                                    actual = values["follower"][side]
                                    if max(
                                        abs(a - b) for a, b in zip(actual[:6], target[:6])
                                    ) > self.home_joint_tolerance:
                                        aligned = False
                                    if abs(actual[6] - target[6]) > self.home_gripper_tolerance:
                                        aligned = False
                            if self._home_active:
                                self._home_stable_count = (
                                    self._home_stable_count + 1 if aligned else 0
                                )
                                if self._home_stable_count >= self.home_stable_samples:
                                    self._home_active = False
                                    for side in ("left", "right"):
                                        self._hold_target[side] = list(
                                            self._home_target[side]
                                        )
                                    self._home_pub.publish(Bool(data=False))
                                    self._set_mode("HOLD")
                                    if self._resume_after_home:
                                        rospy.logwarn("Follower HOME complete; resuming queued I")
                                        self._resume_inference()
                                    else:
                                        rospy.logwarn("Follower HOME complete; remaining in HOLD")
                    elif self._active and not activate_followers:
                        dt = min(
                            0.05,
                            max(1.0 / self.publish_hz, now - self._last_publish_time),
                        )
                        self._last_publish_time = now
                        messages = {}
                        for side in ("left", "right"):
                            output = self._bounded(
                                side, self._squeeze(values["policy"][side]), dt
                            )
                            self._last_output[side] = output
                            messages[side] = self._message(output)

            if messages is not None:
                for side in ("left", "right"):
                    self._command_pub[side].publish(messages[side])
                    self._action_pub[side].publish(messages[side])
            if activate_followers:
                self._follower_enable_pub.publish(Bool(data=True))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("piper_policy_follower_mux")
    PolicyFollowerMux().run()
