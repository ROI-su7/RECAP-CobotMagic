#!/usr/bin/env python3
"""Collection-only follower driver with verified enable recovery.

The generic follower driver is shared by the legacy master/slave and inference
profiles.  Keep those paths unchanged and add collection-specific supervision
here instead.
"""

import threading
import time

import rospy
from std_msgs.msg import Bool

from piper_start_slave_node import C_PiperRosNode


class CollectPiperRosNode(C_PiperRosNode):
    """Report physical enable confirmation and retry while collection requests it."""

    def __init__(self):
        # C_PiperRosNode starts its subscriber threads during construction.  Hold
        # the enable subscriber until the collection-only status publisher and
        # retry state below are ready, including when its input topic is latched.
        self._collection_ready = threading.Event()
        self._follower_enabled_pub = None
        self._collection_enable_attempting = False
        self._next_enable_retry_at = float("inf")
        self._enable_loss_started_at = 0.0
        super().__init__()

        self.enable_retry_backoff = float(
            rospy.get_param("~collection_enable_retry_backoff_s", 1.0)
        )
        self.enable_status_poll_period = float(
            rospy.get_param("~collection_enable_status_poll_s", 0.10)
        )
        self.enable_loss_grace = float(
            rospy.get_param("~collection_enable_loss_grace_s", 0.50)
        )
        if min(
            self.enable_retry_backoff,
            self.enable_status_poll_period,
            self.enable_loss_grace,
        ) <= 0.0:
            raise ValueError("collection enable supervision intervals must be positive")

        self._follower_enabled_pub = rospy.Publisher(
            "/teleop/follower_enabled", Bool, queue_size=1, latch=True
        )
        self._publish_enabled(False)
        self._collection_ready.set()
        self._enable_watchdog = threading.Thread(
            target=self._enable_watchdog_loop,
            name="collection-enable-{}".format(self.can_port),
            daemon=True,
        )
        self._enable_watchdog.start()

    def SubEnableThread(self):
        self._collection_ready.wait()
        super().SubEnableThread()

    def _publish_enabled(self, enabled):
        if self._follower_enabled_pub is not None:
            self._follower_enabled_pub.publish(Bool(data=bool(enabled)))

    def _enable_and_wait(self):
        # Close the externally visible gate before seeding/re-enabling.  The mux
        # will hold both followers at their last commanded poses until this arm
        # has positively confirmed all six motor drivers again.
        if self._collection_enable_attempting:
            return False
        self._collection_enable_attempting = True
        try:
            self._publish_enabled(False)
            enabled = super()._enable_and_wait()
            self._publish_enabled(enabled)
            self._enable_loss_started_at = 0.0
            self._next_enable_retry_at = time.monotonic() + self.enable_retry_backoff
            return enabled
        finally:
            self._collection_enable_attempting = False

    def _disable_and_wait(self):
        self._publish_enabled(False)
        return super()._disable_and_wait()

    def enable_callback(self, enable_flag):
        if bool(enable_flag.data):
            self._next_enable_retry_at = time.monotonic() + self.enable_retry_backoff
        else:
            self._next_enable_retry_at = float("inf")
            self._enable_loss_started_at = 0.0
        super().enable_callback(enable_flag)
        self._publish_enabled(self.GetEnableFlag())

    def emergency_stop_callback(self, stop_flag):
        if bool(stop_flag.data):
            self._publish_enabled(False)
        super().emergency_stop_callback(stop_flag)
        self._publish_enabled(self.GetEnableFlag())

    def _enable_watchdog_loop(self):
        while not rospy.is_shutdown() and not self._shutting_down:
            now = time.monotonic()
            requested = self._enable_requested and not self._emergency_stop_active
            if not requested:
                self._enable_loss_started_at = 0.0
                time.sleep(self.enable_status_poll_period)
                continue

            try:
                states = self._supervised_enable_states()
            except Exception as exc:
                # Stale or unreadable feedback means the arm's real state is
                # unknown.  Report it loudly and drop the externally visible
                # confirmation so the mux stops issuing commands, rather than
                # letting a frozen cache pass for a healthy arm.
                rospy.logerr_throttle(
                    1.0,
                    "%s collection enable-status read failed: %s",
                    self.can_port,
                    exc,
                )
                self._publish_enabled(False)
                time.sleep(self.enable_status_poll_period)
                continue

            if self.GetEnableFlag():
                if all(states):
                    self._enable_loss_started_at = 0.0
                elif self._enable_loss_started_at <= 0.0:
                    self._enable_loss_started_at = now
                elif now - self._enable_loss_started_at >= self.enable_loss_grace:
                    rospy.logerr(
                        "%s lost physical enable confirmation; states=%s; "
                        "holding collection and re-enabling from measured pose",
                        self.can_port,
                        states,
                    )
                    self._enable_and_wait()
            elif (
                not self._collection_enable_attempting
                and now >= self._next_enable_retry_at
            ):
                rospy.logwarn(
                    "%s collection enable is still unconfirmed; retrying safely",
                    self.can_port,
                )
                self._enable_and_wait()

            time.sleep(self.enable_status_poll_period)


if __name__ == "__main__":
    try:
        piper = CollectPiperRosNode()
        piper.Pubilsh()
    except rospy.ROSInterruptException:
        pass
