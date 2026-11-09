#!/usr/bin/env python3
"""Offline regression tests for four-CAN leader role handshakes.

These tests never create a CAN socket.  They exercise stream validation and
SDK call sequences with a fake SDK object so hardware cannot be enabled.
"""

import importlib.util
import math
import pathlib
import struct
import sys
import threading
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "piper_leader_mode_driver.py"
# rosrun puts scripts/ on sys.path for the driver's sibling imports.
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("piper_leader_mode_driver", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Publisher:
    def publish(self, _message):
        pass


def _stream_driver(snapshot):
    driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
    driver.role_timeout = 0.08
    driver.role_stream_stable = 0.03
    driver.role_stream_min_hz = 10.0
    driver.role_stream_freshness = 0.01
    driver._shutting_down = False
    driver._last_feedback = None
    driver._feedback_pub = _Publisher()
    driver._role_stream_snapshot = snapshot
    return driver


class LeaderRoleStreamTest(unittest.TestCase):
    def test_one_changed_sample_is_not_a_role_proof(self):
        driver = _stream_driver(lambda _role: (2.0, 2.0, 200.0, 200.0))
        self.assertFalse(
            driver._wait_for_new_stream("MASTER", (1.0, 1.0, 200.0, 200.0))
        )

    def test_continuously_changing_complete_stream_is_accepted(self):
        counter = [1.0]

        def snapshot(_role):
            counter[0] += 1.0
            return (counter[0], counter[0], 200.0, 200.0)

        driver = _stream_driver(snapshot)
        self.assertTrue(
            driver._wait_for_new_stream("MASTER", (1.0, 1.0, 200.0, 200.0))
        )

    def test_master_stream_accepts_fresh_joints_with_stale_gripper(self):
        counter = [1.0]

        def snapshot(_role):
            counter[0] += 1.0
            return (counter[0], 1.0, 200.0, 0.0)

        driver = _stream_driver(snapshot)
        self.assertTrue(
            driver._wait_for_new_stream("MASTER", (1.0, 1.0, 200.0, 200.0))
        )

    def test_output_stream_still_requires_fresh_gripper(self):
        counter = [1.0]

        def snapshot(_role):
            counter[0] += 1.0
            return (counter[0], 1.0, 200.0, 0.0)

        driver = _stream_driver(snapshot)
        self.assertFalse(
            driver._wait_for_new_stream("OUTPUT", (1.0, 1.0, 200.0, 200.0))
        )


class RawMasterObserverTest(unittest.TestCase):
    def _observer(self):
        observer = MODULE.RawMasterObserver.__new__(MODULE.RawMasterObserver)
        observer.can_port = "fake_can"
        observer._lock = threading.Lock()
        observer._capture_enabled = False
        observer._positions = {
            offset: [0.0] * 7 for offset in MODULE.MASTER_CONTROL_OFFSETS
        }
        observer._counts = {
            offset: {} for offset in MODULE.MASTER_CONTROL_OFFSETS
        }
        observer._group_count = {
            offset: 0 for offset in MODULE.MASTER_CONTROL_OFFSETS
        }
        observer._group_times = {
            offset: MODULE.deque(maxlen=64)
            for offset in MODULE.MASTER_CONTROL_OFFSETS
        }
        observer._gripper_times = {
            offset: MODULE.deque(maxlen=64)
            for offset in MODULE.MASTER_CONTROL_OFFSETS
        }
        observer._active_offset = None
        observer._last_pose = None
        observer._last_group_at = 0.0
        observer._last_gripper_at = 0.0
        return observer

    def test_raw_capture_ignores_output_frames_until_takeover_starts(self):
        observer = self._observer()
        observer._process_frame(0x155, struct.pack(">ii", 1000, 2000), 1.0)
        self.assertEqual(observer.snapshot(), (0.0, 0.0, 0.0, 0.0))

    def test_raw_capture_decodes_complete_shifted_master_groups(self):
        observer = self._observer()
        observer.start_capture([0.0] * 6 + [0.03])
        ids = (0x165, 0x166, 0x167)
        values = ((1000, -2000), (3000, -4000), (5000, -6000))
        for cycle in range(2):
            timestamp = 1.0 + cycle * 0.05
            for can_id, pair in zip(ids, values):
                observer._process_frame(can_id, struct.pack(">ii", *pair), timestamp)
        observer._process_frame(
            0x169, struct.pack(">i", 40000) + b"\x00" * 4, 1.06
        )
        stamp, _gripper_stamp, joint_hz, _gripper_hz = observer.snapshot()
        self.assertGreater(stamp, 0.0)
        self.assertGreaterEqual(joint_hz, 10.0)
        self.assertEqual(observer.active_offset(), 0x10)
        pose = observer.pose()
        self.assertAlmostEqual(pose[0], math.pi / 180.0)
        self.assertAlmostEqual(pose[1], -2.0 * math.pi / 180.0)
        self.assertAlmostEqual(pose[6], 0.04)
        self.assertIn("0x169=", observer.rate_text())

    def test_raw_capture_reports_role_tx_status_timeline_and_can_errors(self):
        observer = self._observer()
        observer.start_capture([0.0] * 7)
        since = observer.transition_marker()
        observer._process_frame(
            MODULE.ROLE_CONFIG_CAN_ID,
            bytes([0xFA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            since + 0.01,
        )
        observer._process_frame(
            MODULE.ARM_STATUS_CAN_ID,
            bytes([0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00]),
            since + 0.02,
        )
        observer._process_frame(
            MODULE.ARM_STATUS_CAN_ID,
            bytes([0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            since + 0.03,
        )
        observer._process_frame(
            MODULE.MOTION_CTRL_2_CAN_ID,
            bytes([0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            since + 0.04,
        )
        observer._process_frame(
            MODULE.RESTORE_MASTER_SLAVE_CAN_ID,
            bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            since + 0.045,
        )
        observer._process_frame(
            MODULE.ARM_STATUS_CAN_ID,
            bytes([0x06, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            since + 0.05,
        )
        observer._process_frame(
            MODULE.CAN_ERR_FLAG | 0x20,
            b"\x00" * 8,
            since + 0.06,
        )

        self.assertTrue(observer.role_config_echo_seen(since))
        self.assertTrue(observer.standby_echo_seen(since))
        self.assertTrue(observer.restore_master_slave_echo_seen(since))
        self.assertEqual(observer.ctrl_modes_since(since), [0x01, 0x00, 0x06])
        self.assertTrue(observer.can_errors_seen(since))
        report = observer.transition_report(since)
        self.assertIn("0x470_TX_ECHO=[", report)
        self.assertIn("FA00000000000000", report)
        self.assertIn("ctrl=0x01", report)
        self.assertIn("ctrl=0x00", report)
        self.assertIn("ctrl=0x06", report)
        self.assertIn("0x191_RESTORE_TX_ECHO=[", report)
        self.assertIn("CAN_ERROR=[", report)


class _FakePiper:
    def __init__(self, ctrl_mode=0x01, restore_ctrl_mode=None):
        self.calls = []
        self.ctrl_mode = ctrl_mode
        self.restore_ctrl_mode = restore_ctrl_mode

    def MasterSlaveConfig(self, *args):
        self.calls.append(("MasterSlaveConfig", args))

    def MotionCtrl_2(self, *args):
        self.calls.append(("MotionCtrl_2", args))
        if args and args[0] == 0x00:
            self.ctrl_mode = 0x00
        elif args and args[0] == 0x01:
            self.ctrl_mode = 0x01

    def MotionCtrl_1(self, *args):
        self.calls.append(("MotionCtrl_1", args))
        if len(args) >= 3 and args[2] == 0x01:
            self.ctrl_mode = 0x02
        elif len(args) >= 3 and args[2] == 0x02:
            self.ctrl_mode = 0x01

    def ReqMasterArmMoveToHome(self, *args):
        self.calls.append(("ReqMasterArmMoveToHome", args))
        if self.restore_ctrl_mode is not None:
            self.ctrl_mode = self.restore_ctrl_mode

    def GripperCtrl(self, *args):
        self.calls.append(("GripperCtrl", args))

    def GetArmStatus(self):
        class _ArmStatus:
            pass

        class _Status:
            time_stamp = 1.0
            arm_status = _ArmStatus()

        result = _Status()
        result.arm_status.ctrl_mode = self.ctrl_mode
        return result


class LeaderMasterSequenceTest(unittest.TestCase):
    def _continuity_driver(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver.role_pose_tolerance = 0.08
        driver.role_gripper_tolerance = 0.015
        driver.role_pose_hard_tolerance = 0.20
        driver.role_gripper_hard_tolerance = 0.030
        return driver

    def test_mapping_soft_tolerance_warns_but_allows_master(self):
        driver = self._continuity_driver()
        reference = [0.0] * 7
        observed = [0.0, 0.041, -0.042, 0.0, -0.140, 0.0, 0.0]

        self.assertTrue(
            driver._validate_role_pose_continuity(reference, observed, "MASTER")
        )

    def test_mapping_hard_tolerance_still_rejects_master(self):
        driver = self._continuity_driver()
        reference = [0.0] * 7
        observed = [0.0, 0.0, 0.0, 0.0, -0.210, 0.0, 0.0]

        self.assertFalse(
            driver._validate_role_pose_continuity(reference, observed, "MASTER")
        )

    def test_verified_runtime_master_allows_operator_motion_during_hand_off(self):
        driver = self._continuity_driver()
        reference = [0.0] * 7
        observed = [0.0, -0.03, -0.228, -0.02, -0.01, -0.02, 0.0]

        self.assertTrue(
            driver._validate_role_pose_continuity(
                reference,
                observed,
                "MASTER",
                allow_operator_motion=True,
            )
        )

    def test_master_preparation_reads_pose_only_after_disable(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver.master_prepare_hold = 0.0
        events = []
        driver._safety_ok = lambda force=False: True

        def disable():
            events.append("disable")
            return True

        def read_pose(role):
            events.append("read_{}".format(role))
            return [0.0] * 7

        driver._disable_and_wait = disable
        driver._read_role_pose = read_pose
        driver._last_feedback = None

        self.assertEqual(driver._prepare_disabled_output_for_master(), [0.0] * 7)
        self.assertEqual(events, ["disable", "read_OUTPUT"])

    def test_master_transition_uses_measured_pose_hold_before_role_command(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._enable_requested = True
        driver._role = "OUTPUT"
        driver._last_feedback = [0.0] * 7
        driver._output_enable_verified = True
        driver._command_output = None
        driver._failed_role_request = None
        driver._lock = threading.RLock()
        driver.role_config_retries = 2
        driver.piper = _FakePiper()
        states = []
        driver._set_state = states.append
        driver._read_role_pose = lambda _role: [0.0] * 7
        driver.master_enable_fallback = True

        def prepare_enabled():
            driver.piper.calls.append(("prepare_enabled", ()))
            return [0.0] * 7

        driver._prepare_enabled_output_for_master = prepare_enabled
        driver._role_stream_snapshot = lambda _role: (1.0, 1.0, 200.0, 200.0)
        driver._wait_for_new_stream = lambda _role, _before, **_kwargs: True
        driver._validate_role_pose_continuity = lambda *_args, **_kwargs: True
        driver._settle_stream = lambda: None

        self.assertTrue(driver._enter_master_role())
        self.assertEqual(
            driver.piper.calls,
            [
                ("prepare_enabled", ()),
                ("MasterSlaveConfig", (0xFA, 0x00, 0x00, 0x00)),
            ],
        )
        self.assertEqual(driver._role, "MASTER")
        self.assertEqual(states[-1], "MASTER")

    def test_status_confirmed_stationary_master_is_armed_without_stream(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._last_feedback = [0.0] * 7
        driver._output_enable_verified = True
        driver._command_output = None
        driver._failed_role_request = None
        driver._lock = threading.RLock()
        driver.role_config_retries = 2
        driver.piper = _FakePiper(ctrl_mode=0x06)
        states = []
        driver._set_state = states.append
        driver._settle_stream = lambda: None
        driver._role_stream_snapshot = lambda _role: (0.0, 0.0, 0.0, 0.0)
        driver._wait_for_new_stream = lambda _role, _before, **_kwargs: False

        self.assertTrue(driver._activate_master_stream([0.0] * 7, "test"))
        self.assertEqual(driver._role, "MASTER")
        self.assertEqual(states[-1], "MASTER_ARMED")
        self.assertEqual(
            driver.piper.calls,
            [
                ("MasterSlaveConfig", (0xFA, 0x00, 0x00, 0x00)),
            ],
        )

    def test_standby_then_restore_linkage_can_activate_master(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._last_feedback = [0.0] * 7
        driver._output_enable_verified = True
        driver._command_output = None
        driver._failed_role_request = None
        driver._lock = threading.RLock()
        driver.role_config_retries = 2
        driver.master_standby_activation = True
        driver.master_restore_linkage_activation = True
        driver.piper = _FakePiper(restore_ctrl_mode=0x06)
        states = []
        driver._set_state = states.append
        driver._settle_stream = lambda: None
        driver._role_stream_snapshot = lambda _role: (0.0, 0.0, 0.0, 0.0)
        driver._wait_for_new_stream = lambda _role, _before, **_kwargs: False

        self.assertTrue(driver._activate_master_stream([0.0] * 7, "test"))
        self.assertIn(("MotionCtrl_2", (0x00, 0x01, 0)), driver.piper.calls)
        self.assertIn(("ReqMasterArmMoveToHome", (0,)), driver.piper.calls)
        self.assertEqual(driver._role, "MASTER")
        self.assertEqual(states[-1], "MASTER_ARMED")

    def test_master_transition_falls_back_to_disabled_output(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._enable_requested = True
        driver._role = "OUTPUT"
        driver._last_feedback = [0.0] * 7
        driver._output_enable_verified = True
        driver._command_output = None
        driver._failed_role_request = None
        driver._lock = threading.RLock()
        driver.role_config_retries = 2
        driver.master_enable_fallback = True
        driver.piper = _FakePiper()
        driver._set_state = lambda _state: None
        driver._read_role_pose = lambda _role: [0.0] * 7
        driver._role_stream_snapshot = lambda _role: (1.0, 1.0, 200.0, 200.0)
        driver._validate_role_pose_continuity = lambda *_args, **_kwargs: True
        driver._settle_stream = lambda: None
        driver._recover_output_feedback_role = (
            lambda: driver.piper.calls.append(("recover_output", ())) or True
        )
        driver._prepare_enabled_output_for_master = (
            lambda: driver.piper.calls.append(("prepare_enabled", ())) or [0.0] * 7
        )
        driver._prepare_disabled_output_for_master = (
            lambda: driver.piper.calls.append(("prepare_disabled", ())) or [0.0] * 7
        )
        outcomes = iter([False, False, True])
        driver._wait_for_new_stream = (
            lambda _role, _before, **_kwargs: next(outcomes)
        )

        self.assertTrue(driver._enter_master_role())
        self.assertEqual(
            driver.piper.calls,
            [
                ("prepare_enabled", ()),
                ("MasterSlaveConfig", (0xFA, 0x00, 0x00, 0x00)),
                ("MasterSlaveConfig", (0xFA, 0x00, 0x00, 0x00)),
                ("MotionCtrl_2", (0x00, 0x01, 0)),
                ("ReqMasterArmMoveToHome", (0,)),
                ("recover_output", ()),
                ("prepare_disabled", ()),
                ("MasterSlaveConfig", (0xFA, 0x00, 0x00, 0x00)),
            ],
        )
        self.assertEqual(driver._role, "MASTER")

    def test_takeover_master_failure_restores_output_without_disable_fallback(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._enable_requested = True
        driver._role = "OUTPUT"
        driver._last_feedback = [0.0] * 7
        driver._failed_role_request = None
        driver._lock = threading.RLock()
        driver.master_enable_fallback = True
        events = []
        driver._set_state = lambda state: events.append(("state", state))
        driver._read_role_pose = lambda _role: [0.0] * 7
        driver._prepare_enabled_output_for_master = (
            lambda: events.append(("prepare_enabled", None)) or [0.0] * 7
        )
        driver._activate_master_stream = lambda *_args: False
        driver._role_stream_active = lambda _role: False
        driver._enter_output_role = (
            lambda: events.append(("restore_output", None)) or True
        )
        driver._prepare_disabled_output_for_master = (
            lambda: events.append(("prepare_disabled", None)) or [0.0] * 7
        )

        self.assertFalse(driver._enter_master_role(allow_disable_fallback=False))
        self.assertIn(("restore_output", None), events)
        self.assertIn(("state", "TEACH_FAILED"), events)
        self.assertNotIn(("prepare_disabled", None), events)

    def test_follower_restore_repeats_measured_pose(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.follower_restore_movej_repeats = 6
        driver.follower_restore_movej_period = 0.000001
        driver.move_speed_percent = 100
        driver.piper = _FakePiper()
        seeds = []
        driver._seed_output_target = (
            lambda pose, gripper_code=0x00: seeds.append(
                (list(pose), gripper_code)
            )
            or True
        )
        pose = [0.0] * 7

        self.assertTrue(driver._repeat_follower_restore_target(pose))
        self.assertEqual(len(seeds), 6)
        self.assertTrue(all(seed == (pose, 0x01) for seed in seeds))
        mode_calls = [
            call
            for call in driver.piper.calls
            if call == ("MotionCtrl_2", (0x01, 0x01, 100))
        ]
        self.assertEqual(len(mode_calls), 6)

    def test_first_output_enable_applies_firmware_restore_sequence(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._raw_master_observer = None
        driver._last_feedback = [0.0] * 7
        driver.piper = _FakePiper()
        driver.move_speed_percent = 100
        driver.follower_restore_movej_repeats = 20
        events = []
        driver._read_role_pose = lambda _role: [0.0] * 7
        driver._settle_stream = lambda: None
        driver._seed_output_target = (
            lambda _pose, gripper_code=0x00: events.append(
                ("seed", gripper_code)
            )
            or True
        )
        driver._enable_and_wait = lambda: events.append(("enable", None)) or True
        driver._repeat_follower_restore_target = (
            lambda _pose: events.append(("restore", None)) or True
        )

        self.assertTrue(driver._arm_output_at_current_pose())
        self.assertEqual(events[-2:], [("seed", 0x01), ("restore", None)])


class LeaderDragTeachSequenceTest(unittest.TestCase):
    def _driver(self):
        driver = MODULE.PiperLeaderModeDriver.__new__(MODULE.PiperLeaderModeDriver)
        driver.can_port = "fake_can"
        driver._enable_requested = True
        driver._role = "OUTPUT"
        driver._last_feedback = [0.0] * 7
        driver._output_enable_verified = True
        driver._command_output = [0.0] * 7
        driver._failed_role_request = None
        driver._drag_teach_active = False
        driver._drag_teach_status_lost_at = 0.0
        driver._lock = threading.RLock()
        driver.role_config_retries = 2
        driver.piper = _FakePiper()
        driver._read_role_pose = lambda _role: [0.0] * 7
        driver._wait_for_drag_teach_mode = lambda expected: (
            driver.piper.ctrl_mode == 0x02
        ) == expected
        driver._arm_output_at_current_pose = lambda: True
        states = []
        driver._set_state = states.append
        return driver, states

    def test_takeover_uses_drag_teach_without_role_write_or_disable(self):
        driver, states = self._driver()

        self.assertTrue(driver._enter_drag_teach())
        self.assertEqual(driver._role, "OUTPUT")
        self.assertTrue(driver._drag_teach_active)
        self.assertEqual(states[-1], "DRAG_TEACH")
        self.assertIn(("MotionCtrl_1", (0x00, 0x00, 0x01)), driver.piper.calls)
        self.assertIn(("GripperCtrl", (0, 1000, 0x00, 0)), driver.piper.calls)
        self.assertFalse(
            any(
                name in ("MasterSlaveConfig", "DisableArm")
                for name, _ in driver.piper.calls
            )
        )

    def test_drag_teach_requires_status_acknowledgement(self):
        driver, states = self._driver()
        driver._wait_for_drag_teach_mode = lambda _expected: False
        driver._restore_output_hold = lambda _reference: True

        self.assertFalse(driver._enter_drag_teach())
        self.assertEqual(states[-1], "TEACH_FAILED")
        teach_calls = [
            call
            for call in driver.piper.calls
            if call == ("MotionCtrl_1", (0x00, 0x00, 0x01))
        ]
        self.assertEqual(len(teach_calls), 2)
        self.assertFalse(
            any(name == "MasterSlaveConfig" for name, _ in driver.piper.calls)
        )


if __name__ == "__main__":
    unittest.main()
