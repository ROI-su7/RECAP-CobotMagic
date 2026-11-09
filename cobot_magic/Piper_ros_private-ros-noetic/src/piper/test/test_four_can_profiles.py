#!/usr/bin/env python3
"""Offline wiring tests for the mutually exclusive four-CAN profiles.

The tests parse source/XML only.  They never start ROS nodes or open SocketCAN.
"""

import ast
import importlib.util
import math
import pathlib
import struct
import threading
import time
import unittest
import xml.etree.ElementTree as ET


PACKAGE = pathlib.Path(__file__).parents[1]
WORKSPACE = PACKAGE.parents[2]
COLLECTOR = WORKSPACE / "collect_data" / "new_collect_data.py"
FIXED_MUX = WORKSPACE / "lingbot_piper"


def _launch(name):
    return ET.parse(str(PACKAGE / "launch" / name)).getroot()


def _arg_value(root, name):
    matches = root.findall(".//arg[@name='{}']".format(name))
    if not matches:
        raise AssertionError("missing launch arg {}".format(name))
    return matches[-1].get("value", matches[-1].get("default"))


def _parser_default(script_path, option):
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == option
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError("missing argparse default for {}".format(option))


class FourCanProfileTest(unittest.TestCase):
    def test_collection_uses_read_only_fixed_master_profile(self):
        root = _launch("start_four_can_collect.launch")
        node_types = {node.get("type") for node in root.findall(".//node")}
        self.assertEqual(
            node_types,
            {
                "piper_fixed_master_reader.py",
                "piper_collect_slave_node.py",
                "piper_fixed_collect_mux.py",
            },
        )
        self.assertNotIn("piper_leader_mode_driver.py", node_types)

        reader_text = (
            PACKAGE / "scripts" / "piper_fixed_master_reader.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("MasterSlaveConfig", reader_text)
        self.assertNotIn("EnableArm", reader_text)
        self.assertNotIn("MotionCtrl", reader_text)

    def test_collection_slew_limits_do_not_undercut_the_follower(self):
        # Measured follower capability is ~3.0 rad/s per joint and ~0.22 m/s
        # at the gripper; a lower limit adds lag the hardware did not ask for.
        root = _launch("start_four_can_collect.launch")
        self.assertGreaterEqual(
            float(_arg_value(root, "max_joint_velocity_rad_s")), 3.0
        )
        self.assertGreaterEqual(
            float(_arg_value(root, "max_gripper_velocity_m_s")), 0.22
        )

    def test_collection_slew_limiter_is_anti_windup(self):
        root = _launch("start_four_can_collect.launch")
        for name in ("max_joint_tracking_lag_rad", "max_gripper_tracking_lag_m"):
            self.assertGreater(float(_arg_value(root, name)), 0.0)
        text = (PACKAGE / "scripts" / "piper_fixed_collect_mux.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_joint_tracking_lag_rad", text)
        self.assertIn("max_gripper_tracking_lag_m", text)

    def test_master_preflight_is_receive_only(self):
        text = (PACKAGE / "scripts" / "piper_verify_fixed_master.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("receiver.recv", text)
        self.assertNotIn(".send(", text)
        self.assertNotIn("MasterSlaveConfig", text)
        self.assertIn("0x165", text)
        self.assertIn("0x175", text)

    def test_fixed_master_reader_accepts_all_protocol_offsets(self):
        text = (PACKAGE / "scripts" / "piper_fixed_master_reader.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("CONTROL_OFFSETS = (0x00, 0x10, 0x20)", text)
        self.assertIn("socket.CAN_RAW", text)
        self.assertNotIn("piper_sdk", text)

    def test_fixed_master_reader_decodes_shifted_joint_group(self):
        path = PACKAGE / "scripts" / "piper_fixed_master_reader.py"
        spec = importlib.util.spec_from_file_location("fixed_master_reader_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reader = module.FixedMasterReader.__new__(module.FixedMasterReader)
        reader.stream_timeout = 0.15
        reader._lock = threading.Lock()
        reader._positions = {
            offset: [0.0] * 7 for offset in module.CONTROL_OFFSETS
        }
        reader._seen_at = {offset: {} for offset in module.CONTROL_OFFSETS}
        reader._counts = {offset: {} for offset in module.CONTROL_OFFSETS}
        reader._active_offset = None
        reader._last_pose = None
        now = time.monotonic()
        values = (1000, -2000, 3000, -4000, 5000, -6000)
        for can_id, pair in zip((0x165, 0x166, 0x167), ((0, 1), (2, 3), (4, 5))):
            reader._process_frame(
                can_id,
                struct.pack(">ii", values[pair[0]], values[pair[1]]),
                now,
            )
        reader._process_frame(0x169, struct.pack(">i", 40000) + b"\x00" * 4, now)
        self.assertTrue(reader._sample(now))
        self.assertEqual(reader._active_offset, 0x10)
        self.assertAlmostEqual(reader._last_pose[0], math.pi / 180.0)
        self.assertAlmostEqual(reader._last_pose[1], -2.0 * math.pi / 180.0)
        self.assertAlmostEqual(reader._last_pose[6], 0.04)

    def test_fixed_master_reader_holds_last_pose_when_master_is_stationary(self):
        path = PACKAGE / "scripts" / "piper_fixed_master_reader.py"
        spec = importlib.util.spec_from_file_location("fixed_master_hold_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reader = module.FixedMasterReader.__new__(module.FixedMasterReader)
        reader.stream_timeout = 0.15
        reader._lock = threading.Lock()
        reader._positions = {offset: [0.0] * 7 for offset in module.CONTROL_OFFSETS}
        reader._seen_at = {offset: {} for offset in module.CONTROL_OFFSETS}
        reader._counts = {offset: {} for offset in module.CONTROL_OFFSETS}
        reader._active_offset = None
        reader._last_pose = None
        reader._last_frame_at = 0.0
        now = time.monotonic()
        for can_id in (0x155, 0x156, 0x157):
            reader._process_frame(can_id, struct.pack(">ii", 0, 0), now)
        self.assertTrue(reader._sample(now, require_fresh=True))
        self.assertTrue(reader._sample(now + 10.0, require_fresh=False))

    def test_master_role_maintenance_uses_protocol_activation_sequence(self):
        text = (
            PACKAGE / "scripts" / "piper_configure_leader_role.py"
        ).read_text(encoding="utf-8")
        configure_start = text.index("def configure_master")
        configure_end = text.index("\ndef main", configure_start)
        configure_text = text[configure_start:configure_end]
        self.assertLess(
            configure_text.index("MasterSlaveConfig(0xFA"),
            configure_text.index("MotionCtrl_2(0x00"),
        )
        self.assertIn("wait_for_master_stream", configure_text)
        self.assertNotIn(".EnableArm(", configure_text)
        self.assertNotIn(".DisableArm(", configure_text)
        self.assertNotIn(".JointCtrl(", configure_text)
        self.assertNotIn(".GripperCtrl(", configure_text)

    def test_default_master_maintenance_uses_official_save_and_restart_path(self):
        path = PACKAGE / "scripts" / "piper_configure_leader_role.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        activation_default = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--master-activation"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default":
                    activation_default = ast.literal_eval(keyword.value)
        self.assertEqual(activation_default, "restart")
        start = text.index("def save_master_for_restart")
        end = text.index("\ndef main", start)
        save_text = text[start:end]
        self.assertIn("ConnectPort(piper_init=True)", save_text)
        self.assertIn("MasterSlaveConfig(0xFA", save_text)
        self.assertNotIn("MotionCtrl", save_text)
        self.assertNotIn("EnableArm", save_text)
        self.assertNotIn("DisableArm", save_text)

    def test_output_restore_always_persists_fc(self):
        text = (
            PACKAGE / "scripts" / "piper_configure_leader_role.py"
        ).read_text(encoding="utf-8")
        start = text.index("def restore_output")
        end = text.index("\ndef configure_master", start)
        restore_text = text[start:end]
        self.assertLess(
            restore_text.index("MasterSlaveConfig(0xFC"),
            restore_text.index("wait_for_output_stream(piper, verify_timeout_s)"),
        )

    def test_follower_only_inference_never_opens_a_leader_driver(self):
        root = ET.parse(str(FIXED_MUX / "launch" / "start_inference_followers.launch")).getroot()
        node_types = {node.get("type") for node in root.findall(".//node")}
        self.assertEqual(
            node_types,
            {"piper_start_slave_node.py", "piper_policy_follower_mux.py"},
        )
        text = (FIXED_MUX / "launch" / "start_inference_followers.launch").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("can_l_leader", text)
        self.assertNotIn("can_r_leader", text)

    def test_follower_only_enable_is_gated_by_policy_mux(self):
        root = ET.parse(str(FIXED_MUX / "launch" / "start_inference_followers.launch")).getroot()
        follower_nodes = [
            node
            for node in root.findall(".//node")
            if node.get("type") == "piper_start_slave_node.py"
        ]
        self.assertEqual(len(follower_nodes), 2)
        for node in follower_nodes:
            remaps = {
                remap.get("from"): remap.get("to")
                for remap in node.findall("remap")
            }
            self.assertEqual(
                remaps.get("/enable_flag"), "/teleop/follower_enable"
            )

        mux_text = (
            FIXED_MUX / "scripts" / "piper_policy_follower_mux.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"/teleop/follower_enable"', mux_text)
        self.assertIn("disabled; missing streams", mux_text)

    def test_collection_startup_wait_does_not_send_quick_estop(self):
        text = (PACKAGE / "scripts" / "piper_fixed_collect_mux.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("followers remain command-inhibited", text)
        self.assertNotIn(
            'self._latch_emergency(\n'
            '                                "fixed MASTER/follower streams were not ready before startup timeout"',
            text,
        )

    def test_collection_waits_for_verified_follower_enable(self):
        root = _launch("start_four_can_collect.launch")
        follower_nodes = [
            node
            for node in root.findall(".//node")
            if node.get("type") == "piper_collect_slave_node.py"
        ]
        self.assertEqual(len(follower_nodes), 2)
        for side, node in zip(("left", "right"), follower_nodes):
            remaps = {
                remap.get("from"): remap.get("to")
                for remap in node.findall("remap")
            }
            self.assertEqual(
                remaps.get("/enable_flag"), "/teleop/follower_enable_request"
            )
            self.assertEqual(
                remaps.get("/teleop/follower_enabled"),
                "/teleop/follower_enabled_{}".format(side),
            )

        mux_text = (
            PACKAGE / "scripts" / "piper_fixed_collect_mux.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"/teleop/follower_enable_request"', mux_text)
        self.assertIn('"/teleop/follower_enabled_{}"', mux_text)
        self.assertIn("followers_enabled = all", mux_text)

    def test_collection_stale_input_enters_powered_hold_not_estop(self):
        text = (PACKAGE / "scripts" / "piper_fixed_collect_mux.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_enter_powered_hold", text)
        self.assertIn("no E-STOP was sent", text)
        self.assertNotIn(
            'self._latch_emergency("a leader/follower input became stale")', text
        )

    def test_collection_driver_retries_without_changing_shared_driver(self):
        text = (PACKAGE / "scripts" / "piper_collect_slave_node.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class CollectPiperRosNode(C_PiperRosNode)", text)
        self.assertIn("collection enable is still unconfirmed", text)
        self.assertIn("lost physical enable confirmation", text)

    def test_inference_keyboard_pause_is_powered_hold(self):
        keyboard_text = (
            PACKAGE / "scripts" / "piper_mode_keyboard.py"
        ).read_text(encoding="utf-8")
        follower_text = (
            FIXED_MUX / "scripts" / "piper_policy_follower_mux.py"
        ).read_text(encoding="utf-8")
        four_arm_text = (
            PACKAGE / "scripts" / "piper_four_state_mux.py"
        ).read_text(encoding="utf-8")
        leader_text = (
            PACKAGE / "scripts" / "piper_leader_mode_driver.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"/teleop/hold_request"', keyboard_text)
        self.assertIn('"/teleop/hold_request"', follower_text)
        self.assertIn('"/teleop/hold_request"', four_arm_text)
        self.assertIn('"HOLD", "COLLECT"', leader_text)
        self.assertIn('output_mode in ("INFERENCE", "HOLD")', four_arm_text)
        self.assertIn("followers remain enabled and position-held", follower_text)

    def test_initial_hold_cannot_enable_a_disabled_leader(self):
        leader_text = (
            PACKAGE / "scripts" / "piper_leader_mode_driver.py"
        ).read_text(encoding="utf-8")
        self.assertIn('desired_mode == "HOLD"', leader_text)
        self.assertIn("and self._output_enable_verified", leader_text)

    def test_takeover_key_enters_powered_hold_before_master_switch(self):
        keyboard_text = (
            PACKAGE / "scripts" / "piper_mode_keyboard.py"
        ).read_text(encoding="utf-8")
        mux_text = (
            PACKAGE / "scripts" / "piper_four_state_mux.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"T": "TAKEOVER"', keyboard_text)
        self.assertIn("Keyboard sent %s on /teleop/mode_request", keyboard_text)
        self.assertIn('self._mode = "HOLD"', mux_text)
        self.assertIn('self._hold_active_pub.publish(Bool(data=True))', mux_text)
        self.assertIn('self._set_inference_active(False)', mux_text)
        self.assertIn("TAKEOVER accepted: INFERENCE -> powered HOLD", mux_text)
        self.assertIn('"/teleop/policy_hold_ack"', mux_text)
        self.assertIn("OpenPI did not acknowledge", mux_text)

    def test_keyboard_reports_the_actual_profile_that_disables_takeover(self):
        keyboard_text = (
            PACKAGE / "scripts" / "piper_mode_keyboard.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"{} is disabled in the {} profile".format(mode, profile)', keyboard_text)

    def test_takeover_uses_stream_confirmed_master_without_disable(self):
        mux_text = (
            PACKAGE / "scripts" / "piper_four_state_mux.py"
        ).read_text(encoding="utf-8")
        leader_text = (
            PACKAGE / "scripts" / "piper_leader_mode_driver.py"
        ).read_text(encoding="utf-8")
        self.assertIn('state in ("MASTER", "MASTER_ARMED")', mux_text)
        self.assertIn('desired_mode in ("COLLECT", "TAKEOVER")', leader_text)
        self.assertIn(
            'allow_disable_fallback=desired_mode != "TAKEOVER"', leader_text
        )
        activation_start = leader_text.index("def _activate_master_stream")
        activation_end = leader_text.index(
            "\n    def _enter_output_role", activation_start
        )
        activation_text = leader_text[activation_start:activation_end]
        self.assertIn("MasterSlaveConfig(0xFA", activation_text)
        self.assertIn("self.piper.MotionCtrl_2(0x00, 0x01, 0)", activation_text)
        self.assertNotIn("DisableArm", activation_text)

    def test_takeover_retries_a_completely_silent_first_master_write(self):
        root = _launch("start_four_can_mux.launch")
        self.assertEqual(
            _arg_value(root, "leader_master_first_no_stream_retry_s"), "4.0"
        )
        self.assertEqual(_arg_value(root, "leader_role_config_retries"), "2")
        self.assertEqual(
            _arg_value(root, "leader_master_standby_activation"), "true"
        )
        self.assertEqual(
            _arg_value(root, "leader_master_restore_linkage_activation"), "false"
        )
        self.assertEqual(
            _arg_value(root, "leader_follower_restore_movej_repeats"), "20"
        )
        self.assertEqual(_arg_value(root, "takeover_release_delay_s"), "1.0")
        self.assertEqual(_arg_value(root, "takeover_policy_quiet_s"), "0.20")
        self.assertEqual(_arg_value(root, "takeover_role_timeout_s"), "15.0")
        leader_text = (
            PACKAGE / "scripts" / "piper_leader_mode_driver.py"
        ).read_text(encoding="utf-8")
        self.assertIn("no_stream_retry_s", leader_text)
        self.assertIn("attempt < self.role_config_retries", leader_text)
        self.assertIn("produced no new joint frames", leader_text)
        self.assertIn("protocol-required", leader_text)

    def test_takeover_stops_policy_before_leader_ownership(self):
        mux_text = (
            PACKAGE / "scripts" / "piper_four_state_mux.py"
        ).read_text(encoding="utf-8")
        self.assertIn('self._mode = "HOLD"', mux_text)
        self.assertIn("self._set_inference_active(False)", mux_text)
        self.assertIn("takeover_policy_quiet_s", mux_text)
        self.assertIn("takeover_policy_ack_timeout_s", mux_text)
        self.assertIn("takeover_leader_tx_quiet_s", mux_text)
        self.assertIn("policy_quiet", mux_text)
        self.assertIn("self._begin_takeover_role_switch(now)", mux_text)
        self.assertIn('"/teleop/takeover_phase"', mux_text)
        self.assertIn('self._set_takeover_phase("POLICY_STOP_REQUEST")', mux_text)
        self.assertIn('self._set_takeover_phase("DRAINING_OUTPUT")', mux_text)
        self.assertIn('self._set_takeover_phase("LEADER_TX_QUIET")', mux_text)
        self.assertIn('self._set_takeover_phase("ACTIVE")', mux_text)

    def test_takeover_waits_for_both_leader_send_barriers(self):
        root = _launch("start_four_can_mux.launch")
        mux_text = (
            PACKAGE / "scripts" / "piper_four_state_mux.py"
        ).read_text(encoding="utf-8")
        leader_text = (
            PACKAGE / "scripts" / "piper_leader_mode_driver.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(_arg_value(root, "takeover_policy_ack_timeout_s"), "2.0")
        self.assertEqual(_arg_value(root, "takeover_leader_tx_quiet_s"), "0.20")
        self.assertIn('"/teleop/leader_tx_enable"', mux_text)
        self.assertIn('"/teleop/leader/tx_quiet_left"', mux_text)
        self.assertIn('"/teleop/leader/tx_quiet_right"', mux_text)
        self.assertIn("self._tx_lock = threading.Lock()", leader_text)
        self.assertIn('"/leader/tx_quiet"', leader_text)
        self.assertIn("if self._leader_tx_enabled", leader_text)
        pending_start = mux_text.index("def _process_pending_takeover")
        pending_end = mux_text.index("\n    def _abort_pending_takeover", pending_start)
        pending_text = mux_text[pending_start:pending_end]
        # OUTPUT hold must remain refreshed for the full operator delay. Only
        # then may alignment be latched and the short CAN drain begin.
        self.assertLess(
            pending_text.index("elapsed >= self.takeover_release_delay"),
            pending_text.index("self._set_leader_tx_enabled(False)"),
        )
        self.assertIn("self._takeover_alignment_latched = True", pending_text)
        self.assertIn("and leader_tx_quiet", pending_text)
        self.assertNotIn(
            "self._takeover_ready\n            and leader_tx_quiet",
            pending_text,
        )

    def test_hold_resume_requires_post_i_policy_messages(self):
        follower_text = (
            FIXED_MUX / "scripts" / "piper_policy_follower_mux.py"
        ).read_text(encoding="utf-8")
        four_arm_text = (
            PACKAGE / "scripts" / "piper_four_state_mux.py"
        ).read_text(encoding="utf-8")
        self.assertIn("WAITING_POLICY_RESYNC", follower_text)
        self.assertIn("> self._resume_requested_at", follower_text)
        self.assertIn("self._policy_resume_after", four_arm_text)
        self.assertIn("> self._policy_resume_after", four_arm_text)

    def test_supervised_preflight_gate_is_owned_by_mux(self):
        root = _launch("start_four_can_mux.launch")
        mux = next(
            node
            for node in root.findall(".//node")
            if node.get("type") == "piper_four_state_mux.py"
        )
        mux_params = {param.get("name") for param in mux.findall("param")}
        self.assertIn("supervised_start_joint_tolerance_rad", mux_params)
        self.assertIn("supervised_start_gripper_tolerance_m", mux_params)
        for node in root.findall(".//node"):
            if node.get("type") != "piper_leader_mode_driver.py":
                continue
            leader_params = {param.get("name") for param in node.findall("param")}
            self.assertNotIn("supervised_start_joint_tolerance_rad", leader_params)
            self.assertNotIn("supervised_start_gripper_tolerance_m", leader_params)

    def test_supervised_inference_is_the_only_auto_inference_profile(self):
        root = _launch("start_inference_takeover.launch")
        self.assertEqual(_arg_value(root, "auto_start_inference"), "true")
        self.assertEqual(_arg_value(root, "auto_start_collect"), "false")
        self.assertEqual(_arg_value(root, "session_profile"), "supervised")

    def test_collection_actions_default_to_independent_leaders(self):
        self.assertEqual(
            _parser_default(COLLECTOR, "--master_arm_left_topic"),
            "/teleop/leader/joint_left",
        )
        self.assertEqual(
            _parser_default(COLLECTOR, "--master_arm_right_topic"),
            "/teleop/leader/joint_right",
        )


if __name__ == "__main__":
    unittest.main()
