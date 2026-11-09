#!/usr/bin/env python3
"""Safely configure and verify both physical Piper leader roles.

The Piper CAN protocol says that 0xFA is saved to Flash, but the teaching-input
role becomes active only after a controller restart *or after entering standby*.
The default MASTER path mirrors the official SDK demo: initialise the SDK,
verify the arm is currently OUTPUT, save exactly 0xFA, then require a real
controller power-cycle and a separate raw-CAN preflight. An optional standby
activation path remains available for diagnostics. OUTPUT recovery writes
0xFC, enters standby, and verifies normal 0x2Ax feedback. Neither path enables
motors or sends a joint/gripper target.
"""

import argparse
import socket
import time

from piper_sdk import C_PiperInterface_V2


def verify_can_interface(name):
    try:
        socket.if_nametoindex(name)
    except OSError as exc:
        raise RuntimeError("CAN interface {} does not exist".format(name)) from exc
    state_path = "/sys/class/net/{}/operstate".format(name)
    with open(state_path, "r", encoding="utf-8") as state_file:
        state = state_file.read().strip().lower()
    if state not in ("up", "unknown"):
        raise RuntimeError("CAN interface {} is not UP ({})".format(name, state))


def wait_for_output_stream(piper, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        joint = piper.GetArmJointMsgs()
        gripper = piper.GetArmGripperMsgs()
        if (
            float(joint.time_stamp) > 0.0
            and float(gripper.time_stamp) > 0.0
            and float(joint.Hz) > 0.0
            and float(gripper.Hz) > 0.0
        ):
            return float(joint.Hz), float(gripper.Hz)
        time.sleep(0.02)
    raise RuntimeError(
        "no complete OUTPUT 0x2A5-0x2A8 stream within {:.1f}s".format(timeout_s)
    )


def wait_for_master_stream(piper, timeout_s, before_timestamp):
    """Require a new complete 0x155-0x157 group after the role command."""
    deadline = time.monotonic() + timeout_s
    last_timestamp = 0.0
    last_hz = 0.0
    gripper_hz = 0.0
    while time.monotonic() < deadline:
        joint = piper.GetArmJointCtrl()
        gripper = piper.GetArmGripperCtrl()
        last_timestamp = float(joint.time_stamp)
        last_hz = float(joint.Hz)
        gripper_hz = float(gripper.Hz)
        if last_timestamp > before_timestamp and last_hz > 0.0:
            return last_hz, gripper_hz
        time.sleep(0.02)
    raise RuntimeError(
        "no fresh complete MASTER 0x155-0x157 stream within {:.1f}s "
        "(joint_stamp={:.3f}, joint_hz={:.1f}, gripper_hz={:.1f})".format(
            timeout_s, last_timestamp, last_hz, gripper_hz
        )
    )


def wait_for_firmware_version(piper, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        version = piper.GetPiperFirmwareVersion()
        if isinstance(version, str) and version.startswith("S-V"):
            return version
        time.sleep(0.02)
    return "UNKNOWN"


def restore_output(can_name, settle_s, verify_timeout_s):
    """Persist OUTPUT with 0xFC, even when runtime feedback already looks normal.

    A controller can be in temporary OUTPUT feedback while its Flash role is
    still 0xFA. Returning early on the existing 0x2Ax stream therefore leaves
    the next power-cycle in gravity-compensated MASTER state.
    """
    verify_can_interface(can_name)
    piper = C_PiperInterface_V2(can_name=can_name)
    connected = False
    try:
        piper.ConnectPort(piper_init=False)
        connected = True
        time.sleep(settle_s)
        # Always persist the requested role. The command is configuration only:
        # it neither enables motors nor sends a joint/gripper target.
        piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
        time.sleep(settle_s)
        piper.MotionCtrl_2(0x00, 0x01, 0)
        time.sleep(settle_s)
        return wait_for_output_stream(piper, verify_timeout_s)
    finally:
        if connected:
            piper.DisconnectPort(thread_timeout=0.2)


def configure_master(can_name, settle_s, verify_timeout_s):
    """Persist 0xFA, enter standby as specified by the protocol, and verify."""
    verify_can_interface(can_name)
    piper = C_PiperInterface_V2(can_name=can_name)
    connected = False
    try:
        piper.ConnectPort(piper_init=False)
        connected = True
        time.sleep(settle_s)
        before_timestamp = float(piper.GetArmJointCtrl().time_stamp)
        piper.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
        time.sleep(settle_s)
        # Protocol 0x470: after setting 0xFA, a restart or a transition to
        # standby is required before linkage teaching-input mode takes effect.
        # This is intentionally standby only: never EnableArm/DisableArm and
        # never send a joint or gripper target from this maintenance utility.
        piper.MotionCtrl_2(0x00, 0x01, 0)
        time.sleep(settle_s)
        return wait_for_master_stream(
            piper, verify_timeout_s, before_timestamp
        )
    finally:
        if connected:
            piper.DisconnectPort(thread_timeout=0.2)


def save_master_for_restart(can_name, settle_s, verify_timeout_s):
    """Use the official minimal save sequence; activation occurs on reboot."""
    verify_can_interface(can_name)
    piper = C_PiperInterface_V2(can_name=can_name)
    connected = False
    try:
        # The official piper_set_master.py demo uses the normal initialisation
        # path before its single role write. PiperInit performs parameter and
        # firmware queries only; it does not enable or command the arm pose.
        piper.ConnectPort(piper_init=True)
        connected = True
        time.sleep(settle_s)
        wait_for_output_stream(piper, min(2.0, verify_timeout_s))
        firmware = wait_for_firmware_version(piper)
        piper.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
        time.sleep(settle_s)
        return firmware
    finally:
        if connected:
            piper.DisconnectPort(thread_timeout=0.2)


def main():
    parser = argparse.ArgumentParser(
        description="Safely configure and verify MASTER or OUTPUT leader roles"
    )
    parser.add_argument("--left-can", default="can_l_leader")
    parser.add_argument("--right-can", default="can_r_leader")
    parser.add_argument("--role", choices=("master", "output"), required=True)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--verify-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--master-activation",
        choices=("restart", "standby"),
        default="restart",
        help=(
            "restart: official minimal 0xFA save followed by a required power-cycle; "
            "standby: attempt immediate 0xFA + standby activation"
        ),
    )
    args = parser.parse_args()

    if args.left_can == args.right_can:
        parser.error("left and right CAN interfaces must be different")
    if args.settle_s <= 0.0 or args.verify_timeout_s <= 0.0:
        parser.error("timeouts must be positive")

    failures = []
    print("STOP all Piper ROS nodes before role maintenance.")
    if args.role == "master":
        if args.master_activation == "restart":
            print(
                "MASTER restart path uses the official minimal 0xFA save. "
                "It will NOT enable, disable, or command either leader pose."
            )
        else:
            print(
                "MASTER standby path sends 0xFA then standby and verifies 0x155-0x157; "
                "it will NOT enable or command either leader pose."
            )
    for can_name in (args.left_can, args.right_can):
        if args.role == "master":
            sequence = "0xFA save + restart" if args.master_activation == "restart" else "0xFA + standby"
            print("Configuring {} as MASTER ({})...".format(can_name, sequence))
        else:
            print("Restoring {} to OUTPUT (0xFC)...".format(can_name))
        try:
            if args.role == "master":
                if args.master_activation == "restart":
                    firmware = save_master_for_restart(
                        can_name, args.settle_s, args.verify_timeout_s
                    )
                    print(
                        "  {} saved 0xFA; firmware={}; controller power-cycle "
                        "and raw verification are still required".format(
                            can_name, firmware
                        )
                    )
                else:
                    joint_hz, gripper_hz = configure_master(
                        can_name, args.settle_s, args.verify_timeout_s
                    )
                    print(
                        "  {} MASTER verified: joint {:.1f} Hz, "
                        "gripper {:.1f} Hz".format(
                            can_name, joint_hz, gripper_hz
                        )
                    )
            else:
                joint_hz, gripper_hz = restore_output(
                    can_name, args.settle_s, args.verify_timeout_s
                )
                print(
                    "  {} OUTPUT verified: joint {:.1f} Hz, "
                    "gripper {:.1f} Hz".format(can_name, joint_hz, gripper_hz)
                )
        except Exception as exc:
            failures.append((can_name, str(exc)))
            print("  ERROR {}: {}".format(can_name, exc))

    if failures:
        print("Role verification failed; do not start a Piper launch.")
        for can_name, reason in failures:
            print("  {}: {}".format(can_name, reason))
        raise SystemExit(1)

    if args.role == "master":
        if args.master_activation == "restart":
            print("Both 0xFA role values were sent using the official SDK sequence.")
            print("Now fully power OFF both leader controllers for 10 seconds, then power ON.")
            print("Do not start any Piper SDK/ROS node before the raw-CAN preflight.")
        else:
            print("Both leaders publish verified MASTER control streams.")
        print("No motors were explicitly enabled/disabled and no pose target was sent.")
    else:
        print("Both leaders publish 0x2Ax feedback; no motors were enabled.")


if __name__ == "__main__":
    main()
