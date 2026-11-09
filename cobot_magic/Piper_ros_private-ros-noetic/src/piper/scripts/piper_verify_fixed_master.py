#!/usr/bin/env python3
"""Read-only preflight for power-cycled Piper 0xFA MASTER leaders.

This utility opens raw SocketCAN receive sockets and never transmits a frame.
Run it with all Piper ROS/SDK clients stopped.  Collection is allowed only when
one complete legal control group is continuously present on each independent
leader bus. Protocol 0x470 permits 0x155-0x159, 0x165-0x169 or 0x175-0x179.
The gripper member is reported separately because stationary-gripper behaviour
depends on firmware.
"""

import argparse
import select
import socket
import struct
import time


CAN_FRAME_FORMAT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)
CAN_ID_MASK = 0x1FFFFFFF
MASTER_GROUPS = {
    0x00: ((0x155, 0x156, 0x157), 0x159),
    0x10: ((0x165, 0x166, 0x167), 0x169),
    0x20: ((0x175, 0x176, 0x177), 0x179),
}
MASTER_IDS = tuple(
    can_id
    for joint_ids, gripper_id in MASTER_GROUPS.values()
    for can_id in joint_ids + (gripper_id,)
)


def verify_can_interface(name):
    try:
        socket.if_nametoindex(name)
    except OSError as exc:
        raise RuntimeError("CAN interface {} does not exist".format(name)) from exc
    with open("/sys/class/net/{}/operstate".format(name), "r", encoding="utf-8") as stream:
        state = stream.read().strip().lower()
    if state not in ("up", "unknown"):
        raise RuntimeError("CAN interface {} is not UP ({})".format(name, state))


def open_receive_socket(name):
    verify_can_interface(name)
    receiver = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    receiver.setblocking(False)
    receiver.bind((name,))
    return receiver


def sample(can_names, duration_s):
    sockets = {}
    counts = {name: {can_id: 0 for can_id in MASTER_IDS} for name in can_names}
    all_counts = {name: {} for name in can_names}
    try:
        for name in can_names:
            sockets[open_receive_socket(name)] = name
        started = time.monotonic()
        deadline = started + duration_s
        while time.monotonic() < deadline:
            timeout = min(0.10, max(0.0, deadline - time.monotonic()))
            readable, _, _ = select.select(list(sockets), [], [], timeout)
            for receiver in readable:
                while True:
                    try:
                        frame = receiver.recv(CAN_FRAME_SIZE)
                    except BlockingIOError:
                        break
                    if len(frame) != CAN_FRAME_SIZE:
                        continue
                    can_id, _dlc, _data = struct.unpack(CAN_FRAME_FORMAT, frame)
                    can_id &= CAN_ID_MASK
                    all_counts[sockets[receiver]][can_id] = (
                        all_counts[sockets[receiver]].get(can_id, 0) + 1
                    )
                    if can_id in counts[sockets[receiver]]:
                        counts[sockets[receiver]][can_id] += 1
        return counts, all_counts, max(0.001, time.monotonic() - started)
    finally:
        for receiver in sockets:
            receiver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Read-only verification of two power-cycled 0xFA MASTER streams"
    )
    parser.add_argument("--left-can", default="can_l_leader")
    parser.add_argument("--right-can", default="can_r_leader")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--min-joint-hz", type=float, default=10.0)
    parser.add_argument(
        "--require-gripper",
        action="store_true",
        help="also fail when the detected group's gripper frame is below min-joint-hz",
    )
    args = parser.parse_args()
    if args.left_can == args.right_can:
        parser.error("left and right CAN interfaces must be different")
    if args.duration_s <= 0.0 or args.min_joint_hz <= 0.0:
        parser.error("duration and minimum rate must be positive")

    names = (args.left_can, args.right_can)
    print("READ-ONLY preflight: no CAN frame will be transmitted.")
    print("Stop all Piper ROS/SDK clients before interpreting these rates.")
    try:
        counts, all_counts, elapsed = sample(names, args.duration_s)
    except Exception as exc:
        print("ERROR: {}".format(exc))
        raise SystemExit(1)

    failures = []
    for name in names:
        rates = {can_id: counts[name][can_id] / elapsed for can_id in MASTER_IDS}
        valid_offsets = []
        print("{}:".format(name))
        for offset, (joint_ids, gripper_id) in MASTER_GROUPS.items():
            print(
                "  offset 0x{:02X}: {}".format(
                    offset,
                    " ".join(
                        "0x{:03X}={:.1f}Hz".format(can_id, rates[can_id])
                        for can_id in joint_ids + (gripper_id,)
                    ),
                )
            )
            joints_valid = all(
                rates[can_id] >= args.min_joint_hz for can_id in joint_ids
            )
            gripper_valid = rates[gripper_id] >= args.min_joint_hz
            if joints_valid and (gripper_valid or not args.require_gripper):
                valid_offsets.append(offset)
        if not valid_offsets:
            observed = sorted(
                all_counts[name].items(), key=lambda item: item[1], reverse=True
            )[:8]
            observed_text = (
                ", ".join(
                    "0x{:03X}={:.1f}Hz".format(can_id, count / elapsed)
                    for can_id, count in observed
                )
                if observed
                else "no CAN frame at all"
            )
            failures.append(
                "{} has no complete 0x15x/0x16x/0x17x group; observed {}".format(
                    name, observed_text
                )
            )
            continue
        selected = valid_offsets[0]
        joint_ids, gripper_id = MASTER_GROUPS[selected]
        print("  DETECTED MASTER control offset: 0x{:02X}".format(selected))
        if rates[gripper_id] < args.min_joint_hz:
            print(
                "  WARNING: joint MASTER stream is valid, but 0x{:03X} is not "
                "continuous; verify gripper recording before collecting."
                .format(gripper_id)
            )

    if failures:
        print("MASTER preflight FAILED; do not start collection:")
        for failure in failures:
            print("  {}".format(failure))
        print(
            "If all three legal groups are zero after saving 0xFA and fully "
            "power-cycling both leader controllers, the remaining issue is the "
            "controller/firmware role transition, not ROS topic wiring."
        )
        raise SystemExit(1)

    print("MASTER preflight PASSED for both leaders; fixed collection may be started.")


if __name__ == "__main__":
    main()
