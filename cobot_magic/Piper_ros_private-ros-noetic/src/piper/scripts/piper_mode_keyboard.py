#!/usr/bin/env python3
"""Terminal keyboard controller for the split Piper operating sessions."""

import select
import sys
import termios
import tty

import rospy
from std_msgs.msg import Empty, String


KEY_TO_MODE = {
    "i": "INFERENCE",
    "I": "INFERENCE",
    "t": "TAKEOVER",
    "T": "TAKEOVER",
}


def main():
    rospy.init_node("piper_mode_keyboard")
    profile = str(rospy.get_param("~profile", "inference")).strip().lower()
    if profile not in ("inference", "collect", "follower"):
        raise ValueError("~profile must be 'inference', 'collect' or 'follower'")
    publisher = rospy.Publisher("/teleop/mode_request", String, queue_size=10)
    home_publisher = rospy.Publisher("/teleop/home_request", Empty, queue_size=1)
    hold_publisher = rospy.Publisher("/teleop/hold_request", Empty, queue_size=1)
    emergency_stop_publisher = rospy.Publisher(
        "/teleop/emergency_stop_request", Empty, queue_size=1
    )
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard node needs an interactive terminal")

    old_settings = termios.tcgetattr(sys.stdin)
    if profile == "inference":
        print(
            "Piper inference keys: i=INFERENCE, t=TAKEOVER "
            "(1s hand-ready delay), Space=software HOLD, "
            "h=HOME from HOLD, q=HOLD + quit keyboard"
        )
    elif profile == "collect":
        print(
            "Piper fixed-collection keys: Space=QUICK E-STOP, "
            "q=stop collection + quit (C/I/T/H are disabled)"
        )
    else:
        print(
            "Piper follower-only inference keys: i=resume INFERENCE, "
            "Space=software HOLD, h=HOME from HOLD, "
            "q=HOLD + quit keyboard (T is disabled)"
        )
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key in ("q", "Q"):
                if profile == "collect":
                    publisher.publish(String(data="IDLE"))
                else:
                    # Leave the command mux alive and the motor position loops
                    # enabled. Quitting the keyboard must not make an arm fall.
                    hold_publisher.publish(Empty())
                rospy.sleep(0.1)
                print("requested HOLD; keyboard is exiting, arm launch stays active")
                break
            if key in ("h", "H"):
                if profile == "collect":
                    print("HOME is disabled in the fixed-collection session")
                    continue
                home_publisher.publish(Empty())
                print("requested controlled HOME")
                continue
            if key == " ":
                if profile == "collect":
                    emergency_stop_publisher.publish(Empty())
                    print("requested QUICK EMERGENCY STOP; support the followers")
                else:
                    # Operator pause is deliberately not Piper 0x150 quick
                    # stop. Protocol 0x150 enters damping and can let an
                    # unbraked arm fall. HOLD freezes the measured pose while
                    # keeping the position loop enabled.
                    hold_publisher.publish(Empty())
                    print("requested software HOLD; press h for HOME or i to resume")
                continue
            mode = KEY_TO_MODE.get(key)
            if mode is not None:
                if profile == "collect" or (
                    profile == "follower" and mode == "TAKEOVER"
                ):
                    print("{} is disabled in the {} profile".format(mode, profile))
                    continue
                publisher.publish(String(data=mode))
                rospy.logwarn(
                    "Keyboard sent %s on /teleop/mode_request (subscribers=%d)",
                    mode,
                    publisher.get_num_connections(),
                )
                if mode == "TAKEOVER":
                    print(
                        "requested TAKEOVER; wait for 'TAKEOVER accepted' in the "
                        "launch terminal, then hold both leaders during the "
                        "one-second delay; move them only after 'TAKEOVER active'"
                    )
                else:
                    print("requested {}".format(mode))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
