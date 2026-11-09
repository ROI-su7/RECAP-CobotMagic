"""Keep RECAP robot sessions off the factory follower mux.

Two follower-only muxes exist and both are started with ``roslaunch``:

* Old: ``roslaunch piper start_inference_followers.launch`` runs
  cobot_magic's ``piper_policy_follower_mux.py`` (2026-10-12). It reads its
  clock before taking its lock, sees fresh feedback as stale 2-4 ms after
  Space/H/I and latches E-STOP: both followers are disabled and drop
  (2026-10-20 x3, 2026-10-21 x2, with LingBot and Pi05).
* New: ``roslaunch lingbot_piper start_inference_followers.launch`` from
  ``cobot_magic/lingbot_piper`` of this repository keeps torque on
  feedback gaps and publishes ``/teleop/follower_mux_impl`` = ``FIXED_MUX_IMPL``.

Only follower muxes publish ``/teleop/follower_enable``; the four-arm takeover
launch has none (its four-state mux holds with torque on). The marker check is
the same as LingBot's ``deploy.piper_real.mux_guard``.

  python -m rlinf.recap_piper.mux_guard --profile followers|takeover
  python -m rlinf.recap_piper.mux_guard --before-launch --package DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FIXED_MUX_IMPL = "lingbot_piper/piper_policy_follower_mux powered-hold v1"
ENABLE_TOPIC = "/teleop/follower_enable"
IMPL_TOPIC = "/teleop/follower_mux_impl"
MODE_TOPIC = "/teleop/mode"
SAFE_LAUNCH = "bash examples/recap_piper/run.sh launch followers"
TAKEOVER_LAUNCH = "bash examples/recap_piper/run.sh launch takeover"


def mux_problem(publishers: dict, profile: str) -> str | None:
    """Return why ``profile`` must not run on this ROS graph, or None.

    ``publishers`` maps topic -> publisher node names (rosgraph system state).
    """
    muxes = set(publishers.get(ENABLE_TOPIC, ()))
    unmarked = muxes - set(publishers.get(IMPL_TOPIC, ()))
    if unmarked:
        return (
            f"{', '.join(sorted(unmarked))} is cobot_magic's ORIGINAL follower mux "
            "(roslaunch piper start_inference_followers.launch), which disables both "
            "followers on Space/H/I. Support the arms, stop that launch and use: "
            + SAFE_LAUNCH
        )
    if profile == "followers" and not muxes:
        return "no follower mux is running; start it with: " + SAFE_LAUNCH
    if profile == "takeover" and muxes:
        return (
            "a follower-only mux is running, but the takeover profile needs the "
            "four-arm launch: " + TAKEOVER_LAUNCH
        )
    return None


def launch_problem(publishers: dict) -> str | None:
    """Refuse a second Piper launch: equal node names would replace live nodes."""
    owners = set(publishers.get(MODE_TOPIC, ())) | set(publishers.get(ENABLE_TOPIC, ()))
    if owners:
        return (
            f"a Piper mux is already running ({', '.join(sorted(owners))}). Press Space, "
            "support both followers, publish /enable_flag false, stop that launch, "
            "then start this one"
        )
    return None


def package_problem(package_dir: Path) -> str | None:
    """Check that ``package_dir`` holds the fixed (marker-publishing) mux."""
    mux = package_dir / "scripts" / "piper_policy_follower_mux.py"
    launch = package_dir / "launch" / "start_inference_followers.launch"
    for path in (package_dir / "package.xml", mux, launch):
        if not path.is_file():
            return f"{path} is missing; set RECAP_FOLLOWER_MUX_ROS to the lingbot_piper package"
    if "<name>lingbot_piper</name>" not in (package_dir / "package.xml").read_text():
        return f"{package_dir} is not the lingbot_piper package"
    if f'MUX_IMPL = "{FIXED_MUX_IMPL}"' not in mux.read_text():
        return f"{mux} does not publish the fixed-mux marker {FIXED_MUX_IMPL!r}"
    if 'pkg="lingbot_piper"' not in launch.read_text():
        return f"{launch} does not start the lingbot_piper mux"
    return None


def ros_problem(profile: str, own_name: str, wait_s: float = 3.0) -> str | None:
    """Check the live ROS graph (and the latched marker) for ``profile``."""
    import rosgraph
    import rospy
    from std_msgs.msg import String

    publishers = dict(rosgraph.Master(own_name).getSystemState()[0])
    problem = mux_problem(publishers, profile)
    if problem or profile != "followers":
        return problem
    try:
        marker = rospy.wait_for_message(IMPL_TOPIC, String, timeout=wait_s).data
    except rospy.ROSException:
        return (
            f"the follower mux did not report its implementation within {wait_s:.0f} s"
        )
    if marker != FIXED_MUX_IMPL:
        return f"unexpected follower mux implementation {marker!r}; use: {SAFE_LAUNCH}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=("followers", "takeover"))
    parser.add_argument("--before-launch", action="store_true")
    parser.add_argument("--package", type=Path, help="lingbot_piper package to verify")
    args = parser.parse_args()
    problem = package_problem(args.package) if args.package else None
    if problem is None and args.before_launch:
        import rosgraph

        try:
            publishers = dict(rosgraph.Master("/recap_mux_guard").getSystemState()[0])
        except Exception:  # noqa: BLE001 - no master yet: roslaunch starts one
            publishers = {}
        problem = launch_problem(publishers)
    if problem is None and args.profile:
        import rospy

        rospy.init_node("recap_mux_guard", anonymous=True, disable_signals=True)
        problem = ros_problem(args.profile, rospy.get_name())
    if problem:
        print("REFUSED: " + problem, file=sys.stderr)
        sys.exit(1)
    print(f"Piper mux check passed ({args.profile or 'launch'})")


if __name__ == "__main__":
    main()
