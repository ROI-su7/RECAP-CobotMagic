"""RECAP refuses cobot_magic's original follower mux (no ROS or robot needed)."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlinf.recap_piper.collect import RosSamples, normalize_mode
from rlinf.recap_piper.mux_guard import (
    ENABLE_TOPIC,
    FIXED_MUX_IMPL,
    IMPL_TOPIC,
    MODE_TOPIC,
    launch_problem,
    mux_problem,
    package_problem,
)

OLD = {
    ENABLE_TOPIC: ["/piper_policy_follower_mux"],
    MODE_TOPIC: ["/piper_policy_follower_mux"],
}
NEW = {**OLD, IMPL_TOPIC: ["/piper_policy_follower_mux"]}
TAKEOVER = {MODE_TOPIC: ["/piper_four_state_mux"]}
LINGBOT_PKG = Path(__file__).resolve().parents[4] / "cobot_magic" / "lingbot_piper"


@pytest.mark.parametrize("profile", ["followers", "takeover"])
def test_original_follower_mux_is_refused(profile):
    assert "ORIGINAL follower mux" in mux_problem(OLD, profile)


def test_followers_profile_needs_the_fixed_mux():
    assert mux_problem(NEW, "followers") is None
    assert "no follower mux" in mux_problem(TAKEOVER, "followers")


def test_takeover_profile_needs_the_four_arm_launch():
    assert mux_problem(TAKEOVER, "takeover") is None
    assert "four-arm launch" in mux_problem(NEW, "takeover")


def test_second_launch_is_refused():
    assert launch_problem({}) is None
    assert launch_problem(TAKEOVER) is not None
    assert launch_problem(NEW) is not None


def test_running_follower_mode_is_recorded_like_inference():
    assert normalize_mode("INFERENCE_FOLLOWERS") == "INFERENCE"
    for mode in ("INFERENCE", "HOLD", "HOME", "WAITING_POLICY_RESYNC", "TAKEOVER"):
        assert normalize_mode(mode) == mode
    samples = object.__new__(RosSamples)
    samples.lock = __import__("threading").Lock()
    samples.status = {
        "mode": "",
        "phase": "",
        "active": False,
        "hold": False,
        "home": False,
    }
    samples._status(SimpleNamespace(data="inference_followers"), "mode")
    samples._status(SimpleNamespace(data=True), "active")
    assert samples.recording_allowed()
    samples._status(SimpleNamespace(data="WAITING_POLICY_RESYNC"), "mode")
    assert not samples.recording_allowed()


@pytest.mark.skipif(
    not LINGBOT_PKG.is_dir(), reason="lingbot_piper package not on this host"
)
def test_lingbot_package_is_the_fixed_mux():
    assert package_problem(LINGBOT_PKG) is None


def test_original_cobot_magic_package_is_rejected(tmp_path):
    # Shape of cobot_magic's piper package: no marker, launch uses pkg="piper".
    (tmp_path / "scripts").mkdir()
    (tmp_path / "launch").mkdir()
    (tmp_path / "package.xml").write_text("<package><name>piper</name></package>")
    (tmp_path / "scripts" / "piper_policy_follower_mux.py").write_text("# original\n")
    (tmp_path / "launch" / "start_inference_followers.launch").write_text('pkg="piper"')
    assert "not the lingbot_piper package" in package_problem(tmp_path)


@pytest.mark.skipif(
    not LINGBOT_PKG.is_dir(), reason="lingbot_piper package not on this host"
)
def test_package_without_marker_is_rejected(tmp_path):
    shutil.copytree(
        LINGBOT_PKG, tmp_path / "pkg", ignore=shutil.ignore_patterns("__pycache__")
    )
    mux = tmp_path / "pkg" / "scripts" / "piper_policy_follower_mux.py"
    mux.write_text(mux.read_text().replace(FIXED_MUX_IMPL, "something else"))
    assert "marker" in package_problem(tmp_path / "pkg")
