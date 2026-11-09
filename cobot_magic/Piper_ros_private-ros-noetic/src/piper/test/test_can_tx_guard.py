#!/usr/bin/env python3
"""Offline tests for CAN TX stall detection and link-reset detection.

A fake sysfs directory stands in for /sys/class/net/<port>; no CAN socket is
opened.
"""

import pathlib
import sys
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import piper_can_tx_guard as guard_module  # noqa: E402


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _FakeNet:
    """A writable stand-in for /sys/class/net/<port>/."""

    def __init__(self, root, port="can_l_follower"):
        self.dir = pathlib.Path(root) / port
        (self.dir / "statistics").mkdir(parents=True)
        self.set(tx_packets=0, operstate="up", ifindex=16, carrier_changes=1)

    def set(self, **values):
        for name, value in values.items():
            path = (
                self.dir / "statistics" / name
                if name == "tx_packets"
                else self.dir / name
            )
            path.write_text("%s\n" % value)

    def remove(self):
        for path in sorted(self.dir.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        self.dir.rmdir()


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.net = _FakeNet(self._tmp.name)
        self.clock = _Clock()
        prefix = self._tmp.name + "/"
        self._original_read = guard_module._read_sysfs
        guard_module._read_sysfs = lambda path: self._original_read(
            path.replace("/sys/class/net/", prefix)
        )

    def tearDown(self):
        guard_module._read_sysfs = self._original_read
        self._tmp.cleanup()

    def step(self, seconds, poll):
        events = []
        end = self.clock.now + seconds
        while self.clock.now < end:
            self.clock.now += 0.05
            events.append(poll())
        return [event for event in events if event]


class CanTxGuardTest(_Fixture):
    def setUp(self):
        super().setUp()
        self.guard = guard_module.CanTxGuard(
            "can_l_follower", stall_timeout=0.5, clock=self.clock
        )

    def command_for(self, seconds, tx_flows):
        events = []
        end = self.clock.now + seconds
        tx = 0
        while self.clock.now < end:
            self.clock.now += 0.01
            self.guard.note_command_sent()
            if tx_flows:
                tx += 4
                self.net.set(tx_packets=tx)
            event = self.guard.poll()
            if event:
                events.append(event)
        return events

    def test_flowing_frames_are_not_a_stall(self):
        self.assertEqual(self.command_for(3.0, tx_flows=True), [])

    def test_frozen_counter_while_commanding_is_reported_once(self):
        self.command_for(0.2, tx_flows=True)
        self.assertEqual(self.command_for(3.0, tx_flows=False), ["stalled"])
        self.assertTrue(self.guard.stalled)

    def test_frames_flowing_again_report_resumed(self):
        self.command_for(1.0, tx_flows=False)
        self.net.set(tx_packets=999)
        self.assertEqual(self.command_for(0.2, tx_flows=False), ["resumed"])
        self.assertFalse(self.guard.stalled)

    def test_frozen_counter_without_commands_is_expected(self):
        self.assertEqual(self.step(3.0, self.guard.poll), [])

    def test_vanished_netdev_while_commanding_is_a_stall(self):
        self.net.remove()
        self.assertEqual(self.command_for(1.0, tx_flows=False), ["stalled"])


class CanLinkWatchTest(_Fixture):
    def setUp(self):
        super().setUp()
        self.watch = guard_module.CanLinkWatch("can_l_follower", clock=self.clock)

    def test_steady_link_needs_no_reopen(self):
        self.assertEqual(self.step(1.0, self.watch.poll), [])

    def test_down_up_reports_once_when_back_up(self):
        self.net.set(operstate="down", carrier_changes=2)
        self.assertEqual(self.step(0.5, self.watch.poll), [])
        self.net.set(operstate="up", carrier_changes=3)
        self.assertEqual(self.step(0.5, self.watch.poll), [True])

    def test_down_up_between_polls_is_still_seen(self):
        self.net.set(carrier_changes=3)
        self.assertEqual(self.step(0.5, self.watch.poll), [True])

    def test_rebind_with_new_ifindex_reports_once(self):
        self.net.remove()
        self.assertEqual(self.step(0.5, self.watch.poll), [])
        self.net = _FakeNet(self._tmp.name)
        self.net.set(ifindex=21)
        self.assertEqual(self.step(0.5, self.watch.poll), [True])


class ResetCommandTest(unittest.TestCase):
    def test_reset_is_down_then_up_of_the_same_port(self):
        self.assertEqual(
            guard_module.reset_command("can_r_leader"),
            "sudo ip link set can_r_leader down && "
            "sudo ip link set can_r_leader up",
        )


if __name__ == "__main__":
    unittest.main()
