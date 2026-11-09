#!/usr/bin/env python3
"""Notice a Piper CAN adapter that stopped transmitting, and its reset.

The candleLight adapters (gs_usb, kernel 5.15) sometimes return the echo of a
transmitted frame with its first bytes missing (dmesg: "Unexpected out of
range echo id 0x01550000" is the echo of command frame 0x155 shifted by two
bytes; "Unexpected unused echo id" is its aftermath). Every lost echo leaks
one of gs_usb's 10 TX slots. Once all of them leak the netdev stays UP but
transmits nothing: the arm holds its last target and piper_sdk reports
SEND_MESSAGE_FAILED (100017) as soon as the 1000-frame qdisc is full.

Closing the netdev frees every TX slot and drops the queued (stale)
commands, while bitrate, restart-ms and txqueuelen survive, so
``sudo ip link set <port> down && sudo ip link set <port> up`` recovers the
adapter. piper_sdk's read thread exits on the ENETDOWN that causes (and a
``can_config.sh`` rebind replaces the netdev), so the drivers reopen their
SDK port when ``CanLinkWatch`` sees the link come back.
"""

import time


def reset_command(can_port):
    return "sudo ip link set {0} down && sudo ip link set {0} up".format(can_port)


def _read_sysfs(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return None  # The netdev itself vanished (USB disconnect, rebind).


class CanTxGuard:
    """Report commands being sent while tx_packets stays frozen.

    ``poll`` returns ``"stalled"`` once when the counter stayed frozen for
    ``stall_timeout`` while commands were sent, ``"resumed"`` when frames
    flow again, and ``None`` otherwise.
    """

    CHECK_PERIOD = 0.1
    # A command older than this means nothing is being commanded (disabled,
    # HOLD without commands), so a frozen counter is expected.
    COMMAND_ACTIVE = 0.2

    def __init__(self, can_port, stall_timeout=0.5, clock=time.monotonic):
        self.can_port = can_port
        self.stall_timeout = stall_timeout
        self._clock = clock
        self._counter_path = "/sys/class/net/%s/statistics/tx_packets" % can_port
        self._last_command_sent = 0.0
        self._last_value = None
        self._last_change = clock()
        self._next_check = 0.0
        self.stalled = False
        self.last_value = None

    def note_command_sent(self):
        self._last_command_sent = self._clock()

    def stalled_for(self):
        return self._clock() - self._last_change

    def poll(self):
        now = self._clock()
        if now < self._next_check:
            return None
        self._next_check = now + self.CHECK_PERIOD
        raw = _read_sysfs(self._counter_path)
        value = int(raw) if raw is not None and raw.isdigit() else None
        self.last_value = value
        if value is not None and value != self._last_value:
            self._last_value = value
            self._last_change = now
            if self.stalled:
                self.stalled = False
                return "resumed"
            return None
        if now - self._last_command_sent > self.COMMAND_ACTIVE:
            self._last_change = now
            return None
        if self.stalled or now - self._last_change < self.stall_timeout:
            return None
        self.stalled = True
        return "stalled"


class CanLinkWatch:
    """Say once when the netdev is UP again after a down/up or re-creation.

    ``ip link set down/up`` bumps ``carrier_changes``; a ``can_config.sh``
    rebind gives the same name a new ``ifindex``. Either one leaves
    piper_sdk without a working read thread.
    """

    CHECK_PERIOD = 0.1

    def __init__(self, can_port, clock=time.monotonic):
        self.can_port = can_port
        self._clock = clock
        self._base = "/sys/class/net/%s/" % can_port
        self._next_check = 0.0
        self._generation = self._read_generation()

    def _read_generation(self):
        if _read_sysfs(self._base + "operstate") != "up":
            return None
        return (
            _read_sysfs(self._base + "ifindex"),
            _read_sysfs(self._base + "carrier_changes"),
        )

    def poll(self):
        now = self._clock()
        if now < self._next_check:
            return False
        self._next_check = now + self.CHECK_PERIOD
        generation = self._read_generation()
        if generation is None or generation == self._generation:
            return False
        self._generation = generation
        return True
