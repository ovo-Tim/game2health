"""Tests for input.py: default mapping, 35 ms release, 70 ms double-tap
spacing, overlap, emergency/stop key release, backend selection matrix and
the Linux uinput protocol (codes, syn() calls, close, no fallback)."""

from __future__ import annotations

import pytest

import game2health.input as inp
from game2health.input import (
    BackendUnavailable,
    EV_KEY,
    InputKey,
    KeyEmitter,
    KeyScheduler,
    PynputBackend,
    UInputBackend,
    default_mapping,
    detect_backend_kind,
)
from game2health.types import Action


class Recorder:
    """Fake KeyboardBackend recording call order."""

    def __init__(self):
        self.events = []
        self.closed = False

    def press(self, token):
        self.events.append(("press", token))

    def release(self, token):
        self.events.append(("release", token))

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self, t0=1_000):
        self.t = t0

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms


def run_until(sched, rec, ms):
    before = len(rec.events)
    sched.run_due(ms)
    return rec.events[before:]


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def test_default_mapping_and_key_set():
    m = default_mapping()
    assert m[Action.MOVE_LEFT] is InputKey.LEFT
    assert m[Action.MOVE_RIGHT] is InputKey.RIGHT
    assert m[Action.JUMP] is InputKey.UP
    assert m[Action.CROUCH] is InputKey.DOWN
    assert m[Action.PAUSE_TOGGLE] is InputKey.ESC
    assert len(set(InputKey)) == 11
    assert sorted(k.value for k in InputKey) == sorted(inp.KEY_OPTIONS)


# ---------------------------------------------------------------------------
# Scheduler timing
# ---------------------------------------------------------------------------


def test_single_tap_presses_immediately_releases_at_35ms():
    rec = Recorder()
    clock = FakeClock()
    sched = KeyScheduler(rec, clock)
    sched.emit([InputKey.LEFT])
    assert run_until(sched, rec, clock.t) == [("press", InputKey.LEFT)]
    assert run_until(sched, rec, clock.t + 34) == []
    assert run_until(sched, rec, clock.t + 35) == [("release", InputKey.LEFT)]
    assert not sched.held


def test_two_same_direction_taps_spaced_70ms():
    rec = Recorder()
    clock = FakeClock()
    sched = KeyScheduler(rec, clock)
    sched.emit([InputKey.RIGHT, InputKey.RIGHT])
    assert run_until(sched, rec, clock.t) == [("press", InputKey.RIGHT)]
    assert run_until(sched, rec, clock.t + 35) == [
        ("release", InputKey.RIGHT)
    ]
    # Second tap must not start before 70 ms.
    assert run_until(sched, rec, clock.t + 69) == []
    assert run_until(sched, rec, clock.t + 70) == [("press", InputKey.RIGHT)]
    assert run_until(sched, rec, clock.t + 105) == [
        ("release", InputKey.RIGHT)
    ]
    assert not sched.held


def test_different_actions_overlap():
    rec = Recorder()
    clock = FakeClock()
    sched = KeyScheduler(rec, clock)
    sched.emit([InputKey.UP, InputKey.LEFT])  # jump while moving
    assert run_until(sched, rec, clock.t) == [
        ("press", InputKey.UP), ("press", InputKey.LEFT),
    ]
    # LEFT tap ends while UP is still down (overlap window).
    assert run_until(sched, rec, clock.t + 35) == [
        ("release", InputKey.UP), ("release", InputKey.LEFT),
    ]


def test_test_key_delayed_tap():
    rec = Recorder()
    clock = FakeClock()
    sched = KeyScheduler(rec, clock)
    sched.test_key(InputKey.SPACE, delay_ms=2_000)
    assert run_until(sched, rec, clock.t + 1_999) == []
    assert run_until(sched, rec, clock.t + 2_000) == [
        ("press", InputKey.SPACE)
    ]
    assert run_until(sched, rec, clock.t + 2_035) == [
        ("release", InputKey.SPACE)
    ]


def test_release_all_and_close_release_every_held_key():
    rec = Recorder()
    clock = FakeClock()
    sched = KeyScheduler(rec, clock)
    sched.emit([InputKey.UP, InputKey.LEFT])
    run_until(sched, rec, clock.t)
    assert sched.held == frozenset({InputKey.UP, InputKey.LEFT})
    sched.release_all()  # synchronous: releases hit the backend immediately
    last_two = {t for kind, t in rec.events[-2:]}
    assert last_two == {InputKey.UP, InputKey.LEFT}
    assert all(kind == "release" for kind, _t in rec.events[-2:])
    assert not sched.held

    sched.emit([InputKey.DOWN])
    run_until(sched, rec, clock.t + 1)
    sched.close()
    assert ("release", InputKey.DOWN) in rec.events
    assert rec.closed


def test_emergency_pause_sends_full_pause_tap_and_drops_held():
    rec = Recorder()
    clock = FakeClock()
    sched = KeyScheduler(rec, clock)
    sched.emit([InputKey.UP])
    run_until(sched, rec, clock.t)
    sched.emergency_pause(InputKey.ESC)
    # Immediate: UP released synchronously, ESC press queued at now.
    assert ("release", InputKey.UP) in rec.events
    run_until(sched, rec, clock.t)
    assert ("press", InputKey.ESC) in rec.events
    assert sched.held == frozenset({InputKey.ESC})
    # Pause tap is a full 35 ms, not cut short.
    assert run_until(sched, rec, clock.t + 34) == []
    assert run_until(sched, rec, clock.t + 35) == [("release", InputKey.ESC)]
    assert not sched.held


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_detect_backend_kind_matrix():
    # darwin / win32 / Linux X11 -> pynput, nothing constructed here.
    assert detect_backend_kind(sys_platform="darwin", environ={}) == (
        "pynput", None)
    assert detect_backend_kind(sys_platform="win32", environ={}) == (
        "pynput", None)
    assert detect_backend_kind(
        sys_platform="linux", environ={"DISPLAY": ":0"}) == ("pynput", None)

    # Wayland -> uinput when /dev/uinput opens; probe device is closed again.
    events = []

    class FakeUI:
        def __init__(self):
            events.append("open")

        def write(self, *a):
            pass

        def syn(self):
            pass

        def close(self):
            events.append("close")

    assert detect_backend_kind(
        sys_platform="linux",
        environ={"WAYLAND_DISPLAY": "wayland-0"},
        uinput_factory=FakeUI,
    ) == ("uinput", None)
    assert events == ["open", "close"]

    # Wayland with /dev/uinput failing -> refused, never fall back.
    def broken():
        raise OSError("permission denied")

    assert detect_backend_kind(
        sys_platform="linux",
        environ={"WAYLAND_DISPLAY": "wayland-0"},
        uinput_factory=broken,
    ) == (None, "permission denied")

    # Headless Linux with no display session -> unavailable.
    kind, reason = detect_backend_kind(sys_platform="linux", environ={})
    assert kind is None and reason


def test_pynput_construction_is_main_thread_only_docs():
    # Guard against reintroducing select_backend (which constructed pynput
    # off-main and crashed via HIToolbox main-queue assertions).
    assert not hasattr(inp, "select_backend")


# ---------------------------------------------------------------------------
# UInput protocol
# ---------------------------------------------------------------------------


class FakeUI:
    def __init__(self, events_arg, name):
        self.declared = events_arg
        self.name = name
        self.writes = []
        self.syns = 0
        self.closed = False

    def write(self, etype, code, value):
        self.writes.append((etype, code, value))

    def syn(self):
        self.syns += 1

    def close(self):
        self.closed = True


def _uinput_backend():
    device = {}

    def factory():
        ui = FakeUI({0x01: [c for c in inp.UINPUT_EV_KEY_CODES.values()]},
                    "Game2Health Virtual Keyboard")
        device["ui"] = ui
        return ui

    return UInputBackend(factory=factory), device


def test_uinput_declares_exactly_eleven_keys():
    be, dev = _uinput_backend()
    ui = dev["ui"]
    codes = set(ui.declared[0x01])
    assert codes == set(inp.UINPUT_EV_KEY_CODES.values())
    assert len(codes) == 11
    assert ui.name == "Game2Health Virtual Keyboard"


def test_uinput_press_release_values_and_syn_counts():
    be, dev = _uinput_backend()
    ui = dev["ui"]
    be.press(InputKey.W)
    be.release(InputKey.W)
    assert ui.writes == [
        (EV_KEY, 17, 1),  # KEY_W
        (EV_KEY, 17, 0),
    ]
    assert ui.syns == 2  # one syn() per value write


def test_uinput_codes_match_linux_keyboard_layout():
    expected = {
        InputKey.ESC: 1, InputKey.P: 25, InputKey.S: 31, InputKey.D: 32,
        InputKey.W: 17, InputKey.A: 30, InputKey.SPACE: 57,
        InputKey.UP: 103, InputKey.DOWN: 108, InputKey.LEFT: 105,
        InputKey.RIGHT: 106,
    }
    assert inp.UINPUT_EV_KEY_CODES == expected


def test_uinput_close_destroys_device_and_rejects_open_failure():
    be, dev = _uinput_backend()
    be.close()
    assert dev["ui"].closed

    def broken():
        raise OSError("no such file")

    with pytest.raises(BackendUnavailable):
        UInputBackend(factory=broken)


def test_scheduler_through_uinput_backend():
    rec = _RecordingUI()
    device = {}

    def factory():
        device["ui"] = rec
        return rec

    be = UInputBackend(factory=factory)
    # The scheduler drives press/release through the real backend protocol.
    sched = KeyScheduler(be, FakeClock())
    sched.emit([InputKey.LEFT])
    sched.run_due(1_000)
    sched.run_due(1_035)
    writes = device["ui"].writes
    assert writes == [(EV_KEY, 105, 1), (EV_KEY, 105, 0)]
    assert device["ui"].syns == 2


class _RecordingUI(FakeUI):
    def __init__(self):
        super().__init__({}, "")
        self.writes = []
        self.syns = 0


# ---------------------------------------------------------------------------
# KeyEmitter worker (threaded, default mapping)
# ---------------------------------------------------------------------------


def test_emitter_thread_applies_default_mapping_and_shutdown_releases():
    import time as _time

    rec = Recorder()
    sched = KeyScheduler(rec, clock=lambda: int(_time.monotonic() * 1000))
    em = KeyEmitter(rec, scheduler=sched)
    em.start()
    em.emit([Action.MOVE_LEFT])
    _time.sleep(0.05)
    assert ("press", InputKey.LEFT) in rec.events
    _time.sleep(0.10)
    assert ("release", InputKey.LEFT) in rec.events

    em.emit([Action.JUMP, Action.MOVE_RIGHT])
    _time.sleep(0.05)
    assert ("press", InputKey.UP) in rec.events
    assert ("press", InputKey.RIGHT) in rec.events
    em.shutdown()
    assert rec.closed
    # Nothing held after shutdown.
    assert not sched.held
