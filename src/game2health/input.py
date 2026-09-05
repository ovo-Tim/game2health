"""Keyboard backends and the key emitter worker.

Design
------
* ``InputKey`` tokens are platform-independent; only a backend converts them
  to ``pynput`` keys or Linux ``evdev`` key codes.
* ``KeyScheduler`` is a pure, clock-injected timing core: it presses every
  token immediately, releases it 35 ms later, and spaces two same-direction
  repeats 70 ms apart so a two-lane sprint reads as two distinct taps.
  Different actions may overlap (jump while moving sideways).
* ``KeyEmitter`` is the worker thread around the scheduler.  Vision and UI
  threads never sleep: they enqueue action batches; the worker performs the
  real key events.
* Every press is tracked; close / stop / emergency release the whole held
  set so a key can never get stuck.
"""

from __future__ import annotations

import heapq
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

from .types import Action

KEY_OPTIONS: tuple[str, ...] = (
    "left", "right", "up", "down", "esc",
    "space", "w", "a", "s", "d", "p",
)

# Platform-independent key tokens (11 keys, matching the uinput device).
class InputKey(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    ESC = "esc"
    SPACE = "space"
    W = "w"
    A = "a"
    S = "s"
    D = "d"
    P = "p"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


def default_mapping() -> dict[Action, InputKey]:
    return {
        Action.MOVE_LEFT: InputKey.LEFT,
        Action.MOVE_RIGHT: InputKey.RIGHT,
        Action.JUMP: InputKey.UP,
        Action.CROUCH: InputKey.DOWN,
        Action.PAUSE_TOGGLE: InputKey.ESC,
    }


class KeyboardBackend(Protocol):
    def press(self, token: InputKey) -> None: ...
    def release(self, token: InputKey) -> None: ...
    def close(self) -> None: ...


class BackendUnavailable(RuntimeError):
    """Backend could not be created; carry a user-facing reason."""


# --------------------------------------------------------------------------
# pynput backend (macOS, Windows, Linux X11)
# --------------------------------------------------------------------------

_PYNPUT_SPECIAL = {
    InputKey.LEFT: "left",
    InputKey.RIGHT: "right",
    InputKey.UP: "up",
    InputKey.DOWN: "down",
    InputKey.ESC: "esc",
    InputKey.SPACE: "space",
}


class PynputBackend:
    """Posts real key events through pynput's global controller."""

    def __init__(self) -> None:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise BackendUnavailable(
                "Failed to load pynput; install the components required for system keyboard events."
            ) from exc
        self._controller = Controller()
        self._key = Key

    def _to_key(self, token: InputKey):
        special = _PYNPUT_SPECIAL.get(token)
        if special is not None:
            return getattr(self._key, special)
        return token.value  # single character

    def press(self, token: InputKey) -> None:
        self._controller.press(self._to_key(token))

    def release(self, token: InputKey) -> None:
        self._controller.release(self._to_key(token))

    def close(self) -> None:  # pragma: no cover - nothing to tear down
        pass


# --------------------------------------------------------------------------
# Linux uinput backend (native Wayland)
# --------------------------------------------------------------------------

# linux/input-event-codes.h values, stable kernel ABI.
UINPUT_EV_KEY_CODES: dict[InputKey, int] = {
    InputKey.ESC: 1,
    InputKey.P: 25,
    InputKey.S: 31,
    InputKey.D: 32,
    InputKey.W: 17,
    InputKey.A: 30,
    InputKey.SPACE: 57,
    InputKey.UP: 103,
    InputKey.DOWN: 108,
    InputKey.LEFT: 105,
    InputKey.RIGHT: 106,
}

EV_KEY = 0x01
EV_SYN = 0x00
SYN_REPORT = 0


class UInputBackend:
    """Creates 'Game2Health Virtual Keyboard' through evdev.UInput.

    ``factory`` is injectable for tests; by default it opens /dev/uinput.
    """

    def __init__(self, factory: Callable | None = None) -> None:
        codes = UINPUT_EV_KEY_CODES
        if factory is None:
            factory = self._default_factory(codes)
        self._codes = codes
        try:
            self._ui = factory()
        except OSError as exc:
            raise BackendUnavailable(str(exc)) from exc
        self._closed = False

    @staticmethod
    def _default_factory(codes):
        def open_device():
            import evdev
            from evdev import UInput, ecodes

            return UInput(
                {ecodes.EV_KEY: [codes[k] for k in InputKey]},
                name="Game2Health Virtual Keyboard",
            )

        return open_device

    # -- KeyboardBackend ---------------------------------------------------

    def press(self, token: InputKey) -> None:
        self._write(token, 1)

    def release(self, token: InputKey) -> None:
        self._write(token, 0)

    def _write(self, token: InputKey, value: int) -> None:
        self._ui.write(EV_KEY, self._codes[token], value)
        self._ui.syn()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._ui.close()


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

def accessibility_trusted() -> bool:
    """macOS: can this process tree post keystrokes to other apps?

    CGEventPost silently drops events without Accessibility trust, so the
    check must be explicit. Fail open: a missing framework must never
    block backend construction.
    """
    if sys.platform != "darwin":
        return True
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("ApplicationServices")
        if path is None:
            return True
        services = ctypes.CDLL(path)
        services.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(services.AXIsProcessTrusted())
    except Exception:  # pragma: no cover - defensive
        return True


def detect_backend_kind(
    sys_platform: str | None = None,
    environ: dict | None = None,
    uinput_factory: Callable | None = None,
) -> tuple[str | None, str | None]:
    """Decide the backend for this session without touching HIToolbox.

    Safe to call from any thread: it never constructs pynput (whose
    Controller must live on the macOS main thread).  For Wayland it probes
    /dev/uinput by opening and closing a device, so permission problems
    surface here instead of at start-button time.

    Returns (kind, reason): kind is "pynput" or "uinput", or None with a
    user-facing reason.
    """
    import os

    platform = sys.platform if sys_platform is None else sys_platform
    env = environ if environ is not None else dict(os.environ)
    if platform == "darwin" or platform == "win32" or platform.startswith("win"):
        return "pynput", None
    if "linux" in platform or platform.startswith("linux"):
        if env.get("WAYLAND_DISPLAY"):
            # Wayland must use uinput; never silently fall back to pynput
            # (it could only reach XWayland clients).
            try:
                UInputBackend(factory=uinput_factory).close()
            except BackendUnavailable as exc:
                return None, str(exc)
            return "uinput", None
        if env.get("DISPLAY"):
            return "pynput", None
        return None, ("No WAYLAND_DISPLAY or DISPLAY found; cannot create a keyboard device.")
    return None, f"Unsupported platform: {platform}"


# --------------------------------------------------------------------------
# Timing core
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Event:
    at_ms: int
    seq: int
    kind: str  # "press" | "release" | "release_all"
    token: InputKey | None = None

    def __lt__(self, other: "_Event") -> bool:
        return (self.at_ms, self.seq) < (other.at_ms, other.seq)


class KeyScheduler:
    """Pure press/release timing core driven by an injected clock."""

    RELEASE_AFTER_MS = 35
    REPEAT_SPACING_MS = 70

    def __init__(self, backend: KeyboardBackend,
                 clock: Callable[[], int] | None = None) -> None:
        self._backend = backend
        self._clock = clock or (lambda: time.monotonic_ns() // 1_000_000)
        self._heap: list[_Event] = []
        self._seq = 0
        self._held: set[InputKey] = set()

    # -- producers (any thread; only the owning worker calls _run_due) -----

    def emit(self, tokens: Sequence[InputKey]) -> None:
        """Press each token now (35 ms), repeats of one token 70 ms apart."""
        now = self._clock()
        prev: InputKey | None = None
        prev_at = now
        for token in tokens:
            if token is prev:
                prev_at += self.REPEAT_SPACING_MS
            else:
                prev_at = now
            self._schedule(prev_at, "press", token)
            self._schedule(prev_at + self.RELEASE_AFTER_MS, "release", token)
            prev = token

    def test_key(self, token: InputKey, delay_ms: int = 2_000) -> None:
        """Send one full 35 ms tap after `delay_ms` (focus hand-off time)."""
        at = self._clock() + delay_ms
        self._schedule(at, "press", token)
        self._schedule(at + self.RELEASE_AFTER_MS, "release", token)

    def emergency_pause(self, token: InputKey) -> None:
        """Cancel every pending tap, release held keys, then send one full
        35 ms pause key so games see a complete tap even when a session
        dies mid-press.
        """
        self._heap = []
        self._release_all_held()
        now = self._clock()
        self._schedule(now, "press", token)
        self._schedule(now + self.RELEASE_AFTER_MS, "release", token)

    def release_all(self) -> None:
        """Release every currently pressed key immediately."""
        self._release_all_held()

    def close(self) -> None:
        self.release_all()
        self._backend.close()

    @property
    def held(self) -> frozenset[InputKey]:
        return frozenset(self._held)

    @property
    def next_due_ms(self) -> int | None:
        return self._heap[0].at_ms if self._heap else None

    # -- driver (worker thread) -------------------------------------------

    def run_due(self, now_ms: int | None = None) -> None:
        """Execute every event due at or before `now_ms` (clock by default)."""
        now = self._clock() if now_ms is None else now_ms
        while self._heap and self._heap[0].at_ms <= now:
            ev = heapq.heappop(self._heap)
            if ev.kind == "press":
                self._backend.press(ev.token)
                self._held.add(ev.token)
            elif ev.kind == "release":
                self._backend.release(ev.token)
                self._held.discard(ev.token)
            else:  # release_all
                self._release_all_held()

    # -- internals ---------------------------------------------------------

    def _schedule(self, at_ms: int, kind: str,
                  token: InputKey | None = None) -> None:
        heapq.heappush(self._heap, _Event(at_ms, self._seq, kind, token))
        self._seq += 1

    def _release_all_held(self) -> None:
        for token in sorted(self._held, key=lambda t: t.value):
            self._backend.release(token)
        self._held.clear()


class KeyEmitter:
    """Worker thread around a KeyScheduler; maps Actions to keys."""

    def __init__(
        self,
        backend: KeyboardBackend,
        mapping: dict[Action, InputKey] | None = None,
        scheduler: KeyScheduler | None = None,
    ) -> None:
        self._mapping = dict(mapping if mapping is not None else default_mapping())
        self._scheduler = scheduler or KeyScheduler(backend)
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def set_mapping(self, mapping: dict[Action, InputKey]) -> None:
        self._mapping = dict(mapping)  # atomic swap; next batch uses it

    # -- producers ---------------------------------------------------------

    def emit(self, actions: Sequence[Action]) -> None:
        self._q.put(("actions", tuple(actions)))

    def test_key(self, action: Action, delay_ms: int = 2_000) -> None:
        self._q.put(("test", action, delay_ms))

    def test_token(self, token: InputKey, delay_ms: int = 2_000) -> None:
        """Send one 35 ms tap of an arbitrary token (settings page tester)."""
        self._q.put(("test_token", token, delay_ms))

    def emergency_pause(self) -> None:
        """One full pause-key tap, then release everything held."""
        self._q.put(("emergency",))

    def release_all(self) -> None:
        self._q.put(("release_all",))

    def shutdown(self) -> None:
        """Stop the worker, release every held key and close the backend."""
        self._q.put(("shutdown",))
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="game2health-keys", daemon=True
            )
            self._thread.start()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=self._poll_interval())
            except queue.Empty:
                self._drain()
                continue
            if item is None:
                break
            kind = item[0]
            if kind == "shutdown":
                self._scheduler.close()
                break
            if kind == "actions":
                self._scheduler.emit(self._to_tokens(item[1]))
            elif kind == "test":
                self._scheduler.test_key(
                    self._mapping[item[1]], delay_ms=item[2]
                )
            elif kind == "test_token":
                self._scheduler.test_key(item[1], delay_ms=item[2])
            elif kind == "emergency":
                self._scheduler.emergency_pause(
                    self._mapping[Action.PAUSE_TOGGLE]
                )
            elif kind == "release_all":
                self._scheduler.release_all()
            self._drain()

    def _drain(self) -> None:
        self._scheduler.run_due()

    def _poll_interval(self) -> float:
        due = self._scheduler.next_due_ms
        if due is None:
            return 0.25
        remaining = (due - self._scheduler._clock()) / 1000.0
        return max(0.0, remaining)

    def _to_tokens(self, actions) -> list[InputKey]:
        return [self._mapping[a] for a in actions]
