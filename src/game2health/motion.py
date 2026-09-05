"""Calibration, pose motion detection and the controller state machine.

This module is the single source of truth for every anti-cheat threshold,
debounce window, latch and pause/resume semantic.  It is pure logic: no Qt,
no I/O, no wall clock.  Time flows exclusively through sample timestamps
(`PoseSample.timestamp_ms`) and explicit `tick(now_ms)` calls, which makes
every behavior deterministically testable with synthetic samples.

Coordinate conventions
----------------------
* Lane membership uses the **raw** MediaPipe normalized x coordinate.
* Angles and distances first scale x by ``frame_width / frame_height`` so
  that horizontal and vertical units both live in "normalized image height"
  space; a 16:9 frame does not distort knee angles.
* Every displacement threshold is a multiple of the calibrated
  ``torso_length`` (measured in normalized-image-height units), so camera
  distance does not change any verdict.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .types import (
    Action,
    CORE_POINTS,
    ControllerMode,
    Lane,
    MotionSnapshot,
    PoseSample,
)

# --------------------------------------------------------------------------
# Optional debug tracing (enabled via enable_debug(); see app: G2H_DEBUG=1)
# --------------------------------------------------------------------------

_debug_sink = None


def enable_debug() -> None:
    """Print detector internals to stderr for live diagnosis."""
    import sys

    global _debug_sink

    def _sink(msg: str) -> None:
        print(f"[g2h] {msg}", file=sys.stderr, flush=True)

    _debug_sink = _sink


def _trace(msg: str) -> None:
    if _debug_sink is not None:
        _debug_sink(msg)


_POINT_NAMES = {
    11: "L_shoulder", 12: "R_shoulder", 23: "L_hip", 24: "R_hip",
    25: "L_knee", 26: "R_knee", 27: "L_ankle", 28: "R_ankle",
}


# --------------------------------------------------------------------------
# Constants (thresholds / debounce / latches)
# --------------------------------------------------------------------------

VISIBILITY_MIN = 0.6

GAP_TOLERANCE_MS = 250  # longest inter-sample gap still considered continuous

CALIB_FULL_BODY_MS = 1_500  # continuous full-body before the standing phase
CALIB_STAND_MS = 2_000  # continuous upright window for baseline capture
CALIB_MIN_SAMPLES = 30

SMOOTH_ALPHA = 0.65  # EWMA weight of the newest sample
SMOOTH_INDICES = (23, 24, 25, 26, 27, 28)  # hips, knees, ankles

THIRD_1 = 1.0 / 3.0
THIRD_2 = 2.0 / 3.0
LANE_MARGIN = 0.015
LANE_LEFT_EDGE = THIRD_1 + LANE_MARGIN
LANE_RIGHT_EDGE = THIRD_2 - LANE_MARGIN

RUN_LIFT = 0.12  # ankle OR knee lift entering "leg lifted" (x torso_length)
RUN_REARM = 0.06  # both must fall back below this before re-lifting
RUN_STEP_MIN_MS = 180
RUN_STEP_MAX_MS = 2_200  # tolerate one missed lift observation
RUN_ENTRY_STEPS = 3
RUN_ENTRY_WINDOW_MS = 2 * RUN_STEP_MAX_MS + 600  # two max gaps plus jitter
RUN_STOP_AFTER_MS = RUN_STEP_MAX_MS + 800  # never expire before a valid gap
RUN_HOVER_REARM_MS = 600  # must be shorter than stop deadline
POSE_LOST_STOP_MS = 1_200
ACTION_GRACE_MS = 800  # locked jump/crouch may extend step grace by <= this

JUMP_LIFT = 0.18  # both ankles above the higher jump line
JUMP_REARM_ANKLE = 0.05
CROUCH_DOWN = 0.12  # hip sink below baseline (x torso_length)
CROUCH_KNEE_MAX_DEG = 140.0
CROUCH_REARM_HIP = 0.10
CROUCH_REARM_KNEE_MIN_DEG = 135.0

ARM_FRAMES = 2  # consecutive qualifying frames before jump/crouch fires

RESUME_GRACE_MS = 300  # RESUMING -> ACTIVE delay after the resume key

# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


class CalibrationPhase(Enum):
    FULL_BODY = "full_body"
    UPRIGHT = "upright"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Median standing baseline, in normalized image-y units."""

    hip_y_left: float
    hip_y_right: float
    knee_y_left: float
    knee_y_right: float
    ankle_y_left: float
    ankle_y_right: float
    hip_mid_y: float
    torso_length: float
    upright_knee_angle_deg: float

    @property
    def valid(self) -> bool:
        return self.torso_length > 0.0


@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    phase: CalibrationPhase
    missing_indices: tuple[int, ...] = ()
    full_body_ms: int = 0
    upright_ms: int = 0
    samples: int = 0
    result: CalibrationResult | None = None


def full_body_ok(sample: PoseSample) -> tuple[bool, tuple[int, ...]]:
    """Every calibration point detected with sufficient visibility.

    Returns (ok, missing_indices). Points merely near the frame edge are
    accepted: the standing pose is usable without strict framing, and the
    lane geometry tolerates off-center subjects.
    """
    from .types import FULL_BODY_POINTS  # keep this module's import graph thin

    missing: list[int] = []
    for idx in FULL_BODY_POINTS:
        lm = sample.landmarks[idx]
        if lm.visibility < VISIBILITY_MIN:
            missing.append(idx)
    return not missing, tuple(missing)


class CalibrationTracker:
    """Two-stage calibration: full body 1.5 s, then upright 2.0 s median."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase = CalibrationPhase.FULL_BODY
        self._last_valid_ts: int | None = None
        self._window_since = 0
        self._upright_since = 0
        self._rows: list[tuple[float, ...]] = []
        self._result: CalibrationResult | None = None

    @property
    def result(self) -> CalibrationResult | None:
        return self._result

    def update(self, sample: PoseSample | None) -> CalibrationStatus:
        if sample is None:
            self.reset()
            return self._status()
        ts = sample.timestamp_ms
        ok, missing = full_body_ok(sample)
        if not ok:
            # Any missing point resets the continuous-valid timers.
            self.reset()
            return self._status(missing=missing)
        continuous = self._last_valid_ts is not None and (
            ts - self._last_valid_ts <= GAP_TOLERANCE_MS
        )
        self._last_valid_ts = ts

        if self._phase is CalibrationPhase.FULL_BODY:
            if not continuous:
                self._window_since = ts
            if ts - self._window_since >= CALIB_FULL_BODY_MS:
                self._phase = CalibrationPhase.UPRIGHT
                self._upright_since = ts
                self._rows = []
            return self._status(full_body_ms=ts - self._window_since)

        if self._phase is CalibrationPhase.UPRIGHT:
            if not continuous:
                self._upright_since = ts
                self._rows = []
            self._rows.append(self._row(sample))
            elapsed = ts - self._upright_since
            if elapsed >= CALIB_STAND_MS:
                if not self._finalize():
                    # Too few samples or a broken torso estimate: retry.
                    self._phase = CalibrationPhase.FULL_BODY
                    self._window_since = ts
                    self._rows = []
            return self._status(upright_ms=elapsed, samples=len(self._rows))

        return self._status()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _row(sample: PoseSample) -> tuple[float, ...]:
        lm = sample.landmarks
        hip_mid_y = (lm[23].y + lm[24].y) / 2.0
        sh_mid_y = (lm[11].y + lm[12].y) / 2.0
        angles: list[float] = []
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            angles.append(_knee_angle_raw(lm[hip], lm[knee], lm[ankle], sample))
        return (
            lm[23].y, lm[24].y,
            lm[25].y, lm[26].y,
            lm[27].y, lm[28].y,
            hip_mid_y,
            hip_mid_y - sh_mid_y,
            (angles[0] + angles[1]) / 2.0,
        )

    def _finalize(self) -> bool:
        rows = self._rows
        if len(rows) < CALIB_MIN_SAMPLES:
            return False
        cols = list(zip(*rows))
        hip_l = _median(cols[0])
        hip_r = _median(cols[1])
        knee_l = _median(cols[2])
        knee_r = _median(cols[3])
        ankle_l = _median(cols[4])
        ankle_r = _median(cols[5])
        hip_mid_y = _median(cols[6])
        torso = _median(cols[7])
        angle = _median(cols[8])
        if torso <= 0.0:
            return False
        self._result = CalibrationResult(
            hip_y_left=hip_l,
            hip_y_right=hip_r,
            knee_y_left=knee_l,
            knee_y_right=knee_r,
            ankle_y_left=ankle_l,
            ankle_y_right=ankle_r,
            hip_mid_y=hip_mid_y,
            torso_length=torso,
            upright_knee_angle_deg=angle,
        )
        self._phase = CalibrationPhase.COMPLETE
        return True

    def _status(
        self,
        missing: tuple[int, ...] = (),
        full_body_ms: int = 0,
        upright_ms: int = 0,
        samples: int = 0,
    ) -> CalibrationStatus:
        return CalibrationStatus(
            phase=self._phase,
            missing_indices=missing,
            full_body_ms=full_body_ms,
            upright_ms=upright_ms,
            samples=samples,
            result=self._result,
        )


# --------------------------------------------------------------------------
# Shared geometry helpers
# --------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _knee_angle_raw(
    hip, knee, ankle, sample: PoseSample,
) -> float:
    """Interior angle at the knee in degrees from raw landmarks."""
    aspect = sample.frame_width / sample.frame_height
    return _knee_angle_xy(
        (hip.x * aspect, hip.y),
        (knee.x * aspect, knee.y),
        (ankle.x * aspect, ankle.y),
    )


def _knee_angle_xy(hip, knee, ankle) -> float:
    """Interior angle at the knee in degrees from aspect-scaled (x, y)."""
    ax = hip[0] - knee[0]
    ay = hip[1] - knee[1]
    bx = ankle[0] - knee[0]
    by = ankle[1] - knee[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    cos = (ax * bx + ay * by) / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _region_of(raw_x: float) -> Lane:
    """Classify raw head x; side zones absorb the old boundary dead-zones."""
    if raw_x <= LANE_LEFT_EDGE:
        return Lane.LEFT
    if raw_x >= LANE_RIGHT_EDGE:
        return Lane.RIGHT
    return Lane.CENTER




def _core_usable(sample: PoseSample) -> bool:
    from .types import CORE_POINTS

    if len(sample.landmarks) < 33:
        return False
    return all(
        sample.landmarks[i].visibility >= VISIBILITY_MIN for i in CORE_POINTS
    )


# --------------------------------------------------------------------------
# Motion detector
# --------------------------------------------------------------------------


class MotionDetector:
    """Turns pose samples into lane commits, jump/crouch latches, running.

    Instantiate with a baseline (``set_baseline``) before use.  All time
    advances through ``update()`` / ``tick()``; every call returns a
    ``MotionSnapshot`` describing that instant.
    """

    def __init__(self) -> None:
        self._cal: CalibrationResult | None = None
        self.clear_transients()

    # -- lifecycle ---------------------------------------------------------

    def set_baseline(self, cal: CalibrationResult) -> None:
        self._cal = cal
        self.clear_transients()

    def clear_transients(self) -> None:
        """Drop latches, smoothing, steps and lanes; keep the baseline."""
        self._last_ts = -1
        self._last_usable_ms = -1
        self._dbg_ts: dict[str, int] = {}
        self._smooth: dict[int, tuple[float, float]] = {}
        self._stable_lane = Lane.CENTER
        self._raw_lane = Lane.CENTER
        self._jump_locked = False
        self._jump_frames = 0
        self._crouch_armed = 0
        self._crouch_locked = False
        self._leg_up: dict[int, bool] = {25: False, 26: False}
        self._leg_up_since: dict[int, int] = {25: -1, 26: -1}
        self._steps: deque[int] = deque()
        self._deadline_ms = -1
        self._running = False
        self._step_count = 0
        self._last_step_side: int | None = None
        self._last_lift_ms = -1
        self._has_run = False
        self._stopped_at_step_count = 0

    def _trace_rate(self, key: str, msg: str, min_ms: int = 500) -> None:
        now = time.monotonic_ns() // 1_000_000
        if now - self._dbg_ts.get(key, -10**9) < min_ms:
            return
        self._dbg_ts[key] = now
        _trace(msg)

    @property
    def calibrated(self) -> bool:
        return self._cal is not None

    @property
    def raw_lane(self) -> Lane:
        """Latest visible head region for UI; never unknown."""
        return self._raw_lane

    # -- public time API ---------------------------------------------------

    def update(self, sample: PoseSample) -> MotionSnapshot:
        return self._process(sample.timestamp_ms, sample)

    def tick(self, now_ms: int) -> MotionSnapshot:
        return self._process(now_ms, None)

    # -- internals ---------------------------------------------------------

    def _process(
        self, ts: int, sample: PoseSample | None,
    ) -> MotionSnapshot:
        if ts <= self._last_ts:
            return self._snapshot(ts)
        self._last_ts = ts

        events: list[Action] = []
        usable = False
        head_region = None
        if sample is not None and sample.landmarks:
            nose = sample.landmarks[0]
            if nose.visibility >= VISIBILITY_MIN:
                head_region = _region_of(nose.x)
                self._raw_lane = head_region

        if self._cal is None:
            pass
        elif sample is not None:
            if head_region is not None:
                self._update_lane(head_region, events)

            usable = _core_usable(sample)
            if usable:
                self._last_usable_ms = ts
                self._smooth_sample(sample)
                self._update_latches(sample, ts, events)
            else:
                failed = [
                    f"{_POINT_NAMES[i]}={sample.landmarks[i].visibility:.2f}"
                    for i in CORE_POINTS
                    if sample.landmarks[i].visibility < VISIBILITY_MIN
                ]
                self._trace_rate(
                    "unusable",
                    "frame dropped, low visibility: " + " ".join(failed))
        self._evaluate_running(ts)
        if sample is not None:
            pose_valid = usable
        else:
            pose_valid = self._pose_recent(ts)
        return MotionSnapshot(
            timestamp_ms=ts,
            pose_valid=pose_valid,
            stable_lane=self._stable_lane,
            running=self._running,
            events=tuple(events),
            steps=self._step_count,
        )

    def _pose_recent(self, now_ms: int) -> bool:
        if self._last_usable_ms < 0:
            return False
        return now_ms - self._last_usable_ms <= POSE_LOST_STOP_MS

    def _snapshot(self, ts: int) -> MotionSnapshot:
        return MotionSnapshot(
            timestamp_ms=ts,
            pose_valid=self._pose_recent(ts),
            stable_lane=self._stable_lane,
            running=self._running,
            steps=self._step_count,
        )

    # -- sample processing -------------------------------------------------

    def _smooth_sample(self, sample: PoseSample) -> None:
        lms = sample.landmarks
        for idx in SMOOTH_INDICES:
            lm = lms[idx]
            prev = self._smooth.get(idx)
            if prev is None:
                self._smooth[idx] = (lm.x, lm.y)
            else:
                self._smooth[idx] = (
                    prev[0] + SMOOTH_ALPHA * (lm.x - prev[0]),
                    prev[1] + SMOOTH_ALPHA * (lm.y - prev[1]),
                )

    def _update_lane(
        self, region: Lane, events: list[Action],
    ) -> None:
        if self._stable_lane is region:
            return
        old = self._stable_lane
        self._stable_lane = region
        _trace(f"lane changed: {old.name.lower()} -> {region.name.lower()}")
        diff = int(region) - int(old)
        if diff > 0:
            events.extend([Action.MOVE_RIGHT] * diff)
        else:
            events.extend([Action.MOVE_LEFT] * (-diff))

    def _update_latches(
        self, sample: PoseSample, ts: int, events: list[Action],
    ) -> None:
        cal = self._cal
        assert cal is not None
        t = cal.torso_length
        if t <= 0.0:
            return
        aspect = sample.frame_width / sample.frame_height
        sm = self._smooth

        hip_mid_y = (sm[23][1] + sm[24][1]) / 2.0
        ankle_l_y = sm[27][1]
        ankle_r_y = sm[28][1]
        knee_l_y = sm[25][1]
        knee_r_y = sm[26][1]

        # --- jumping ------------------------------------------------------
        lift_ankle_l = cal.ankle_y_left - ankle_l_y
        lift_ankle_r = cal.ankle_y_right - ankle_r_y
        candidate = (
            lift_ankle_l >= JUMP_LIFT * t
            and lift_ankle_r >= JUMP_LIFT * t
        )
        if self._jump_locked:
            ankle_l_home = (
                abs(ankle_l_y - cal.ankle_y_left) <= JUMP_REARM_ANKLE * t
            )
            ankle_r_home = (
                abs(ankle_r_y - cal.ankle_y_right) <= JUMP_REARM_ANKLE * t
            )
            if ankle_l_home and ankle_r_home:
                self._jump_locked = False
        elif candidate:
            self._jump_frames += 1
            if self._jump_frames >= ARM_FRAMES:
                self._jump_locked = True
                self._jump_frames = 0
                _trace("action: jump")
                events.append(Action.JUMP)
                self._extend_grace(ts)
        else:
            self._jump_frames = 0

        # --- crouching ----------------------------------------------------
        hip_down = hip_mid_y - cal.hip_mid_y
        knee_l_deg = _knee_angle_xy(
            _scaled_pt(sm[23], aspect), _scaled_pt(sm[25], aspect),
            _scaled_pt(sm[27], aspect),
        )
        knee_r_deg = _knee_angle_xy(
            _scaled_pt(sm[24], aspect), _scaled_pt(sm[26], aspect),
            _scaled_pt(sm[28], aspect),
        )
        c_cond = (
            hip_down >= CROUCH_DOWN * t
            and knee_l_deg < CROUCH_KNEE_MAX_DEG
            and knee_r_deg < CROUCH_KNEE_MAX_DEG
        )
        if self._crouch_locked:
            home = (
                abs(hip_mid_y - cal.hip_mid_y) <= CROUCH_REARM_HIP * t
                and knee_l_deg > CROUCH_REARM_KNEE_MIN_DEG
                and knee_r_deg > CROUCH_REARM_KNEE_MIN_DEG
            )
            if home:
                self._crouch_locked = False
                self._crouch_armed = 0
        else:
            self._crouch_armed = self._crouch_armed + 1 if c_cond else 0
            if self._crouch_armed >= ARM_FRAMES:
                self._crouch_locked = True
                self._crouch_armed = 0
                events.append(Action.CROUCH)
                _trace("action: crouch")
                self._extend_grace(ts)

        # --- running steps -------------------------------------------------
        # Decide after updating BOTH legs. A simultaneous first crossing is
        # a two-foot takeoff, not two running steps. Do not require the
        # opposite leg to be fully below threshold: smoothing commonly keeps
        # it slightly elevated during ordinary alternating jogging.
        entered: list[int] = []
        if self._update_leg(
            25, knee_l_y, cal.knee_y_left, ankle_l_y, cal.ankle_y_left, ts,
        ):
            entered.append(25)
        if self._update_leg(
            26, knee_r_y, cal.knee_y_right, ankle_r_y, cal.ankle_y_right, ts,
        ):
            entered.append(26)
        if len(entered) == 1:
            self._last_lift_ms = ts
            if self._running:
                # Once activated, a real lift is liveness even if cadence or
                # alternation rejects it as a new step. Missed frames must not
                # turn active jogging into an automatic pause.
                self._deadline_ms = max(
                    self._deadline_ms, ts + RUN_STOP_AFTER_MS)
            self._consider_step(entered[0], ts)
        elif len(entered) == 2:
            self._trace_rate(
                "reject-two-foot",
                "step rejected: both legs crossed together (jump, not run)")

    def _update_leg(
        self,
        knee_idx: int,
        knee_y: float,
        base_knee: float,
        ankle_y: float,
        base_ankle: float,
        ts: int,
    ) -> bool:
        cal = self._cal
        assert cal is not None
        t = cal.torso_length
        lift_knee = base_knee - knee_y
        lift_ankle = base_ankle - ankle_y
        thr = RUN_LIFT * t
        lifted = lift_knee >= thr or lift_ankle >= thr
        side = "L" if knee_idx == 25 else "R"
        up = self._leg_up[knee_idx]
        if lifted and not up:
            self._leg_up[knee_idx] = True
            self._leg_up_since[knee_idx] = ts
            _trace(f"lift {side} knee={lift_knee / thr:.0%} "
                   f"ankle={lift_ankle / thr:.0%} of threshold")
            return True
        if not lifted and up:
            # If posture drift keeps a leg between REARM and LIFT, it would
            # stay "up" forever and no subsequent lift could register.
            # Re-arm after a short below-LIFT dwell, before RUNNING expires.
            below_rearm = lift_knee < RUN_REARM * t and lift_ankle < RUN_REARM * t
            hover_expired = (
                ts - self._leg_up_since[knee_idx] >= RUN_HOVER_REARM_MS
            )
            if below_rearm or hover_expired:
                self._leg_up[knee_idx] = False
        return False

    def _consider_step(self, knee_idx: int, ts: int) -> None:
        while self._steps and ts - self._steps[0] > RUN_ENTRY_WINDOW_MS:
            self._steps.popleft()
        side = "L" if knee_idx == 25 else "R"
        if self._steps:
            dt = ts - self._steps[-1]
            if dt < RUN_STEP_MIN_MS:
                self._trace_rate(
                    "reject-dt",
                    f"step {side} rejected: dt={dt}ms "
                    f"(minimum {RUN_STEP_MIN_MS}ms)")
                return
            if dt > RUN_STEP_MAX_MS:
                # The previous cadence chain is over. The current lift is
                # the first step of a new chain; rejecting it would create a
                # dead zone until RUN_ENTRY_WINDOW_MS eventually elapsed.
                self._steps.clear()
                _trace(f"step sequence restarted after {dt}ms gap")
            elif self._last_step_side == knee_idx:
                self._trace_rate(
                    "reject-side",
                    f"step {side} rejected: same side as previous")
                return
        self._steps.append(ts)
        self._last_step_side = knee_idx
        self._deadline_ms = max(self._deadline_ms, ts + RUN_STOP_AFTER_MS)
        self._step_count += 1
        _trace(f"step #{self._step_count} {side} (running entry needs "
               f"{RUN_ENTRY_STEPS} within {RUN_ENTRY_WINDOW_MS}ms)")

    def _extend_grace(self, ts: int) -> None:
        """A jump/crouch fire extends the step-stop deadline by <= 0.8 s."""
        if not self._steps:
            return
        self._deadline_ms = max(
            self._deadline_ms,
            self._steps[-1] + RUN_STOP_AFTER_MS + ACTION_GRACE_MS,
        )

    def _evaluate_running(self, now: int) -> None:
        if self._cal is None:
            self._running = False
            return

        prev = self._running
        if now - self._last_usable_ms > POSE_LOST_STOP_MS:
            if prev:
                _trace(f"RUNNING stop: pose lost "
                       f"({now - self._last_usable_ms}ms without usable frame)")
                self._stopped_at_step_count = self._step_count
            self._running = False
            return

        recent = [t for t in self._steps if now - t <= RUN_ENTRY_WINDOW_MS]
        first_entry = (
            not self._has_run and len(recent) >= RUN_ENTRY_STEPS
        )
        steps_since_stop = self._step_count - self._stopped_at_step_count
        restart_entry = (
            self._has_run
            and not prev
            and steps_since_stop >= RUN_ENTRY_STEPS
        )
        if prev:
            running = now <= self._deadline_ms
        else:
            running = first_entry or restart_entry

        if running and self._steps:
            self._deadline_ms = max(
                self._deadline_ms, self._steps[-1] + RUN_STOP_AFTER_MS
            )
        if running and not prev:
            if self._has_run:
                _trace(f"RUNNING restart ({steps_since_stop} new steps "
                       "since stop)")
            else:
                _trace(f"RUNNING start ({len(recent)} steps in "
                       f"{RUN_ENTRY_WINDOW_MS}ms window)")
            self._has_run = True
        elif prev and not running:
            self._stopped_at_step_count = self._step_count
            since = now - self._last_lift_ms if self._last_lift_ms >= 0 else -1
            _trace(f"RUNNING stop: no leg lift for {since}ms "
                   f"(deadline {RUN_STOP_AFTER_MS}ms after activity)")
        self._running = running


def _scaled_pt(xy: tuple[float, float], aspect: float) -> tuple[float, float]:
    return (xy[0] * aspect, xy[1])


# --------------------------------------------------------------------------
# Controller state machine
# --------------------------------------------------------------------------


class ControllerStateMachine:
    """Tracks the mode the UI/keyboard are in.

    Maintains ``game_lane`` — the lane the target game character occupies —
    from direction events forwarded while ACTIVE.  Separately remembers the
    user's physical lane when auto-pause begins; resume must happen in that
    physical lane because it can legitimately differ from ``game_lane``.
    """

    def __init__(self) -> None:
        self._mode = ControllerMode.CALIBRATING
        self._game_lane = Lane.CENTER
        self._resuming_since = -1
        self._paused_steps = 0
        self._resume_wait_signature: tuple[bool, Lane | None, int] | None = None
        self._active_grace_until = -1

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Back to calibration; forget lane and pause memory."""
        self._mode = ControllerMode.CALIBRATING
        self._game_lane = Lane.CENTER
        self._resuming_since = -1
        self._paused_steps = 0
        self._resume_wait_signature = None
        self._active_grace_until = -1

    def arm(self) -> None:
        """After the start countdown: wait for the user to start running."""
        if self._mode is ControllerMode.CALIBRATING:
            self._mode = ControllerMode.WAITING_FOR_RUN
            self._game_lane = Lane.CENTER
            self._resuming_since = -1
            self._paused_steps = 0
            self._resume_wait_signature = None
            self._active_grace_until = -1

    @property
    def mode(self) -> ControllerMode:
        return self._mode

    @property
    def game_lane(self) -> Lane:
        return self._game_lane

    # -- main entry --------------------------------------------------------

    def update(self, snapshot: MotionSnapshot) -> tuple[Action, ...]:
        mode = self._mode
        if mode is ControllerMode.CALIBRATING:
            return ()

        if mode is ControllerMode.WAITING_FOR_RUN:
            # Any lane: the character starts in the game's center regardless
            # of where the user physically stands; requiring a CENTER commit
            # here stranded off-center users in WAITING forever.
            if snapshot.running:
                self._mode = ControllerMode.ACTIVE
                _trace(f"mode {mode.value} -> active "
                       f"(running, {snapshot.steps} steps)")
            return ()
        if mode is ControllerMode.ACTIVE:
            if snapshot.running:
                self._active_grace_until = -1
                return self._forward(snapshot)
            if snapshot.timestamp_ms < self._active_grace_until:
                # Detector re-entry normally flips running immediately.
                # This bounded grace prevents a resume Esc followed by an
                # auto-pause Esc before the next pose callback arrives.
                return ()
            self._mode = ControllerMode.AUTO_PAUSED
            self._paused_steps = snapshot.steps
            self._active_grace_until = -1
            self._resume_wait_signature = None
            return (Action.PAUSE_TOGGLE,)


        if mode is ControllerMode.AUTO_PAUSED:
            new_steps = max(0, snapshot.steps - self._paused_steps)
            # Any lane: the physical pause lane is a per-frame instant value
            # that flickers near boundaries; requiring a match stranded
            # users whose nose had drifted across a boundary.  Three new
            # steps of jogging in place are the only resume condition.
            if new_steps >= RUN_ENTRY_STEPS:
                self._mode = ControllerMode.RESUMING
                self._resuming_since = snapshot.timestamp_ms
                self._resume_wait_signature = None
                _trace("mode auto_paused -> resuming "
                       f"(+{new_steps} steps, sending Esc)")
                return (Action.PAUSE_TOGGLE,)
            return ()

        if mode is ControllerMode.RESUMING:
            # Wait out the grace, keep suppressing actions.
            if snapshot.timestamp_ms - self._resuming_since >= RESUME_GRACE_MS:
                self._mode = ControllerMode.ACTIVE
                self._active_grace_until = (
                    snapshot.timestamp_ms + RUN_STOP_AFTER_MS
                )
                _trace("mode resuming -> active")
        return ()

    def _forward(self, snapshot: MotionSnapshot) -> tuple[Action, ...]:
        forwarded: list[Action] = []
        for action in snapshot.events:
            if action is Action.MOVE_LEFT:
                self._game_lane = Lane(
                    max(int(Lane.LEFT), int(self._game_lane) - 1)
                )
                forwarded.append(action)
            elif action is Action.MOVE_RIGHT:
                self._game_lane = Lane(
                    min(int(Lane.RIGHT), int(self._game_lane) + 1)
                )
                forwarded.append(action)
            elif action in (Action.JUMP, Action.CROUCH):
                forwarded.append(action)
        return tuple(forwarded)
