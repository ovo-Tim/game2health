"""Synthetic pose builders for tests.

All poses are described in aspect-corrected "scaled space": x coordinates are
in normalized-image-height units (x_s = raw_x * w/h), y is the raw normalized
y.  The builder converts back to raw landmark x per frame aspect, then
translates the whole body horizontally so the hip mid sits at a target raw x
(e.g. 0.5 = middle lane in every aspect).  Pure translations never change
angles, so the same scaled shape yields identical knee angles and verdicts in
4:3 and 16:9 frames.
"""

from __future__ import annotations

from game2health.types import LandmarkPoint, PoseSample

# Scaled-space (x_s, y) per landmark, hip mid at x_s = 0.  torso_length 0.20.
BASE = {
    0: (0.000, 0.22),  # nose
    1: (0.000, 0.20), 2: (0.000, 0.20),  # eye inner
    3: (0.000, 0.19), 4: (0.000, 0.19),  # eyes
    5: (0.000, 0.20), 6: (0.000, 0.20),  # ears
    7: (0.000, 0.21), 8: (0.000, 0.21), 9: (0.000, 0.20),
    10: (0.000, 0.21),
    11: (-0.085, 0.36), 12: (0.085, 0.36),  # shoulders
    13: (-0.115, 0.44), 14: (0.115, 0.44),  # elbows
    15: (-0.125, 0.55), 16: (0.125, 0.55),  # wrists
    17: (-0.125, 0.58), 18: (0.125, 0.58), 19: (-0.125, 0.58),
    20: (0.125, 0.58), 21: (-0.125, 0.58), 22: (0.125, 0.58),
    23: (-0.050, 0.56), 24: (0.050, 0.56),  # hips
    25: (-0.060, 0.74), 26: (0.060, 0.74),  # knees
    27: (-0.070, 0.93), 28: (0.070, 0.93),  # ankles
    29: (-0.080, 0.95), 30: (0.080, 0.95),  # heels
    31: (-0.090, 0.96), 32: (0.090, 0.96),  # foot index
}

# Movement presets: idx -> (x_s_delta, y_delta) applied to the base pose.
# Lifts are clearly above detector thresholds with torso_length = 0.20:
# run lift 0.024, jump 0.024, crouch sink 0.024.
RUN_LIFT_LEFT = {25: (0.0, -0.035), 27: (0.0, -0.035), 29: (0.0, -0.035),
                 31: (0.0, -0.035)}
RUN_LIFT_RIGHT = {26: (0.0, -0.035), 28: (0.0, -0.035), 30: (0.0, -0.035),
                  32: (0.0, -0.035)}
JUMP_UP = {23: (0.0, -0.030), 24: (0.0, -0.030), 25: (0.0, -0.020),
           26: (0.0, -0.020), 27: (0.0, -0.050), 28: (0.0, -0.050),
           29: (0.0, -0.050), 30: (0.0, -0.050), 31: (0.0, -0.050),
           32: (0.0, -0.050)}
# Hip down 0.04 (> 0.036) with flexed knees (~110 deg at the knee).
CROUCH_DOWN = {23: (0.0, 0.040), 24: (0.0, 0.040),
               25: (-0.025, -0.125), 26: (0.025, -0.125)}
# Half squat: hip sink 0.14 x torso, knee angle ~133 deg — above the old
# 0.18 sink / 125 deg limits, inside the current ones.
CROUCH_LIGHT = {23: (0.0, 0.028), 24: (0.0, 0.028),
                25: (-0.045, -0.085), 26: (0.045, -0.085)}
# Hip down with straight knees: bend-over only, never a crouch.
BEND_OVER = {23: (0.0, 0.040), 24: (0.0, 0.040)}


class PoseBuilder:
    """Builds pose samples; landmarks are raw-normalized per aspect."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.aspect = width / height

    def to_raw_x_delta(self, idx: int, raw_x: float,
                       center_raw: float = 0.5) -> tuple[float, float]:
        """Delta moving landmark `idx` to raw x `raw_x` (keeps its y)."""
        x_s, _y = BASE[idx]
        ox = center_raw * self.aspect
        dx = raw_x * self.aspect - x_s - ox
        return (dx, 0.0)

    def landmarks(self, center_raw: float = 0.5, deltas=None,
                  visibility: float = 0.95,
                  hidden=()) -> tuple[LandmarkPoint, ...]:
        if deltas is None:
            deltas = {}
        ox = center_raw * self.aspect
        pts: list[LandmarkPoint] = []
        for idx in range(33):
            x_s, y = BASE[idx]
            dx, dy = deltas.get(idx, (0.0, 0.0))
            raw_x = (x_s + dx + ox) / self.aspect
            vis = 0.0 if idx in hidden else visibility
            pts.append(LandmarkPoint(x=raw_x, y=y + dy, z=0.0, visibility=vis))
        return tuple(pts)

    def sample(self, ts: int, center_raw: float = 0.5, deltas=None,
               visibility: float = 0.95, hidden=()) -> PoseSample:
        return PoseSample(
            timestamp_ms=ts,
            frame_width=self.width,
            frame_height=self.height,
            landmarks=self.landmarks(center_raw, deltas, visibility, hidden),
        )

    def stream(self, segments, dt_ms: int = 33, ts0: int = 0):
        """Yield (ts, sample) pairs; each segment is (frames, sample_kwargs).

        Pass None as a segment's kwargs to emit that many "no pose" instants
        (useful to advance time without pose data).  ``ts0`` lets a follow-up
        stream continue a previous timeline monotonically.
        """
        ts = ts0
        for frames, kwargs in segments:
            for _ in range(frames):
                yield ts, (None if kwargs is None else self.sample(ts, **kwargs))
                ts += dt_ms


def run_motion(detector, sm, frames):
    """Drive detector + state machine over (ts, sample) frames.

    Returns list of (ts, snapshot, actions).
    """
    out = []
    for ts, sample in frames:
        snap = detector.update(sample) if sample is not None else detector.tick(ts)
        actions = sm.update(snap) if sm is not None else ()
        out.append((ts, snap, actions))
    return out
