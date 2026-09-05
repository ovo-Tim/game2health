"""Immutable domain types shared by vision, motion, input and UI layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

# MediaPipe Pose landmark indices used across the app.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# The eight points that decide lane membership.
CORE_POINTS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

# Points required for full-body-in-frame calibration.
FULL_BODY_POINTS = (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """Normalized 2.5D landmark; x,y in [0,1] of the (mirrored) frame."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


@dataclass(frozen=True, slots=True)
class PoseSample:
    """One pose result with its acquisition metadata."""

    timestamp_ms: int
    frame_width: int
    frame_height: int
    landmarks: tuple[LandmarkPoint, ...]  # 33 entries, MediaPipe order


class Lane(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class Action(Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    JUMP = "jump"
    CROUCH = "crouch"
    PAUSE_TOGGLE = "pause_toggle"


@dataclass(frozen=True, slots=True)
class MotionSnapshot:
    """Detector verdict for one instant; events are one-shot actions."""

    timestamp_ms: int
    pose_valid: bool
    stable_lane: Lane | None
    running: bool
    events: tuple[Action, ...] = ()
    steps: int = 0  # accepted alternating steps since detector reset


class ControllerMode(Enum):
    CALIBRATING = "calibrating"
    WAITING_FOR_RUN = "waiting_for_run"
    ACTIVE = "active"
    AUTO_PAUSED = "auto_paused"
    RESUMING = "resuming"
