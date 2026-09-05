"""Low-latency pose pipeline: camera capture + MediaPipe on one QThread.

Threading contract
------------------
* The ``VisionWorker`` QThread exclusively owns ``cv2.VideoCapture`` and the
  ``PoseLandmarker``.
* MediaPipe LIVE_STREAM calls its result callback on an internal MediaPipe
  thread; the callback only converts 33 landmarks into a ``PoseSample`` and
  hands it to ``on_pose`` (the app's pipeline hook).  It never waits.
* ``detect_async`` drops frames while the previous inference is still busy —
  that is the intended low-latency strategy: never queue, never block.
* The worker never sleeps; the UI keeps ticking regardless, so a blocked
  camera or a stalled callback still lets ``MotionDetector.tick`` auto-pause.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .types import LandmarkPoint, PoseSample

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
FRAME_TS_EPS = 1  # strictly-increasing timestamp guard
CROP_MIN_FRACTION = 0.2  # keep enough context for full-body pose


def capture_backend_flag() -> int:
    """Per-platform VideoCapture backend (AVFoundation / DSHOW / ANY)."""
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win") or sys.platform == "cygwin":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_capture(index: int) -> cv2.VideoCapture | None:
    """Open camera `index` with the requested geometry; None on failure."""
    backend = capture_backend_flag()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:  # pragma: no cover - backend dependent
        pass
    return cap


def probe_cameras(limit: int = 4) -> list[int]:
    """Return indices 0..limit-1 that open and deliver a first frame."""
    working: list[int] = []
    for index in range(limit):
        cap = open_capture(index)
        if cap is None:
            continue
        ok, _frame = cap.read()
        cap.release()
        if ok:
            working.append(index)
    return working


# --------------------------------------------------------------------------
# Latest-value mailbox (single shared lock with the app's pipeline)
# --------------------------------------------------------------------------


@dataclass
class CameraInfo:
    index: int | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    buffer_size: int | None = None


class VisionMailbox:
    """Latest-value slots read by the UI; every access under one lock."""

    def __init__(self, lock: threading.RLock) -> None:
        self.lock = lock
        self.latest_frame: np.ndarray | None = None  # RGB uint8 HxWx3
        self.frame_ts_ms: int = -1
        self.landmarks: tuple[LandmarkPoint, ...] | None = None
        self.landmark_ts_ms: int = -1
        self.error: str | None = None
        self.camera: CameraInfo = CameraInfo()
        self.frames_seen: int = 0

    def set_frame(self, frame: np.ndarray | None, ts_ms: int) -> None:
        with self.lock:
            self.latest_frame = frame
            self.frame_ts_ms = ts_ms
            if frame is not None:
                self.frames_seen += 1

    def set_pose(
        self, landmarks: tuple[LandmarkPoint, ...] | None, ts_ms: int,
    ) -> None:
        with self.lock:
            self.landmarks = landmarks
            self.landmark_ts_ms = ts_ms

    def set_error(self, message: str) -> None:
        with self.lock:
            self.error = message


# --------------------------------------------------------------------------
# Vision worker thread
# --------------------------------------------------------------------------


class VisionWorker(QThread):
    """Owns capture + PoseLandmarker; converts results and drops old frames."""

    fatal = Signal(str)  # open/init/read failure -> UI reacts immediately

    def __init__(
        self,
        model_bytes: bytes,
        camera_index: int,
        on_pose: Callable[[PoseSample | None, int], None],
        mailbox: VisionMailbox,
        crop_width: float = 1.0,
        crop_height: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model_bytes = model_bytes
        self._camera_index = camera_index
        self._on_pose = on_pose
        self._mailbox = mailbox
        self._stop = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._landmarker = None
        self._dims_lock = threading.Lock()
        self._pending_dims: dict[int, tuple[int, int]] = {}
        self._last_ts = -1
        self._crop_w = self._clamp_crop(crop_width)
        self._crop_h = self._clamp_crop(crop_height)

    @staticmethod
    def _clamp_crop(fraction: float) -> float:
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            return 1.0
        return min(1.0, max(CROP_MIN_FRACTION, value))

    def set_crop(self, width: float, height: float) -> None:
        """Adjust the live crop; plain float stores are atomic under the
        GIL, so the capture loop needs no lock here."""
        self._crop_w = self._clamp_crop(width)
        self._crop_h = self._clamp_crop(height)

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def run(self) -> None:  # QThread body
        info = self._mailbox.camera
        with self._mailbox.lock:
            info.index = self._camera_index
        try:
            self._open_pose_model()
            self._capture = open_capture(self._camera_index)
            if self._capture is None:
                raise RuntimeError(
                    f"Cannot open camera {self._camera_index}"
                )
            with self._mailbox.lock:
                info.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                info.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
                info.buffer_size = _get_buffer_size(self._capture)
            self._capture_loop()
        except Exception as exc:  # open/init failures surface to the UI
            message = str(exc) or exc.__class__.__name__
            self._mailbox.set_error(message)
            self.fatal.emit(message)
        finally:
            self._teardown()

    # -- pose model --------------------------------------------------------

    def _open_pose_model(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision import RunningMode

        self._mp = mp

        def callback(result, output_image, timestamp_ms):
            self._handle_result(result, int(timestamp_ms))

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=self._model_bytes),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_segmentation_masks=False,
            result_callback=callback,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # -- capture loop ------------------------------------------------------

    def _capture_loop(self) -> None:
        assert self._capture is not None
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 10:
                    raise RuntimeError(
                        "Camera read failed repeatedly; check the connection and retry."
                    )
                time.sleep(0.02)
                continue
            failures = 0
            ts = self._monotonic_ms()
            self._last_ts = ts
            rgb = _mirror_to_rgb(frame)
            rgb = crop_frame(rgb, self._crop_w, self._crop_h)
            self._mailbox.set_frame(rgb, ts)
            h, w = rgb.shape[:2]
            with self._dims_lock:
                self._pending_dims[ts] = (w, h)
                self._prune_dims(ts)
            if self._landmarker is not None:
                mp_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=rgb
                )
                # Busy -> detect_async returns False and drops this frame:
                # exactly the intended low-latency behaviour.
                self._landmarker.detect_async(mp_image, ts)

    def _prune_dims(self, ts: int) -> None:
        stale = [t for t in self._pending_dims if ts - t > 1_000]
        for t in stale:
            self._pending_dims.pop(t, None)

    # -- result callback (MediaPipe internal thread) ------------------------

    def _handle_result(self, result, ts_ms: int) -> None:
        if self._stop.is_set():
            return
        with self._dims_lock:
            dims = self._pending_dims.pop(ts_ms, None)
        sample = None
        landmarks = None
        if dims is not None and result is not None and result.pose_landmarks:
            w, h = dims
            raw = result.pose_landmarks[0]
            pts = tuple(
                LandmarkPoint(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                for lm in raw
            )
            landmarks = pts
            sample = PoseSample(
                timestamp_ms=ts_ms, frame_width=w, frame_height=h,
                landmarks=pts,
            )
        self._mailbox.set_pose(landmarks, ts_ms)
        if sample is not None or dims is not None:
            self._on_pose(sample, ts_ms)

    def _monotonic_ms(self) -> int:
        ts = time.monotonic_ns() // 1_000_000
        if ts <= self._last_ts:
            ts = self._last_ts + FRAME_TS_EPS
        return ts

    # -- teardown ----------------------------------------------------------

    def _teardown(self) -> None:
        # Order per plan: stop flag -> close landmarker -> release capture.
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._landmarker = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _get_buffer_size(cap) -> int | None:
    try:
        value = cap.get(cv2.CAP_PROP_BUFFERSIZE)
        return int(value) if value and value > 0 else None
    except cv2.error:  # pragma: no cover - backend dependent
        return None


def _mirror_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Horizontal mirror first, then RGB; screen == pose coordinate space."""
    return np.ascontiguousarray(cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB))



def crop_frame(
    rgb: np.ndarray, width_fraction: float, height_fraction: float
) -> np.ndarray:
    """Crop width (centered) and height (keep the bottom) independently.

    The vertical anchor keeps the bottom edge: feet and ankles are
    required landmarks for calibration and running detection, while the
    ceiling above the head is expendable.
    """
    if width_fraction >= 1.0 and height_fraction >= 1.0:
        return rgb
    h, w = rgb.shape[:2]
    new_w = max(2, min(w, int(round(w * width_fraction))))
    x0 = (w - new_w) // 2
    new_h = max(2, min(h, int(round(h * height_fraction))))
    y0 = h - new_h  # top-anchored crop: drop ceiling, keep floor
    return np.ascontiguousarray(rgb[y0:y0 + new_h, x0:x0 + new_w])
