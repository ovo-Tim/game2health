"""PySide6 assembly: settings/calibration page, control page, wiring.

Owns the shared pipeline lock, the mailbox, the motion objects, the key
emitter and the VisionWorker.  The MediaPipe result callback (a foreign
thread) and the main-thread QTimer both serialize through ``self._lock``;
nothing mutates the detector/state machine outside it.  The UI thread never
sleeps; it ticks MotionDetector so timeouts fire even when the camera is
blocked or has stopped producing results.
"""

from __future__ import annotations

import os
import sys
import threading
import time

from PySide6.QtCore import QPointF, QRectF, QSettings, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from .bridge import GAME_URL, GameBridge

from .input import (
    BackendUnavailable,
    InputKey,
    KeyEmitter,
    PynputBackend,
    UInputBackend,
    accessibility_trusted,
    default_mapping,
    detect_backend_kind,
)
from .motion import (
    LANE_LEFT_EDGE,
    LANE_RIGHT_EDGE,
    JUMP_LIFT,
    RUN_LIFT,
    CalibrationPhase,
    CalibrationTracker,
    ControllerStateMachine,
    MotionDetector,
)
from .types import (
    Action,
    ControllerMode,
    Lane,
    LandmarkPoint,
    PoseSample,
)
from .vision import VisionMailbox, VisionWorker

KEY_NAMES = {
    "left": "Left", "right": "Right", "up": "Up", "down": "Down",
    "esc": "Esc", "space": "Space", "w": "W", "a": "A", "s": "S", "d": "D",
    "p": "P",
}

OUTPUT_KEYBOARD = "keyboard"
OUTPUT_WEBSOCKET = "websocket"

# Horizontal center-crop choices; 1.0 = full frame. Labels shown in the UI.
CROP_CHOICES: tuple[tuple[str, float], ...] = (
    ("Full", 1.0), ("80%", 0.8), ("65%", 0.65), ("50%", 0.5),
)

MODE_TEXT = {
    ControllerMode.WAITING_FOR_RUN: "Waiting for run",
    ControllerMode.ACTIVE: "Controlling",
    ControllerMode.CALIBRATING: "Calibrating…",
    ControllerMode.AUTO_PAUSED: "Auto-paused",
    ControllerMode.RESUMING: "Resuming",
}

BODY_PART_LABELS = {
    0: "Head", 11: "L shoulder", 12: "R shoulder", 13: "L elbow", 14: "R elbow",
    15: "L wrist", 16: "R wrist", 23: "L hip", 24: "R hip", 25: "L knee",
    26: "R knee", 27: "L ankle", 28: "R ankle", 29: "L heel", 30: "R heel",
    31: "L toe", 32: "R toe",
}

BADGE_TEXT = {
    Action.MOVE_LEFT: "← LEFT",
    Action.MOVE_RIGHT: "→ RIGHT",
    Action.JUMP: "↑ JUMP",
    Action.CROUCH: "↓ CROUCH",
}

SKELETON_LINES = (
    (0, 1), (1, 3), (0, 2), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

WAYLAND_INSTRUCTIONS = (
    "Run the following steps as a regular user, then click Re-detect (re-login if needed):\n"
    "  sudo install -Dm644 packaging/linux/game2health-uinput.conf "
    "/etc/modules-load.d/game2health-uinput.conf\n"
    "  sudo install -Dm644 packaging/linux/70-game2health-uinput.rules "
    "/etc/udev/rules.d/70-game2health-uinput.rules\n"
    "  sudo modprobe uinput\n"
    "  sudo udevadm control --reload-rules\n"
    "  sudo udevadm trigger --subsystem-match=misc"
)


def load_model_bytes() -> bytes:
    from importlib.resources import files

    return files("game2health").joinpath(
        "assets/pose_landmarker_full.task"
    ).read_bytes()


# ---------------------------------------------------------------------------
# Video widget: mirrored frame, lane overlay, skeleton, badges
# ---------------------------------------------------------------------------


class Badge:
    __slots__ = ("text", "born_ms")

    def __init__(self, text: str, born_ms: int) -> None:
        self.text = text
        self.born_ms = born_ms


class VideoWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame = None
        self._landmarks = None
        self._mode = None
        self._stable_lane = None
        self._badges: list[Badge] = []
        self._missing: tuple[int, ...] = ()
        self._image = None
        self._guides: list[tuple] = []
        self.setMinimumSize(420, 240)

    def update_view(self, frame, landmarks, mode, stable_lane, badges,
                    missing=(), guides=()) -> None:
        self._frame = frame
        self._landmarks = landmarks
        self._mode = mode
        self._stable_lane = stable_lane
        self._badges = badges
        self._missing = missing
        self._guides = list(guides)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 20, 24))
        frame = self._frame
        if frame is None:
            p.setPen(QColor(140, 146, 158))
            p.drawText(self.rect(), Qt.AlignCenter, "No picture — check the camera")
            return
        h, w = frame.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        dw, dh = w * scale, h * scale
        ox = (self.width() - dw) / 2.0
        oy = (self.height() - dh) / 2.0
        # Rebuild every paint from a strongly-referenced buffer: QImage does
        # not own the numpy data, so a cached image kept pointing at the
        # first frame's buffer — once freed, the picture degraded to black
        # and only the skeleton remained visible.
        buf = frame.data
        image = QImage(buf, w, h, 3 * w, QImage.Format_RGB888)
        p.drawImage(int(ox), int(oy),
                    image.scaled(int(dw), int(dh),
                                 Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation))
        if self._mode is not ControllerMode.CALIBRATING:
            bounds = (0.0, LANE_LEFT_EDGE, LANE_RIGHT_EDGE, 1.0)
            for lane in (Lane.LEFT, Lane.CENTER, Lane.RIGHT):
                x0 = ox + bounds[int(lane)] * dw
                zone_w = (bounds[int(lane) + 1] - bounds[int(lane)]) * dw
                color = QColor(0, 0, 0, 120)
                if lane is self._stable_lane:
                    color = QColor(60, 220, 130, 70)
                p.fillRect(int(x0), int(oy), int(zone_w) + 1, int(dh), color)
            p.setPen(QPen(QColor(255, 255, 255, 160), 2))
            for edge in (LANE_LEFT_EDGE, LANE_RIGHT_EDGE):
                lx = ox + edge * dw
                p.drawLine(int(lx), int(oy), int(lx), int(oy + dh))
            font = p.font()
            font.setBold(True)
            font.setPointSize(max(10, int((LANE_RIGHT_EDGE - LANE_LEFT_EDGE) * dw / 6)))
            p.setFont(font)
            for idx, lane in enumerate((Lane.LEFT, Lane.CENTER, Lane.RIGHT)):
                x0 = ox + bounds[int(lane)] * dw
                zone_w = (bounds[int(lane) + 1] - bounds[int(lane)]) * dw
                p.setPen(QColor(255, 255, 255, 210))
                p.drawText(QRectF(x0, oy, zone_w, zone_w), Qt.AlignCenter,
                           str(idx + 1))
            p.setFont(QFont())
        else:
            # Calibration keeps the neutral thirds as a positioning guide.
            p.setPen(QPen(QColor(255, 255, 255, 130), 1))
            for edge in (1.0 / 3.0, 2.0 / 3.0):
                lx = ox + edge * dw
                p.drawLine(int(lx), int(oy), int(lx), int(oy + dh))
        self._paint_skeleton(p, ox, oy, w, h, scale)
        self._paint_guides(p, ox, oy, w, h, scale)
        if self._mode is not ControllerMode.CALIBRATING:
            self._paint_badges(p)
        p.end()

    def _paint_skeleton(self, p: QPainter, ox: float, oy: float,
                        w: int, h: int, scale: float) -> None:
        lms = self._landmarks
        if not lms or len(lms) < 33:
            return
        pts = []
        for lm in lms:
            if lm.visibility < 0.3:
                pts.append(None)
            else:
                pts.append((ox + lm.x * w * scale, oy + lm.y * h * scale))
        p.setPen(QPen(QColor(90, 220, 255, 220), 3))
        for a, b in SKELETON_LINES:
            pa, pb = pts[a], pts[b]
            if pa is not None and pb is not None:
                p.drawLine(int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1]))

    def _paint_guides(self, p: QPainter, ox: float, oy: float,
                      w: int, h: int, scale: float) -> None:
        for x_norm, line_y, cur_y, lifted, ratio in self._guides:
            cx = ox + x_norm * w * scale
            ly = oy + line_y * h * scale
            half = 0.055 * w * scale
            color = QColor(60, 230, 120, 230) if lifted \
                else QColor(255, 200, 60, 170)
            p.setPen(QPen(color, 3, Qt.DashLine))
            p.drawLine(int(cx - half), int(ly), int(cx + half), int(ly))
            # current joint position dot
            cy = oy + cur_y * h * scale
            p.setPen(QPen(color, 2))
            p.drawEllipse(QPointF(cx, cy), 4.0, 4.0)
            p.setPen(QColor(255, 255, 255, 200))
            p.drawText(QPointF(cx + half + 6, ly + 4),
                       f"{min(999, round(ratio * 100))}%")

    def _paint_badges(self, p: QPainter) -> None:
        now = time.monotonic() * 1000.0
        shown = [b for b in self._badges if now - b.born_ms < 900.0]
        if not shown:
            return
        font = p.font()
        font.setBold(True)
        font.setPointSize(17)
        p.setFont(font)
        widths = [p.fontMetrics().horizontalAdvance(b.text) + 28
                  for b in shown]
        total = sum(widths) + 10 * (len(shown) - 1)
        cur = (self.width() - total) / 2.0
        cy = self.height() / 2.0 - 50.0
        for badge, bw in zip(shown, widths):
            age = now - badge.born_ms
            alpha = 255 if age <= 600 else max(
                0, int(255 * (1 - (age - 600) / 300.0)))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(150 * alpha / 255)))
            p.drawRoundedRect(int(cur), int(cy), bw, 42, 12, 12)
            p.setPen(QColor(255, 255, 255, alpha))
            p.drawText(QRectF(cur, cy, bw, 42), Qt.AlignCenter, badge.text)
            cur += bw + 10
        p.setFont(QFont())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class Game2HealthWindow(QMainWindow):
    backend_ready = Signal()
    cameras_ready = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Game2Health Motion Controller")
        self.resize(1020, 720)

        self.settings_store = QSettings("Game2Health", "Game2Health")
        saved_output = str(
            self.settings_store.value("control/output", OUTPUT_KEYBOARD)
        )
        self._output_mode = (
            saved_output if saved_output in {OUTPUT_KEYBOARD, OUTPUT_WEBSOCKET}
            else OUTPUT_KEYBOARD
        )
        self._lock = threading.RLock()
        self._mailbox = VisionMailbox(self._lock)
        self._model_bytes = load_model_bytes()

        if os.environ.get("G2H_DEBUG"):
            from . import motion as _motion
            _motion.enable_debug()

        self._pipeline = self._fresh_pipeline()
        self._last_cal_status = None
        self._worker = None
        self._backend = None
        self._emitter = None
        self._bridge: GameBridge | None = None
        self._current_camera = None
        self._last_dims = None
        self._dims_changed = False
        self._countdown_value = 0
        self._game_start_armed = False
        self._pending_cal_start = False
        self._tick_counter = 0
        self._mapping = default_mapping()
        self._feedback: list[Badge] = []
        self._crop_w, self._crop_h = self._load_crops()
        self._ui = QStackedWidget()
        self.setCentralWidget(self._ui)
        self._settings_page = self._build_settings_page()
        self._control_page = self._build_control_page()
        self._ui.addWidget(self._settings_page)
        self._ui.addWidget(self._control_page)

        self._load_mapping()
        self._configure_output_mode(self._output_mode)
        self.backend_ready.connect(self._backend_probe_done)
        self.cameras_ready.connect(self._populate_camera_combo)
        self._probe_backend()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._ui_tick)
        self._timer.start(33)
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._countdown_step)

        self._refresh_cameras()

    # ------------------------------------------------------------------
    # pipeline helpers (always under self._lock)
    # ------------------------------------------------------------------

    def _fresh_pipeline(self):
        return (
            CalibrationTracker(), MotionDetector(), ControllerStateMachine(),
        )

    def _reset_pipeline(self) -> None:
        with self._lock:
            self._feedback.clear()
            self._pipeline = self._fresh_pipeline()
            self._last_dims = None
            self._last_cal_status = None
            self._dims_changed = False
        if self._emitter is not None:
            self._emitter.release_all()

    # ------------------------------------------------------------------
    # settings page construction
    # ------------------------------------------------------------------

    def _section(self, title: str) -> QLabel:
        label = QLabel(title)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(16, 12, 16, 12)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(170)
        self.camera_combo.currentIndexChanged.connect(
            lambda _i: self._on_camera_selected())
        cam_row.addWidget(self.camera_combo)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_cameras)
        cam_row.addWidget(QLabel("Width:"))
        self.crop_w_combo = self._make_crop_combo(self._crop_w)
        cam_row.addWidget(self.crop_w_combo)
        cam_row.addWidget(QLabel("Height:"))
        self.crop_h_combo = self._make_crop_combo(self._crop_h)
        cam_row.addWidget(self.crop_h_combo)
        self.retry_button = QPushButton("Retry")
        self.retry_button.clicked.connect(
            lambda: self._select_camera(self._current_camera))
        cam_row.addWidget(self.retry_button)
        self.cam_error_label = QLabel("")
        self.cam_error_label.setStyleSheet("color: #ff8a8a;")
        cam_row.addWidget(self.cam_error_label)
        cam_row.addStretch(1)
        root.addLayout(cam_row)
        self.cam_info_label = QLabel("")
        self.cam_info_label.setStyleSheet("color: #9aa0aa;")
        root.addWidget(self.cam_info_label)

        body = QHBoxLayout()
        self._preview = VideoWidget()
        body.addWidget(self._preview, 3)

        side = QVBoxLayout()
        side.addWidget(self._section("Control output"))
        self.output_combo = QComboBox()
        self.output_combo.addItem("Keyboard keys (Esc pause)", OUTPUT_KEYBOARD)
        self.output_combo.addItem(
            "Game userscript (WebSocket)", OUTPUT_WEBSOCKET
        )
        self.output_combo.setCurrentIndex(
            self.output_combo.findData(self._output_mode)
        )
        self.output_combo.currentIndexChanged.connect(
            self._on_output_mode_changed
        )
        side.addWidget(self.output_combo)
        self.bridge_label = QLabel("")
        self.bridge_label.setWordWrap(True)
        side.addWidget(self.bridge_label)
        side.addWidget(self._section("Key mapping"))
        self._key_rows = QVBoxLayout()
        side.addLayout(self._key_rows)
        side.addWidget(self._section("Keyboard device"))
        self.backend_label = QLabel("Detecting keyboard device…")
        self.backend_label.setWordWrap(True)
        side.addWidget(self.backend_label)
        retry_be = QPushButton("Re-detect")
        retry_be.clicked.connect(self._probe_backend)
        side.addWidget(retry_be)
        side.addWidget(self._section("Test key"))
        test_row = QHBoxLayout()
        self.test_combo = QComboBox()
        for key, name in KEY_NAMES.items():
            self.test_combo.addItem(name, key)
        test_row.addWidget(self.test_combo)
        self.test_button = QPushButton("Send test key")
        self.test_button.clicked.connect(self._send_test_key)
        test_row.addWidget(self.test_button)
        side.addLayout(test_row)
        self.test_hint = QLabel("Sends after a 2 s countdown — focus the target window first.")
        self.test_hint.setWordWrap(True)
        side.addWidget(self.test_hint)
        side.addWidget(self._section("Full-body calibration"))
        self.cal_label = QLabel("Waiting for camera picture…")
        self.cal_label.setWordWrap(True)
        side.addWidget(self.cal_label)
        self.missing_label = QLabel("")
        self.missing_label.setWordWrap(True)
        side.addWidget(self.missing_label)
        self.start_button = QPushButton("Start control")
        self.start_button.setMinimumHeight(42)
        self.start_button.clicked.connect(self._begin_control)
        side.addWidget(self.start_button)
        side.addStretch(1)
        body.addLayout(side, 2)
        root.addLayout(body)

        self._key_action_rows: dict[Action, QComboBox] = {}
        rows = [
            (Action.MOVE_LEFT, "Move left"),
            (Action.MOVE_RIGHT, "Move right"),
            (Action.JUMP, "Jump"),
            (Action.CROUCH, "Crouch"),
            (Action.PAUSE_TOGGLE, "Pause / resume"),
        ]
        for action, title in rows:
            row = QHBoxLayout()
            label = QLabel(title)
            label.setMinimumWidth(96)
            combo = QComboBox()
            for key, name in KEY_NAMES.items():
                combo.addItem(name, key)
            # NOTE: currentIndexChanged fires while addItem inserts the first
            # item; connect only after ALL rows exist, otherwise
            # _apply_mapping snapshots index-0 values for rows not yet built
            row.addWidget(label)
            row.addWidget(combo)
            row.addStretch(1)
            self._key_rows.addLayout(row)
            self._key_action_rows[action] = combo
        for combo in self._key_action_rows.values():
            combo.currentIndexChanged.connect(self._on_mapping_changed)
        return page

    # ------------------------------------------------------------------
    # control page construction
    # ------------------------------------------------------------------

    def _build_control_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 10, 12, 10)
        top = QHBoxLayout()
        self.mode_label = QLabel("—")
        font = self.mode_label.font()
        font.setBold(True)
        font.setPointSize(16)
        self.mode_label.setFont(font)
        top.addWidget(self.mode_label)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        top.addWidget(self.hint_label, 1)
        self.countdown_label = QLabel("")
        cf = QFont()
        cf.setBold(True)
        cf.setPointSize(26)
        self.countdown_label.setFont(cf)
        top.addWidget(self.countdown_label)
        stop_btn = QPushButton("Stop control")
        stop_btn.clicked.connect(self._stop_control)
        top.addWidget(stop_btn)
        recal_btn = QPushButton("Recalibrate")
        recal_btn.clicked.connect(self._recalibrate)
        top.addWidget(recal_btn)
        root.addLayout(top)
        self.control_video = VideoWidget()
        root.addWidget(self.control_video, 1)
        return page

    # ------------------------------------------------------------------
    # key mapping
    def _load_mapping(self) -> None:
        defaults = default_mapping()
        # Schema 1 (pre-fix builds) persisted index-0 leftovers for rows the
        # build-time signal storm had not configured; stored values from that
        # era are untrustworthy -> reset to defaults exactly once.
        schema = self.settings_store.value("keys/schema", 0)
        fresh_start = str(schema) != "2"
        for action, combo in self._key_action_rows.items():
            saved = defaults[action].value if fresh_start else \
                self.settings_store.value(f"keys/{action.value}",
                                          defaults[action].value)
            idx = combo.findData(saved)
            if idx < 0:
                # Corrupted value in the settings store: reset to the
                # built-in default instead of the first combo entry.
                idx = combo.findData(defaults[action].value)
            combo.blockSignals(True)
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)
        self.settings_store.setValue("keys/schema", 2)
        self._apply_mapping()

    def _on_mapping_changed(self, _index: int) -> None:
        self._apply_mapping()

    def _make_crop_combo(self, current: float) -> QComboBox:
        combo = QComboBox()
        for label, fraction in CROP_CHOICES:
            combo.addItem(label, fraction)
        fractions = [f for _label, f in CROP_CHOICES]
        idx = fractions.index(current) if current in fractions else 0
        combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._on_crop_changed)
        return combo

    def _load_crops(self) -> tuple[float, float]:
        store = self.settings_store
        # Single-axis builds stored only video/crop: migrate it to width.
        width = store.value("video/crop_w", store.value("video/crop", 1.0))
        return (VisionWorker._clamp_crop(width),
                VisionWorker._clamp_crop(store.value("video/crop_h", 1.0)))

    def _on_crop_changed(self, _index: int) -> None:
        width = float(self.crop_w_combo.currentData())
        height = float(self.crop_h_combo.currentData())
        self._crop_w, self._crop_h = width, height
        store = self.settings_store
        store.setValue("video/crop_w", width)
        store.setValue("video/crop_h", height)
        store.remove("video/crop")  # migrated
        if self._worker is not None:
            self._worker.set_crop(width, height)

    # ------------------------------------------------------------------
    # control output
    # ------------------------------------------------------------------

    def _on_output_mode_changed(self, _index: int) -> None:
        mode = str(self.output_combo.currentData())
        if mode == self._output_mode:
            return
        self._output_mode = mode
        self.settings_store.setValue("control/output", mode)
        self._configure_output_mode(mode)
        self._update_settings_ui()

    def _configure_output_mode(self, mode: str) -> None:
        if mode == OUTPUT_WEBSOCKET:
            if self._bridge is None:
                bridge = GameBridge(parent=self)
                bridge.status_changed.connect(self._on_bridge_status)
                bridge.client_changed.connect(
                    lambda _connected: self._update_settings_ui()
                )
                self._bridge = bridge
            self.bridge_label.setText(
                self._bridge.status + self._userscript_help()
            )
        else:
            if self._bridge is not None:
                self._bridge.close()
                self._bridge = None
            self.bridge_label.setText(
                "WebSocket integration disabled; ordinary keyboard keys "
                "and Esc are used."
            )

    def _on_bridge_status(self, status: str) -> None:
        self.bridge_label.setText(status + self._userscript_help())

    @staticmethod
    def _userscript_help() -> str:
        return (
            "\nInstall userscripts/game2health-subway.user.js in Tampermonkey "
            "once. Chrome 147+ asks for Local Network Access on first use; "
            "choose Allow."
        )

    def _output_ready(self) -> bool:
        if self._output_mode == OUTPUT_WEBSOCKET:
            return self._bridge is not None and self._bridge.listening
        return self._backend is not None and self._emitter is not None

    def _open_game(self) -> None:
        if not QDesktopServices.openUrl(QUrl(GAME_URL)):
            self._on_bridge_status(
                "Could not open the default browser; open the game URL manually"
            )

    def _apply_mapping(self) -> None:
        mapping = {}
        for action, combo in self._key_action_rows.items():
            key = combo.currentData()
            mapping[action] = InputKey(key)
            self.settings_store.setValue(f"keys/{action.value}", key)
        self._mapping = mapping
        if self._emitter is not None:
            self._emitter.set_mapping(mapping)

    # ------------------------------------------------------------------
    # backend
    # ------------------------------------------------------------------

    def _probe_backend(self) -> None:
        self._backend = None
        self._emitter = None
        self._probe_result = None
        self.backend_label.setText("Detecting keyboard device…")
        self._update_settings_ui()
        threading.Thread(target=self._probe_backend_work, daemon=True).start()

    def _probe_backend_work(self) -> None:
        # Never construct pynput here: its Controller must be created on
        # the macOS main thread (HIToolbox asserts the main queue).
        try:
            kind, reason = detect_backend_kind(sys.platform, dict(os.environ))
        except Exception as exc:  # pragma: no cover - defensive
            kind, reason = None, f"{type(exc).__name__}: {exc}"
        self._probe_result = (kind, reason)
        self.backend_ready.emit()

    def _backend_probe_done(self) -> None:
        kind, message = self._probe_result
        is_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        if kind is None:
            text = "Keyboard device unavailable: " + message
            if is_wayland:
                text += "\n" + WAYLAND_INSTRUCTIONS
            self.backend_label.setText(text)
            self._update_settings_ui()
            return
        # Construct on the main thread (pynput requirement on macOS).
        try:
            if kind == "uinput":
                backend = UInputBackend()
            else:
                backend = PynputBackend()
        except (BackendUnavailable, Exception) as exc:  # noqa: BLE001
            text = "Keyboard device unavailable: " + str(exc)
            if is_wayland:
                text += "\n" + WAYLAND_INSTRUCTIONS
            self.backend_label.setText(text)
            self._update_settings_ui()
            return
        self._backend = backend
        self._emitter = KeyEmitter(backend)
        self._emitter.start()
        self._apply_mapping()
        if isinstance(backend, UInputBackend):
            text = "Native Wayland keyboard device ready (Game2Health Virtual Keyboard)."
        elif sys.platform == "darwin" and not accessibility_trusted():
            text = ("pynput key injection is BLOCKED: this terminal does not have "
                    "Accessibility permission, so macOS silently drops every key. "
                    "System Settings → Privacy & Security → Accessibility → add "
                    "this terminal app, then fully quit and restart it.")
        else:
            text = "pynput key injection ready."
        self.backend_label.setText(text)
        self._update_settings_ui()

    # ------------------------------------------------------------------
    # camera probing / selection
    # ------------------------------------------------------------------

    def _refresh_cameras(self) -> None:
        from .vision import probe_cameras

        def work():
            try:
                return probe_cameras(4)
            except Exception:  # pragma: no cover - camera enumeration
                return []

        threading.Thread(
            target=lambda: self.cameras_ready.emit(work()), daemon=True
        ).start()

    def _populate_camera_combo(self, found) -> None:
        combo = self.camera_combo
        combo.blockSignals(True)
        combo.clear()
        if found:
            for idx in found:
                combo.addItem(f"Camera {idx}", idx)
            combo.setEditable(False)
        else:
            combo.addItem("No device found — type 0–9 manually", None)
            combo.setEditable(True)
        combo.blockSignals(False)
        if found:
            self._select_camera(found[0])
        else:
            self._current_camera = None
            self._update_settings_ui()

    def _on_camera_selected(self) -> None:
        data = self.camera_combo.currentData()
        if data is None:
            text = self.camera_combo.currentText().strip()
            if not text.isdigit():
                return
            data = int(text)
        if data != self._current_camera:
            self._select_camera(data)

    def _select_camera(self, index: int | None) -> None:
        if index is None or index == self._current_camera and self._worker:
            return
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(4000)
            self._worker = None
        with self._lock:
            self._mailbox.error = None
            self._mailbox.set_frame(None, -1)
            self._mailbox.set_pose(None, -1)
            self._mailbox.camera.index = None
        self._reset_pipeline()
        self._current_camera = index
        self._stop_on_top()
        if self._ui.currentWidget() is self._control_page:
            self._ui.setCurrentWidget(self._settings_page)
        self.cam_error_label.setText("Opening camera…")
        self._update_settings_ui()
        worker = VisionWorker(self._model_bytes, index, self._on_pose,
                              self._mailbox,
                              crop_width=self._crop_w,
                              crop_height=self._crop_h)
        self._worker = worker
        worker.start()

    # ------------------------------------------------------------------
    # pose callback (MediaPipe thread, serialized under self._lock)
    # ------------------------------------------------------------------

    def _on_pose(self, sample: PoseSample | None, ts_ms: int) -> None:
        with self._lock:
            tracker, detector, sm = self._pipeline
            try:
                if sample is not None:
                    dims = (sample.frame_width, sample.frame_height)
                    if self._last_dims is not None and dims != self._last_dims:
                        self._dims_changed = True
                        return
                    self._last_dims = dims
                if sm.mode is ControllerMode.CALIBRATING:
                    status = tracker.update(sample)
                    self._last_cal_status = status
                    if (status.phase is CalibrationPhase.COMPLETE
                            and status.result is not None
                            and not detector.calibrated):
                        detector.set_baseline(status.result)
                else:
                    snap = (detector.update(sample)
                            if sample is not None else detector.tick(ts_ms))
                    self._dispatch(detector, sm, snap)
            except Exception:  # pragma: no cover - never kill MP thread
                import traceback
                traceback.print_exc()

    def _dispatch(self, detector, sm, snap) -> None:
        actions = sm.update(snap)
        if not actions:
            return
        mode = sm.mode
        use_bridge = self._output_mode == OUTPUT_WEBSOCKET
        now_ms = time.monotonic() * 1000.0
        for action in actions:
            if action is Action.PAUSE_TOGGLE:
                target = "GAME" if use_bridge else "ESC"
                text = (f"⏸ {target}"
                        if mode is ControllerMode.AUTO_PAUSED
                        else f"▶ {target}")
            else:
                text = BADGE_TEXT[action]
            self._feedback.append(Badge(text, now_ms))
        if use_bridge:
            if self._bridge is not None:
                if Action.PAUSE_TOGGLE in actions:
                    command = ("pause"
                               if mode is ControllerMode.AUTO_PAUSED
                               else "resume")
                    self._bridge.command(command)
                self._bridge.emit(actions)
        elif self._emitter is not None:
            self._emitter.emit(actions)

    # ------------------------------------------------------------------
    # UI pump (main thread)
    # ------------------------------------------------------------------

    def _leg_guides(self, det: MotionDetector) -> list[tuple]:
        """Draw knee run lines and the higher ankle jump lines."""
        cal = det._cal
        if cal is None or cal.torso_length <= 0 or not det._smooth:
            return []
        run_thr = RUN_LIFT * cal.torso_length
        jump_thr = JUMP_LIFT * cal.torso_length
        guides = []
        for knee_idx, ankle_idx, base_knee, base_ankle in (
                (25, 27, cal.knee_y_left, cal.ankle_y_left),
                (26, 28, cal.knee_y_right, cal.ankle_y_right)):
            leg_up = det._leg_up.get(knee_idx, False)
            for idx, base, threshold in (
                    (knee_idx, base_knee, run_thr),
                    (ankle_idx, base_ankle, jump_thr)):
                cur = det._smooth.get(idx)
                if cur is None:
                    continue
                lift = base - cur[1]
                active = leg_up if idx == knee_idx else lift >= threshold
                guides.append((
                    cur[0], base - threshold, cur[1], active,
                    lift / threshold if threshold > 0 else 0.0,
                ))
        return guides

    def _ui_tick(self) -> None:
        self._tick_counter += 1
        now_ms = time.monotonic() * 1000.0
        with self._lock:
            frame = self._mailbox.latest_frame
            landmarks = self._mailbox.landmarks
            tracker, detector, sm = self._pipeline
            stable = detector.raw_lane
            if self._tick_counter % 2 == 0:
                snap = detector.tick(now_ms)
                self._dispatch(detector, sm, snap)
            mode = sm.mode
            error = self._mailbox.error
            badges = list(self._feedback)
            guides = self._leg_guides(detector)
            status = self._last_cal_status
            dims_changed = self._dims_changed

        if dims_changed:
            self._on_vision_fatal("Frame size changed — control stopped and calibration cleared.")
            return

        if self._ui.currentWidget() is self._control_page:
            self.control_video.update_view(frame, landmarks, mode, stable,
                                           badges, guides=guides)
            # Uncalibrated start: begin the countdown once calibration
            # completes on the control page.
            if (self._pending_cal_start and not self._countdown_timer.isActive()
                    and self._pipeline[1].calibrated):
                self._pending_cal_start = False
                self._start_countdown()
            self._refresh_control_status(mode, stable, landmarks,
                                         detector._step_count)
        else:
            info = self._mailbox.camera
            if info.index is not None and info.width:
                crops = []
                if self._crop_w < 1.0:
                    crops.append(f"W{int(round(self._crop_w * 100))}%")
                if self._crop_h < 1.0:
                    crops.append(f"H{int(round(self._crop_h * 100))}%")
                crop = f" · crop {' '.join(crops)}" if crops else ""
                self.cam_info_label.setText(
                    f"Camera {info.index}: {info.width}×{info.height} @ "
                    f"{info.fps:.0f} fps · model input 256×256{crop}")
            missing = status.missing_indices if status is not None else ()
            self._preview.update_view(frame, landmarks, mode, stable, badges,
                                      missing, guides=guides)
            self._refresh_settings_status(status)

        if error is not None:
            self._on_vision_fatal(error)

    def _refresh_control_status(self, mode, stable, landmarks,
                                steps=0) -> None:
        if self._game_start_armed and landmarks is not None and mode is not (
                ControllerMode.CALIBRATING):
            status = self._last_cal_status
            if status is None or not status.missing_indices:
                # All key landmarks detected: start once, after calibration.
                self._game_start_armed = False
                if self._output_mode == OUTPUT_WEBSOCKET:
                    self._feedback.append(
                        Badge("▶ GAME", time.monotonic() * 1000.0))
                    if self._bridge is not None:
                        self._bridge.command("start")
                else:
                    self._feedback.append(
                        Badge("▶ ESC", time.monotonic() * 1000.0))
                    if self._emitter is not None:
                        self._emitter.emit((Action.PAUSE_TOGGLE,))
        pose_live = landmarks is not None and (
            time.monotonic() * 1000.0 - self._mailbox.landmark_ts_ms < 1_000
        )
        text = MODE_TEXT.get(mode, "")
        if text and not pose_live:
            text += " · no full body"
        elif not text:
            text = "No full body" if not pose_live else "—"
        if mode is not ControllerMode.CALIBRATING:
            text += f" · head lane {int(stable) + 1}"
        self.mode_label.setText(text)
        if mode is ControllerMode.AUTO_PAUSED:
            self.hint_label.setText("Auto-paused. Jog in place to resume.")
        elif mode is ControllerMode.RESUMING:
            self.hint_label.setText("Resuming — keep jogging…")
        elif mode is ControllerMode.WAITING_FOR_RUN:
            self.hint_label.setText(
                f"Jog in place to activate control — steps {min(steps, 3)}/3…")
        elif mode is ControllerMode.ACTIVE:
            self.hint_label.setText(
                "Controlling — keep jogging; shift lanes left/right, jump or "
                "crouch in place to trigger keys.")
        elif mode is ControllerMode.CALIBRATING:
            if self._output_mode == OUTPUT_WEBSOCKET:
                suffix = "the userscript starts the game after calibration."
            else:
                suffix = "Esc to start the game is sent after calibration."
            self.hint_label.setText(
                "Step back until the whole skeleton is visible — calibration "
                f"runs automatically; {suffix}")

    def _refresh_settings_status(self, status) -> None:
        calibrated = self._pipeline[1].calibrated
        if status is not None and status.phase is CalibrationPhase.COMPLETE:
            self.cal_label.setText("✓ Calibrated — ready to start control.")
            self.missing_label.setText("")
        elif status is not None and status.missing_indices:
            names = "、".join(BODY_PART_LABELS.get(i, f"#{i}")
                              for i in status.missing_indices)
            self.missing_label.setText(
                f"Missing or out of frame: {names} — step back / adjust the camera so the whole body head-to-toe is visible.")
            self.cal_label.setText(self._cal_phase_text(status))
        elif status is not None:
            self.missing_label.setText("")
            self.cal_label.setText(self._cal_phase_text(status))
        elif calibrated:
            self.cal_label.setText("✓ Existing calibration — ready to start control.")
        self.start_button.setEnabled(
            self._output_ready() and self._worker is not None)

    @staticmethod
    def _cal_phase_text(status) -> str:
        if status.phase is CalibrationPhase.FULL_BODY:
            secs = status.full_body_ms / 1000.0
            return f"Full body in frame {secs:.1f} / 1.5 s — keep the whole body inside the safe frame…"
        if status.phase is CalibrationPhase.UPRIGHT:
            secs = status.upright_ms / 1000.0
            return f"Stand upright in the center zone {secs:.1f} / 2.0 s…"
        return "Waiting for picture…"

    def _update_settings_ui(self) -> None:
        output_ready = self._output_ready()
        keyboard = self._output_mode == OUTPUT_KEYBOARD
        self.start_button.setEnabled(
            output_ready and self._worker is not None)
        self.test_button.setEnabled(
            keyboard and self._backend is not None and self._emitter is not None
        )
        self.test_combo.setEnabled(keyboard)
        for combo in self._key_action_rows.values():
            combo.setEnabled(keyboard)
        self.retry_button.setEnabled(self._current_camera is not None)

    # ------------------------------------------------------------------
    # control flow
    # ------------------------------------------------------------------

    def _begin_control(self) -> None:
        with self._lock:
            _, detector, _sm = self._pipeline
            if not self._output_ready():
                return
            detector.clear_transients()
        self._game_start_armed = True
        self._ui.setCurrentWidget(self._control_page)
        self._apply_on_top(True)
        if self._output_mode == OUTPUT_WEBSOCKET:
            # LaunchServices activates the default browser after this window
            # has applied its always-on-top flags, so focus remains in-game.
            self._open_game()
        if self._pipeline[1].calibrated:
            self._start_countdown()
        else:
            # Countdown starts once calibration completes on the control
            # page (checked in _ui_tick).
            self._pending_cal_start = True

    def _start_countdown(self) -> None:
        self._countdown_value = 3
        self.countdown_label.setText("3")
        self._countdown_timer.start(1000)

    def _countdown_step(self) -> None:
        self._countdown_value -= 1
        if self._countdown_value <= 0:
            self._countdown_timer.stop()
            self.countdown_label.setText("")
            with self._lock:
                _, _, sm = self._pipeline
                sm.arm()
            return
        self.countdown_label.setText(str(self._countdown_value))

    def _apply_on_top(self, on: bool) -> None:
        flags = self.windowFlags()
        flags = (flags | Qt.WindowStaysOnTopHint) if on \
            else (flags & ~Qt.WindowStaysOnTopHint)
        if flags != self.windowFlags():
            self.setWindowFlags(flags)
            self.show()

    def _stop_on_top(self) -> None:
        self._apply_on_top(False)

    def _stop_control(self) -> None:
        with self._lock:
            _, detector, sm = self._pipeline
            sm.reset()
            detector.clear_transients()
            self._feedback.clear()
        if self._emitter is not None:
            self._emitter.release_all()
        if self._output_mode == OUTPUT_WEBSOCKET and self._bridge is not None:
            self._bridge.command("pause")
        self._stop_on_top()
        self._ui.setCurrentWidget(self._settings_page)
        self._update_settings_ui()

    def _recalibrate(self) -> None:
        self._stop_control()
        self._reset_pipeline()
        self.cam_error_label.setText("Calibration cleared — please recalibrate.")
        self._ui.setCurrentWidget(self._settings_page)
        self._update_settings_ui()

    def _send_test_key(self) -> None:
        if self._emitter is None:
            return
        key = self.test_combo.currentData()
        name = KEY_NAMES[key]
        self.test_hint.setText(f"Sending {name} in 2 s — focus the target window now…")
        self._emitter.test_token(InputKey(key), delay_ms=2_000)
        QTimer.singleShot(2300, self._reset_test_hint)

    def _reset_test_hint(self) -> None:
        self.test_hint.setText("Sends after a 2 s countdown — focus the target window first.")

    # ------------------------------------------------------------------
    # fatal handling
    # ------------------------------------------------------------------

    def _on_vision_fatal(self, message: str) -> None:
        was_running = False
        with self._lock:
            _, _, sm = self._pipeline
            was_running = sm.mode is not ControllerMode.CALIBRATING
            self._mailbox.error = None
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(4000)
            self._worker = None
        self._reset_pipeline()
        if self._emitter is not None:
            self._emitter.release_all()
        if was_running:
            if self._output_mode == OUTPUT_WEBSOCKET:
                if self._bridge is not None:
                    self._bridge.command("pause")
            elif self._emitter is not None:
                self._emitter.emergency_pause()
        self._stop_on_top()
        self._ui.setCurrentWidget(self._settings_page)
        self.cam_error_label.setText("Error: " + message)
        self._update_settings_ui()

    # ------------------------------------------------------------------
    # window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._countdown_timer.stop()
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(4000)
            self._worker = None
        if self._emitter is not None:
            self._emitter.shutdown()
            self._emitter = None
        if self._bridge is not None:
            self._bridge.command("pause")
            self._bridge.close()
            self._bridge = None
        super().closeEvent(event)


def main() -> int:
    app = QApplication([])
    window = Game2HealthWindow()
    window.show()
    return app.exec()
