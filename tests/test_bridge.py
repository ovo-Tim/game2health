"""Observable contract for the localhost game-userscript bridge."""

from __future__ import annotations

import json
import time

from PySide6.QtCore import QCoreApplication, QUrl
from PySide6.QtNetwork import QNetworkRequest
from PySide6.QtWebSockets import QWebSocket

from game2health.bridge import GameBridge
from game2health.types import Action


def _wait_until(predicate, timeout: float = 2.0) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    assert predicate()


def test_bridge_delivers_retained_command_actions_and_game_state():
    QCoreApplication.instance() or QCoreApplication([])
    bridge = GameBridge(port=0)
    assert bridge.listening
    bridge.command("start")  # retained until the userscript connects

    client = QWebSocket()
    received = []
    client.textMessageReceived.connect(
        lambda text: received.append(json.loads(text))
    )
    request = QNetworkRequest(QUrl(f"ws://127.0.0.1:{bridge.port}"))
    request.setRawHeader(b"Origin", b"https://files.gamezhero.com")
    client.open(request)

    _wait_until(lambda: bridge.connected and len(received) >= 2)
    assert received[:2] == [
        {"type": "hello", "version": 1},
        {"type": "command", "command": "start"},
    ]

    bridge.emit((Action.MOVE_LEFT, Action.PAUSE_TOGGLE, Action.JUMP))
    _wait_until(lambda: len(received) == 3)
    assert received[-1] == {
        "type": "actions",
        "actions": ["move_left", "jump"],
    }

    client.sendTextMessage(json.dumps({"type": "ready", "version": 1}))
    client.sendTextMessage(json.dumps({"type": "state", "state": "running"}))
    _wait_until(lambda: bridge.game_state == "running")
    assert bridge.status == "Userscript connected · game running"

    client.close()
    bridge.close()
