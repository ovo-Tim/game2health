"""Local WebSocket bridge between the controller and the game userscript."""

from __future__ import annotations

import json
from collections.abc import Sequence

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QHostAddress
from PySide6.QtWebSockets import QWebSocketServer

from .types import Action

BRIDGE_PORT = 8765
BRIDGE_URL = f"ws://127.0.0.1:{BRIDGE_PORT}"
GAME_URL = (
    "https://www.gamezhero.com/get-game-code/"
    "ed058f6930a27493b8a5aabc8b0d4f70"
)
_ALLOWED_ORIGINS = {
    "https://files.gamezhero.com",
    "https://www.gamezhero.com",
}
_COMMANDS = {"start", "pause", "resume"}


class GameBridge(QObject):
    """Single-client localhost server with one retained lifecycle command.

    Motion actions are transient and are dropped while the userscript is
    disconnected.  The latest lifecycle command is retained and delivered to
    the next accepted client so game loading cannot lose start/pause intent.
    """

    status_changed = Signal(str)
    client_changed = Signal(bool)
    game_state_changed = Signal(str)

    def __init__(self, port: int = BRIDGE_PORT, parent: QObject | None = None):
        super().__init__(parent)
        self._server = QWebSocketServer(
            "Game2Health game bridge", QWebSocketServer.NonSecureMode, self
        )
        self._client = None
        self._pending_command: str | None = None
        self._status = ""
        self._game_state = "disconnected"
        self._server.newConnection.connect(self._accept_client)
        self._listening = self._server.listen(QHostAddress.LocalHost, port)
        if self._listening:
            actual_port = self._server.serverPort()
            self._set_status(
                f"Waiting for userscript on ws://127.0.0.1:{actual_port}"
            )
        else:
            self._set_status(
                f"WebSocket port {port} unavailable: {self._server.errorString()}"
            )

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def port(self) -> int:
        return self._server.serverPort()

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> str:
        return self._status

    @property
    def game_state(self) -> str:
        return self._game_state

    def command(self, command: str) -> None:
        if command not in _COMMANDS:
            raise ValueError(f"Unknown game command: {command}")
        if not self._send({"type": "command", "command": command}):
            self._pending_command = command

    def emit(self, actions: Sequence[Action]) -> None:
        values = [action.value for action in actions
                  if action is not Action.PAUSE_TOGGLE]
        if values:
            self._send({"type": "actions", "actions": values})

    def close(self) -> None:
        self._pending_command = None
        if self._client is not None:
            self._client.close()
            self._client.deleteLater()
            self._client = None
        self._server.close()
        self._listening = False
        self._game_state = "disconnected"

    def _accept_client(self) -> None:
        socket = self._server.nextPendingConnection()
        origin = socket.origin().rstrip("/")
        if origin not in _ALLOWED_ORIGINS:
            self._set_status(f"Rejected WebSocket origin: {origin or 'none'}")
            socket.close()
            socket.deleteLater()
            return
        if self._client is not None:
            self._client.close()
            self._client.deleteLater()
        self._client = socket
        socket.textMessageReceived.connect(self._on_message)
        socket.disconnected.connect(self._client_disconnected)
        self._set_status("Userscript connected; waiting for Unity")
        self._send({"type": "hello", "version": 1})
        if self._pending_command is not None:
            command = self._pending_command
            self._pending_command = None
            self.command(command)
        self.client_changed.emit(True)

    def _client_disconnected(self) -> None:
        socket = self.sender()
        if socket is not self._client:
            return
        self._client.deleteLater()
        self._client = None
        self._game_state = "disconnected"
        self._set_status("Userscript disconnected; reconnecting when game opens")
        self.client_changed.emit(False)
        self.game_state_changed.emit(self._game_state)

    def _on_message(self, text: str) -> None:
        try:
            message = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "ready" and message.get("version") == 1:
            self._set_status("Userscript and Unity ready")
            return
        if kind != "state":
            return
        state = message.get("state")
        if state not in {"loading", "ready", "running", "paused", "gameover"}:
            return
        self._game_state = state
        self._set_status(f"Userscript connected · game {state}")
        self.game_state_changed.emit(state)

    def _send(self, message: dict) -> bool:
        if self._client is None:
            return False
        payload = json.dumps(message, separators=(",", ":"))
        return self._client.sendTextMessage(payload) == len(payload)

    def _set_status(self, status: str) -> None:
        self._status = status
        self.status_changed.emit(status)
