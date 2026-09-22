# Game2Health

Desktop motion game controller: track your full body with a regular webcam,
turn running in place, side-steps, jumps and crouches into real keyboard
events — and control browser games without touching the keyboard.

Built with MediaPipe Pose Landmarker + PySide6 (Qt). Runs on macOS, Windows
and Linux (X11 and Wayland).

## How it works

- A camera worker streams frames into MediaPipe's pose landmarker (model
  bundled in the package, no network access at runtime).
- A motion state machine maps body movements to actions:
  - Move left / right — step fully into the left / third of the frame.
  - Jump — hop straight up with both feet off the ground.
  - Crouch — squat down with both knees bent.
  - Pause — stop running; start running again to resume.
- Default key mapping: `LEFT=←`, `RIGHT=→`, `JUMP=↑`, `CROUCH=↓`,
  `PAUSE=Esc`. Re-mappable to `←/→/↑/↓/Esc/Space/W/A/S/D/P` in the settings
  page.
- A localhost WebSocket bridge (`ws://127.0.0.1:8765`) lets a browser
  userscript hook into the game lifecycle (auto-start / pause / restart).

## Requirements

- Python 3.12 (strictly `>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) as the package manager
- A webcam, positioned so your whole body (head to toe) is in frame
- OS permissions:
  - **macOS**: grant camera access, and Accessibility permission for sending
    keystrokes (System Settings → Privacy & Security → Accessibility).
  - **Linux (Wayland)**: `uinput` access for the virtual keyboard backend.
  - **Linux (X11)** / **Windows**: works out of the box.

## Run

```sh
uv run game2health
```

First run resolves dependencies from `uv.lock` and builds the package
automatically. The bundled pose model ships inside the wheel, so no extra
download is needed.

## Usage

1. Launch `game2health`. The settings page opens.
2. Pick a camera from the dropdown (indices `0–4` are probed automatically).
3. Follow the calibration prompts: step back until your whole body is inside
   the safe frame, stand upright in the center lane for ~2 seconds.
4. Optionally adjust the key mapping and output backend, and send a test key
   to verify focus handling.
5. Click **Start control**, switch focus to your game within the 3-second
   countdown, then jog in place to begin.

## Browser game bridge (Subway Surfers)

1. Install [Tampermonkey](https://www.tampermonkey.net/) (or a similar
   userscript manager) in your browser.
2. Install `userscripts/game2health-subway.user.js`.
3. Open the game at
   https://www.gamezhero.com/get-game-code/ed058f6930a27493b8a5aabc8b0d4f70
   — the userscript connects to the local bridge automatically (the app must
   be running) and handles game start / pause / restart for you.

Without the userscript the controller still works in any game that accepts
the mapped keys.

## Linux (Wayland) uinput setup

Wayland clients only receive synthetic input reliably via a virtual
keyboard. If the app reports a uinput error, run once as a regular user with
sudo, then click **Re-detect** (re-login if needed):

```sh
sudo install -Dm644 packaging/linux/game2health-uinput.conf \
    /etc/modules-load.d/game2health-uinput.conf
sudo install -Dm644 packaging/linux/70-game2health-uinput.rules \
    /etc/udev/rules.d/70-game2health-uinput.rules
sudo modprobe uinput
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=misc
```

## Development

```sh
uv sync                      # set up the venv
uv run pytest                # run the test suite
uv add <package>             # add a dependency (updates uv.lock)
```

## License

See [LICENSE](LICENSE).
