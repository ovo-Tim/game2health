"""Entry point: `uv run game2health`."""

from __future__ import annotations

import sys


def main() -> int:
    from .app import main as app_main

    return app_main()


if __name__ == "__main__":
    sys.exit(main())
