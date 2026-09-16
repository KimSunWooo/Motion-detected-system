#!/usr/bin/env python3
"""Compatibility entrypoint. Implementation lives in src/helmet_action."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helmet_action.compat import *  # noqa: F403
from helmet_action.compat import main

if __name__ == "__main__":
    raise SystemExit(main())
