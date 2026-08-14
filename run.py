"""Project entry point for the eDASread desktop tool.

Run from the project root:  python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())