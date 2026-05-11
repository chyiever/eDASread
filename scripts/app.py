"""Application entry point for the eDASread desktop tool."""

from __future__ import annotations

import sys
from pathlib import Path

# Support both `python -m scripts.app` from the project root and
# `python app.py` from inside the `scripts` directory.
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from scripts.core.resources import asset_path
from scripts.ui.main_window import MainWindow


def _resolve_logo_path() -> Path | None:
    candidates = [
        asset_path("eDASread_logo.png"),
        CURRENT_FILE.parent / "eDASread_logo.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    """Launch the desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("eDASread")
    app.setOrganizationName("eDASread")

    logo_path = _resolve_logo_path()
    if logo_path is not None:
        icon = QIcon(str(logo_path))
        if not icon.isNull():
            app.setWindowIcon(icon)

    window = MainWindow(default_directory=PROJECT_ROOT / "data")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
