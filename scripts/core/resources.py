"""Helpers for resolving bundled asset paths."""

from __future__ import annotations

import sys
from pathlib import Path


def app_base_dir() -> Path:
    """Return the application base directory for source and PyInstaller runs."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def asset_path(*parts: str) -> Path:
    """Return an absolute path inside the packaged assets directory."""
    return app_base_dir().joinpath("scripts", "assets", *parts)
