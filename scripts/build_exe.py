"""Build a single-file Windows executable for eDASread with PyInstaller."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except Exception as exc:  # pragma: no cover - setup guard
        raise RuntimeError(
            "PyInstaller is not installed. Install it first: pip install pyinstaller"
        ) from exc


def _build_icon(png_path: Path, ico_path: Path) -> Path:
    if not png_path.exists():
        raise FileNotFoundError(f"Logo not found: {png_path}")

    try:
        from PyQt5.QtGui import QImage

        image = QImage(str(png_path))
        if image.isNull() or not image.save(str(ico_path), "ICO"):
            raise RuntimeError("Qt ICO conversion failed")
        return ico_path
    except Exception:
        # PyInstaller on Windows prefers .ico. If conversion fails, fallback to png.
        return png_path


def build() -> int:
    _ensure_pyinstaller()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    spec_file = project_root / "eDASread.spec"

    logo_png = script_dir / "assets" / "eDASread_logo.png"
    logo_ico = script_dir / "assets" / "eDASread_logo.ico"
    icon_path = _build_icon(logo_png, logo_ico)

    for path in [dist_dir, build_dir]:
        if path.exists():
            shutil.rmtree(path)
    if spec_file.exists():
        spec_file.unlink()

    sep = ";" if sys.platform.startswith("win") else ":"
    data_arg = f"{logo_png}{sep}scripts/assets"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "eDASread",
        "--icon",
        str(icon_path),
        "--add-data",
        data_arg,
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "IPython",
        "--exclude-module",
        "jupyter",
        "--exclude-module",
        "notebook",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "PIL",
        str(script_dir / "app.py"),
    ]

    print("Running:", " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode == 0:
        print(f"Build complete: {dist_dir / 'eDASread.exe'}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(build())
