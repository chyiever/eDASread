"""Build a single-file Windows executable for eDASpy with PyInstaller.

Run from the project root:  python build_exe.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
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


def _find_zstd_dll() -> Path | None:
    """Find zstd.dll in the active Python/Conda runtime."""
    candidates = [
        Path(sys.prefix) / "Library" / "bin" / "zstd.dll",
        Path(sys.prefix) / "zstd.dll",
        Path(sys.executable).resolve().parent / "zstd.dll",
        Path(sys.executable).resolve().parent / "Library" / "bin" / "zstd.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    found = shutil.which("zstd.dll")
    return Path(found).resolve() if found else None


def _unique_output_name(dist_dir: Path) -> str:
    """Return a dated executable stem that will not overwrite an existing exe."""
    date_text = datetime.now().strftime("%Y%m%d")
    base_name = f"eDASpy-{date_text}"
    candidate = base_name
    counter = 2
    while (dist_dir / f"{candidate}.exe").exists():
        candidate = f"{base_name}-{counter}"
        counter += 1
    return candidate


def _cleanup_build_files(build_dir: Path, spec_path: Path, output_name: str) -> None:
    """Remove build artifacts (build dir and spec file) after a successful build."""
    work_path = build_dir / output_name
    for target in (work_path, spec_path, build_dir):
        try:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"Warning: failed to clean up {target}: {exc}")


def build() -> int:
    _ensure_pyinstaller()

    project_root = Path(__file__).resolve().parent
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"

    logo_png = project_root / "assets" / "eDASread_logo.png"
    logo_ico = project_root / "assets" / "eDASread_logo.ico"
    icon_path = _build_icon(logo_png, logo_ico)

    dist_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    output_name = _unique_output_name(dist_dir)
    work_path = build_dir / output_name
    spec_path = build_dir / "specs"
    spec_path.mkdir(parents=True, exist_ok=True)

    sep = ";" if sys.platform.startswith("win") else ":"
    data_arg = f"{logo_png}{sep}assets"
    zstd_dll = _find_zstd_dll()
    zstd_binary_args = ["--add-binary", f"{zstd_dll}{sep}."] if zstd_dll is not None else []

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        output_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        "--icon",
        str(icon_path),
        "--add-data",
        data_arg,
        *zstd_binary_args,
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
        str(project_root / "run.py"),
    ]

    print("Running:", " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode == 0:
        print(f"Build complete: {dist_dir / f'{output_name}.exe'}")
        _cleanup_build_files(build_dir, spec_path, output_name)
        print("Build artifacts cleaned up (build/ and spec files removed).")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(build())