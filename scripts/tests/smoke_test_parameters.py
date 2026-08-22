"""Off-screen smoke test: parameter persistence across sessions and files.

Run from the project root:
    python scripts\\tests\\smoke_test_parameters.py

Verifies:
1. Startup restores the most recently used Tab1/Tab2/Tab3 parameters.
2. Switching data files never resets those parameters; Point Start/End are
   only clamped when they exceed the newly loaded file's channel count.
3. Closing the window persists the latest values (closeEvent flush).
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from scripts.core.models import FileMetadata  # noqa: E402
from scripts.ui.main_window import MainWindow  # noqa: E402


def make_metadata(points: int) -> FileMetadata:
    return FileMetadata(
        file_path=Path(f"fake_{points}pts.bin"),
        sample_rate_hz=1000.0,
        points=points,
        frame_count=1000,
        duration_s=1.0,
        file_size_bytes=1024,
        start_time=datetime.now(),
    )


def main() -> int:
    temp_dir = Path(tempfile.mkdtemp(prefix="edas_smoke_settings_"))
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(temp_dir))

    def make_store() -> QSettings:
        return QSettings(
            QSettings.IniFormat,
            QSettings.UserScope,
            "eDASread",
            "eDASread",
        )

    app = QApplication([])
    failures: list[str] = []

    def check(label: str, actual, expected) -> None:
        passed = actual == expected
        print(("PASS" if passed else "FAIL"), "-", label, "| actual:", repr(actual))
        if not passed:
            failures.append(label)

    def check_true(label: str, condition: bool) -> None:
        check(label, bool(condition), True)

    # --- Seed settings, simulating a previous working session -------------
    seed = {
        "tab1/point_start": "3",
        "tab1/point_end": "120",
        "tab1/vmin": "-0.9",
        "tab1/vmax": "0.9",
        "tab1/filter_type": "lowpass",
        "tab1/save_filtered": True,
        "tab2/channels": "7,8",
        "tab2/psd_overlap": "60",
        "tab3/channel_range": "4-30",
        "tab3/remove_background": False,
    }
    writer = make_store()
    for key, value in seed.items():
        writer.setValue(key, value)
    writer.sync()

    # --- Session 1: startup restores the most recent parameters ----------
    window = MainWindow(default_directory=None)
    panel = window.control_panel
    tab2 = window.channel_analysis_view
    tab3 = window.fk_spectrum_view

    check("restore tab1/point_start", panel.point_start_edit.text(), "3")
    check("restore tab1/point_end", panel.point_end_edit.text(), "120")
    check("restore tab1/vmin", panel.vmin_edit.text(), "-0.9")
    check("restore tab1/filter_type", panel.filter_type_combo.currentText(), "lowpass")
    check_true("restore tab1/save_filtered checked", panel.save_filtered_checkbox.isChecked())
    check("restore tab2/channels", tab2.channels_edit.text(), "7,8")
    check("restore tab2/psd_overlap", tab2.psd_overlap_edit.text(), "60")
    check("restore tab3/channel_range", tab3.channel_range_edit.text(), "4-30")
    check_true(
        "restore tab3/remove_background unchecked",
        not tab3.remove_bg_checkbox.isChecked(),
    )
    check_true("restored point_end marks manual override", window._point_end_manual_override)

    # --- Switching files keeps parameters; only out-of-range clamps ------
    window._apply_auto_point_limit(metadata=make_metadata(5000))
    check("file switch keeps point_start", panel.point_start_edit.text(), "3")
    check("file switch keeps point_end", panel.point_end_edit.text(), "120")
    check("file switch keeps tab2/channels", tab2.channels_edit.text(), "7,8")
    check("file switch keeps tab3/channel_range", tab3.channel_range_edit.text(), "4-30")

    window._apply_auto_point_limit(metadata=make_metadata(40))
    check("small file clamps point_start only if needed", panel.point_start_edit.text(), "3")
    check("small file clamps point_end to 40", panel.point_end_edit.text(), "40")
    check("small file still keeps tab2/channels", tab2.channels_edit.text(), "7,8")
    check("small file still keeps tab3/channel_range", tab3.channel_range_edit.text(), "4-30")

    # --- closeEvent flushes the latest values -----------------------------
    panel.vmin_edit.setText("-1.25")
    panel.colormap_combo.setCurrentText("hsv")
    tab3.spacing_edit.setText("7.5")
    window.show()
    app.processEvents()
    window.close()
    app.processEvents()

    reader = make_store()
    check("closeEvent saved tab1/vmin", str(reader.value("tab1/vmin")), "-1.25")
    check("closeEvent saved tab1/colormap", str(reader.value("tab1/colormap")), "hsv")
    check("closeEvent saved tab3/spacing", str(reader.value("tab3/spacing")), "7.5")
    check_true(
        "closeEvent saved tab1/save_filtered",
        str(reader.value("tab1/save_filtered")).strip().lower() in {"1", "true"},
    )

    # --- Session 2: fresh window loads the flushed values ------------------
    window2 = MainWindow(default_directory=None)
    check("session2 restores tab1/vmin", window2.control_panel.vmin_edit.text(), "-1.25")
    check("session2 restores colormap", window2.control_panel.colormap_combo.currentText(), "hsv")
    check("session2 restores tab3/spacing", window2.fk_spectrum_view.spacing_edit.text(), "7.5")
    check("session2 restores tab2/channels", window2.channel_analysis_view.channels_edit.text(), "7,8")
    check_true("session2 marks point_end override", window2._point_end_manual_override)
    window2.close()
    app.processEvents()

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("All parameter-persistence smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
