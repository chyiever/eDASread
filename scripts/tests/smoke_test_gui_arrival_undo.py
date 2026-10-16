"""Qt regression: compact themed GUI, full trace spacing, and Esc key repeat."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PyQt5.QtCore import QEvent, QSettings, Qt
from PyQt5.QtGui import QKeyEvent, QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLineEdit

from scripts.core.models import FileMetadata
from scripts.ui.main_window import MainWindow


def main():
    app = QApplication.instance() or QApplication([])
    # The Windows offscreen backend exposes no system fonts by default. Load
    # installed fonts for realistic text measurement and visual QA only.
    if not QFontDatabase().families():
        for name in ("segoeui.ttf", "msyh.ttc", "msyhbd.ttc", "simsun.ttc"):
            font = Path("C:/Windows/Fonts") / name
            if font.exists():
                QFontDatabase.addApplicationFont(str(font))
    with tempfile.TemporaryDirectory(prefix="edas_gui_undo_") as folder:
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, folder)
        window = MainWindow(default_directory=None)
        assert window.windowTitle() == "eDAS数据处理软件（eDASpy ）"
        assert window.title_label.text() == window.windowTitle()
        assert window.title_label.alignment() == Qt.AlignCenter
        assert window.plot_tabs.count() == 5
        assert all(not window.plot_tabs.tabIcon(i).isNull() for i in range(5))
        assert len({window.plot_tabs.tabIcon(i).cacheKey() for i in range(5)}) == 5
        frames, channels = 96, 750
        data = np.sin(np.arange(frames,dtype=np.float32)[:,None]/5 + np.arange(channels)[None,:]).astype(np.float32)
        metadata = FileMetadata(Path("gui-test.bin"),1000,channels,frames,frames/1000,data.nbytes)
        window.current_metadata = metadata
        window.current_data = data
        window.current_file_path = metadata.file_path
        window.current_file_paths = [metadata.file_path]
        window.loaded_file_paths = [metadata.file_path]
        panel = window.control_panel
        panel.point_range_edit.setText("0~749")
        panel.display_mode_combo.setCurrentText("多道波形(Wiggle)")
        panel.spacing_combo.setCurrentText("地震道")
        panel.spacing_edit.setText("1")
        started=time.perf_counter()
        window._render_current_data()
        view=window.space_time_view
        view._wiggle_timer.stop();view._refresh_wiggle()
        elapsed=time.perf_counter()-started
        assert view._wiggle_stored_channels == list(range(750)), "Spacing 1 must not silently become spacing 5"
        assert len(view._wiggle_items)==750
        panel.spacing_edit.setText("2")
        window._apply_wiggle_settings();view._wiggle_timer.stop();view._refresh_wiggle()
        assert view._wiggle_stored_channels == list(range(0,750,2))
        # Keep screenshot/interaction light after the high-channel-count check.
        panel.point_range_edit.setText("0~15")
        panel.spacing_edit.setText("1")
        window._render_current_data();view._wiggle_timer.stop();view._refresh_wiggle()
        panel_heights = []
        for width,height in ((1280,800),(1500,900)):
            window.resize(width,height);window.show();app.processEvents()
            header = window.title_label.parentWidget()
            assert abs(window.title_label.geometry().center().x() - header.rect().center().x()) <= 1
            assert panel.point_range_edit.font().pixelSize() == 18
            assert window.title_label.font().pixelSize() >= 24
            combo = panel.display_mode_combo
            assert combo.fontMetrics().horizontalAdvance(combo.currentText()) + 28 <= combo.width()
            assert panel.height() <= 165, (width,panel.height())
            assert panel.export_button.isVisible() and panel.export_format_combo.isVisible()
            assert panel.export_button.mapTo(panel,panel.export_button.rect().topRight()).x() < panel.width()
            assert view.height() >= height*.5, (height,view.height())
            panel_heights.append((width,panel.height()))
            for tab in range(5):
                window.plot_tabs.setCurrentIndex(tab);app.processEvents()
                for edit in window.plot_tabs.currentWidget().findChildren(QLineEdit):
                    if edit.isVisible():
                        parent=edit.parentWidget()
                        assert edit.geometry().right() < parent.width(), (width,tab,edit.text(),parent.width(),edit.geometry())
            window.plot_tabs.setCurrentIndex(0);app.processEvents()
        view.plot_widget.setFocus();app.processEvents()
        for channel in (2,4,6):view.mark_arrival(channel,.03)
        QTest.keyClick(view.plot_widget,Qt.Key_Escape)
        assert list(view.arrivals)==[2,4]
        view.mark_arrival(2,.04)
        QTest.keyClick(view.plot_widget,Qt.Key_Escape)
        assert list(view.arrivals)==[4], "Esc deletes the most recently edited channel, including a replacement"
        view.mark_arrival(6,.03);view.mark_arrival(8,.03)
        for _ in range(5):
            QApplication.sendEvent(view.plot_widget,QKeyEvent(QEvent.KeyPress,Qt.Key_Escape,Qt.NoModifier,"",True,1))
        assert not view.arrivals and not view._arrival_items
        view.mark_arrival_segment((8,.02),(5,.04))
        assert list(view.arrivals)==[8,7,6,5]
        QTest.keyClick(view.plot_widget,Qt.Key_Escape)
        assert list(view.arrivals)==[8,7,6]
        # Esc should dismiss a settings popover rather than erase a mark.
        menu=panel.geometry_menu_button.menu()
        menu.popup(panel.geometry_menu_button.mapToGlobal(panel.geometry_menu_button.rect().bottomLeft()))
        app.processEvents()
        QTest.keyClick(menu,Qt.Key_Escape)
        assert list(view.arrivals)==[8,7,6]
        app.processEvents()
        # Optional output used by the visual review, never a production data file.
        if os.environ.get("EDAS_GUI_PREVIEW_DIR"):
            target=Path(os.environ["EDAS_GUI_PREVIEW_DIR"]);target.mkdir(parents=True,exist_ok=True)
            for i in range(5):
                window.plot_tabs.setCurrentIndex(i);app.processEvents()
                window.grab().save(str(target/f"tab{i+1}.png"))
        window.close();app.processEvents();window.deleteLater();app.processEvents()
        print(f"PASS: 5 tab icons, compact 1280/1500px layout, 750 traces at step 1 ({elapsed:.3f}s), panel heights {panel_heights}, step 2, Esc/repeat/recency/drag order and popup isolation")


if __name__=="__main__":
    main()
