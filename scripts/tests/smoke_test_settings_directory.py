"""Real 10,000-file scans, UI heartbeat, cancellation and persistent font settings."""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PyQt5.QtCore import QSettings, QTimer, Qt
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialogButtonBox
from scripts.io.directory_scan import scan_directory
from scripts.ui.main_window import MainWindow
from scripts.ui.settings_dialog import SettingsDialog
from scripts.ui.control_scroll import ControlScrollArea


def main():
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase().families():
        for name in ("segoeui.ttf", "msyh.ttc", "msyhbd.ttc", "simsun.ttc"):
            QFontDatabase.addApplicationFont(str(Path("C:/Windows/Fonts") / name))
    with tempfile.TemporaryDirectory(prefix="edas_large_directory_") as temp:
        root = Path(temp)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root/"settings"))
        folder=root/"files";folder.mkdir()
        for i in range(10000):
            (folder/f"data-{i:05d}.bin").touch()
        (folder/"ignored.txt").touch()
        (folder/"not-a-file.npz").mkdir()
        window=MainWindow()
        window.resize(1280,800);window.show();app.processEvents()
        # Test browsing only; empty synthetic files must not be opened as data.
        window._file_selection_timer.timeout.disconnect()
        window._show_error=lambda message: (_ for _ in ()).throw(AssertionError(message))
        assert window.interface_font_size == 18
        assert window.control_panel.point_range_edit.font().pixelSize() == 18
        axis=window.fk_spectrum_view.plot_widget.getAxis("left")
        assert axis.labelText == "Frequency" and axis.labelUnits == "Hz"
        dialog=SettingsDialog(window,18,window.set_interface_font_size)
        preview = os.environ.get("EDAS_GUI_PREVIEW_DIR")
        if preview:
            Path(preview).mkdir(parents=True,exist_ok=True)
            dialog.show();app.processEvents()
            dialog.grab().save(str(Path(preview)/"settings.png"))
        dialog.font_size_spin.setValue(22)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
        assert window.interface_font_size == 22
        dialog.reject();dialog.deleteLater()
        assert window._parameter_store.interface_font_size() == 22
        for size in (12,18,28,18):
            window.set_interface_font_size(size);app.processEvents()
            assert window.control_panel.point_range_edit.font().pixelSize() == size
            assert axis.labelText == "Frequency" and axis.labelUnits == "Hz"
            for scroll in window.findChildren(ControlScrollArea):
                scroll.refresh_geometry()
                assert scroll.widget().height() < 300
        ticks=[]
        timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda: ticks.append(time.perf_counter()));timer.start()
        main_thread=threading.get_ident()
        def scanner(directory,mode,cancelled):
            assert threading.get_ident()!=main_thread
            # Simulate slow storage without blocking event processing.
            cancelled.wait(.2)
            return scan_directory(directory,mode,cancelled)
        def wait_scan():
            deadline=time.perf_counter()+15
            while window._directory_worker is not None or window._pending_directory_request is not None:
                app.processEvents();time.sleep(.001)
                assert time.perf_counter()<deadline,"Directory scan timeout"
            app.processEvents()
        with patch("scripts.ui.main_window.scan_directory",scanner):
            start=time.perf_counter()
            window.file_browser.path_edit.setText(str(folder))
            window.load_directory(False)
            assert time.perf_counter()-start < .15,"Scan call blocked the UI"
            wait_scan()
            elapsed=time.perf_counter()-start
            assert len(window.file_paths)==10000
            assert window.file_browser.file_list.count()==100
            assert len(ticks)>=10,"UI heartbeat stopped while scanning"
            assert window.file_paths[0].name=="data-00000.bin"
            if preview:
                window.grab().save(str(Path(preview)/"file-list.png"))
            window.file_browser.file_list.setCurrentRow(25)
            selected=window._selected_file_path()
            added=folder/"new.npz";added.touch()
            (folder/"data-00000.bin").unlink()
            window.file_browser.refresh_button.click();wait_scan()
            assert len(window.file_paths)==10000 and added in window.file_paths
            assert selected in window._selected_file_paths(),"Refresh lost selection"
            generation = window._directory_generation
            QApplication.setActiveWindow(window)
            QTest.keyClick(window,Qt.Key_F5)
            QTest.qWait(150)  # QPushButton shortcuts animate before emitting clicked.
            wait_scan()
            assert window._directory_generation > generation,"F5 did not refresh"
            window.file_browser.page_jump_edit.setText("end");window.jump_to_page()
            assert window.current_page==99 and window.file_browser.file_list.count()==100
            # Newest/oldest ordering and request coalescing run in the worker.
            os.utime(added,ns=(2000000000000000000,2000000000000000000))
            window.file_browser.sort_combo.setCurrentIndex(1);wait_scan()
            assert window.file_paths[0]==added
            window.file_browser.sort_combo.setCurrentIndex(2);wait_scan()
            assert window.file_paths[-1]==added
            other=root/"other";other.mkdir();(other/"only.bin").touch()
            window.load_directory(False)
            window.file_browser.path_edit.setText(str(other));window.load_directory(False)
            window.file_browser.refresh_button.click();wait_scan()
            assert window.file_paths==[other/"only.bin"],"Stale scan overwrote newer directory"
            window.file_browser.path_edit.clear();window.load_directory(False);wait_scan()
            assert not window.file_paths
            window.file_browser.path_edit.setText(str(root/"missing"));window.load_directory(False);wait_scan()
            assert not window.file_paths and "失败" in window.file_browser.file_list_status.text()
        timer.stop()
        window.file_browser.path_edit.clear();window._directory_load_timer.stop()
        window.set_interface_font_size(22)
        window.close();app.processEvents()
        restored=MainWindow()
        assert restored.interface_font_size==22
        assert restored.control_panel.point_range_edit.font().pixelSize()==22
        restored.close();app.processEvents()
        print(f"PASS: font default/range/apply/persistence, 10000 files in {elapsed:.3f}s (including .2s slow-storage simulation), {len(ticks)} UI ticks, refresh/selection/pagination/mtime/cancel/stale/error")


if __name__=="__main__":main()
