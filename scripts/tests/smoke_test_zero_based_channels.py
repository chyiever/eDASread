"""Offscreen regression for zero-based channels across all five tabs and exports."""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QLineEdit
from scripts.core.models import FileMetadata, FilterConfig
from scripts.core.settings import ParameterStore
from scripts.processing.view_data import prepare_image_matrix
from scripts.render.traffic_trajectory_view import TrafficTrajectoryView
from scripts.ui.control_panel import ControlPanel, parse_point_range_text
from scripts.ui.main_window import MainWindow


def main():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="edas_zero_based_") as folder:
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, folder)
        store = ParameterStore()
        legacy = {"tab1/point_range": "1~4", "tab1/waveform_point": "1",
                  "tab2/channels": "1,4", "tab3/channel_range": "1-4",
                  "tab4/channel": "4", "tab5/channel_start": "1", "tab5/channel_end": "4"}
        for key, value in legacy.items():
            store._settings.setValue(key, value)
        widgets = {key: QLineEdit() for key in legacy}
        store.restore_widgets(widgets)
        expected = ["0~3", "0", "0,3", "0-3", "3", "0", "3"]
        assert [w.text() for w in widgets.values()] == expected
        ParameterStore().restore_widgets(widgets)
        assert [w.text() for w in widgets.values()] == expected, "Migration must run only once"
        store._settings.clear()
        store._settings.setValue("tab1/point_start", "1")
        store._settings.setValue("tab1/point_end", "1")
        legacy_range = QLineEdit()
        store.restore_widgets({"tab1/point_range": legacy_range})
        assert legacy_range.text() == "0~0"
        store._settings.clear()
        store._settings.setValue("schema/channel_index_base", 0)
        store._settings.sync()
        panel = ControlPanel()
        assert panel.point_range_edit.text() == "0~99"
        assert panel.waveform_point_edit.text() == "0"
        assert parse_point_range_text("~3") == (0, 3)
        win = MainWindow(default_directory=None)
        meta = FileMetadata(Path("synthetic-eDAS-1000Hz-0004pt-20261016T120000.bin"), 1000, 4, 128, .128, 128*4*4)
        t = np.arange(128, dtype=np.float32)
        data = np.stack([np.sin(t / (i+2)) * (i+1) for i in range(4)], axis=1)
        win.current_metadata = meta
        win.current_data = data
        win.current_file_path = meta.file_path
        win.current_file_paths = [meta.file_path]
        win.loaded_file_paths = [meta.file_path]
        win.control_panel.point_range_edit.setText("0~3")
        assert win._read_point_range() == (0, 3)
        assert win._parse_channel_list("0,3") == [0, 3]
        assert win._parse_channel_range("0-3") == (0, 3)
        assert win._read_channel_number(QLineEdit("0"), "通道") == 0
        assert win._read_channel_number(QLineEdit("3"), "通道") == 3
        for a,b in ((-1,0),(0,4)):
            try: prepare_image_matrix(data,a,b)
            except ValueError: pass
            else: raise AssertionError("Invalid point range accepted")
        for text in ("-1", "4"):
            try: win._parse_channel_list(text)
            except ValueError: pass
            else: raise AssertionError("Invalid channel accepted")
        for a,b in ((0,0), (3,3), (0,3), (1,2)):
            np.testing.assert_array_equal(prepare_image_matrix(data,a,b), data[:,a:b+1].T)
            view = win.space_time_view
            view.set_data(meta,data,a,b,a)
            np.testing.assert_array_equal(view.waveform_curve.getData()[1], data[:,a])
            np.testing.assert_array_equal(view.image_item.image, data[:,a:b+1].T)
            view.set_plot_mode("多道波形(Wiggle)")
            assert view._wiggle_stored_channels == list(range(a,b+1))
            for ch,(_,envelope,_) in zip(view._wiggle_stored_channels,view._wiggle_display_cache):
                np.testing.assert_array_equal(envelope,data[:,ch])
            view.set_plot_mode("时空颜色图")
        single = FileMetadata(meta.file_path,1000,1,128,.128,128*4)
        view.set_data(single,data[:,:1],0,0,0)
        np.testing.assert_array_equal(view.waveform_curve.getData()[1],data[:,0])
        assert win._default_point_end_value(metadata=single) == "0"
        view.set_data(meta,data,0,3,0)
        with patch("scripts.ui.main_window.winsound") as sound, patch.object(
            win,"_write_waveform_audio",return_value=(Path(folder)/"mock.wav",8000)
        ) as writer:
            for ch in (0,3):
                win.control_panel.waveform_point_edit.setText(str(ch))
                win.play_current_waveform()
                np.testing.assert_array_equal(writer.call_args.args[0],data[:,ch])
            assert sound.PlaySound.call_count == 2
        view.set_y_axis_mode(5,100,"距离","m")
        assert view.distance_axis._to_distance(0) == 100
        assert view.distance_axis._to_distance(3) == 115
        win.channel_analysis_view.set_data(meta,data,[0,3],psd_nperseg=32)
        for trace,ch in zip(win.channel_analysis_view._traces,[0,3]):
            np.testing.assert_allclose(trace,data[:,ch]-np.median(data[:,ch]),atol=1e-6)
        options = dict(taper_fraction=(0,0),remove_background=False,nfft_channels=None,nfft_samples=None,frequency_range=(None,None),wavenumber_range=(None,None))
        with patch("scripts.ui.main_window.compute_fk_spectrum_db",return_value={}) as fn:
            win._compute_fk_spectrum_for_view(data,1000,0,3,5,options,(-20,0),None)
            np.testing.assert_array_equal(fn.call_args.args[0],data)
        with patch("scripts.ui.main_window.compute_time_frequency_map",return_value={}) as fn:
            for ch in (0,3):
                win._compute_time_frequency_for_view(data,meta,ch,FilterConfig(),dict(window_seconds=.01,overlap_percent=50,mode="PSD"),None)
                np.testing.assert_array_equal(fn.call_args.args[0],data[:,ch])
        tab5 = win.traffic_trajectory_view
        tab5.channel_start_edit.setText("0")
        tab5.channel_end_edit.setText("3")
        tab5.display_mode_combo.setCurrentIndex(tab5.display_mode_combo.findData("phase"))
        win.generate_traffic_energy_image()
        np.testing.assert_array_equal(tab5.image_data,data)
        assert tab5.point_start == 0
        tab5.trajectories = [[(0.,0.),(.1,3.)]]
        tab5.trajectory_visible = [True]
        doc = tab5.annotation_document()
        assert doc["coordinate_system"]["y"] == "0_based_channel"
        assert TrafficTrajectoryView.decode_annotation_trajectories(doc) == tab5.trajectories
        legacy_doc = {"format":"edasread-traffic-trajectories-v1", "trajectories":[{"points":[{"time_s":0,"channel":1},{"time_s":.1,"channel":4}]}]}
        assert TrafficTrajectoryView.decode_annotation_trajectories(legacy_doc) == tab5.trajectories
        assert win._default_point_end_value(metadata=meta) == "3"
        win.control_panel.point_range_edit.setText("0~99")
        win._clamp_point_range_to_points(4)
        assert win._read_point_range() == (0,3)
        view.set_data(meta,data,0,3,0)
        for a,b in ((0,3),(0,0),(3,3)):
            view.set_y_range(a,b+1)
            result = win._export_current_view_data(Path(folder),True,True)
            assert result["point_count"] == b-a+1
            np.testing.assert_allclose(np.loadtxt(result["output_path"],ndmin=2),data[:,a:b+1],atol=1e-6)
        raw = np.arange(128*4,dtype=np.int32).reshape(128,4)
        view.set_y_range(0,4)
        with patch("scripts.io.view_export.read_bin_memmap",return_value=raw):
            result = win._export_current_view_data(Path(folder),False,False)
            np.testing.assert_array_equal(np.fromfile(result["output_path"],dtype=np.int32).reshape(128,4),raw)
        view.set_y_range(0,2)
        win._on_vertical_scroll(1000)
        np.testing.assert_allclose(view.current_ranges()[1],(2,4))
        win._on_vertical_scroll(0)
        np.testing.assert_allclose(view.current_ranges()[1],(0,2))
        win.close()
        panel.close()
    print("PASS: zero-based input, first/last channel, all tabs, distance, export, scrollbars, settings and annotation migration")

if __name__ == "__main__":
    main()
