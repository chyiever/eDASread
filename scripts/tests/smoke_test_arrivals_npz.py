"""Integration regression: native arrival gestures and lossless NPZ crop/stride round trips."""
from __future__ import annotations
import json
import os
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PyQt5.QtCore import QEvent, QPointF, Qt, QSettings
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from scripts.core.models import FilterConfig
from scripts.io.bin_reader import inspect_bin_file, read_bin_memmap, read_phase_radians, list_bin_files
from scripts.io.npz_codec import read_edas_npz, write_edas_npz
from scripts.io.view_export import export_view
from scripts.ui.main_window import MainWindow


def main():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="edas_arrivals_npz_") as temp:
        root = Path(temp)
        QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(root/"settings"))
        paths = [root/f"{i}-eDAS-1000Hz-0008pt-20261016T12000{i}.bin" for i in range(2)]
        raw = (np.arange(202*8,dtype=np.int32).reshape(202,8) + 2**25)
        raw[:101].tofile(paths[0]); raw[101:].tofile(paths[1])
        win = MainWindow(default_directory=None)
        group,sources = win._inspect_file_group(paths)
        # Simulate filtering: exported NPZ must still equal the raw int32 source.
        payload = win._load_file_data(group,FilterConfig(),sources,win._file_load_generation)
        payload['data'] = payload['data']*0 + 123.0
        win._on_data_loaded(payload)
        view = win.space_time_view
        win.control_panel.point_range_edit.setText("0~7")
        win.control_panel.display_mode_combo.setCurrentText("多道波形(Wiggle)")
        win._render_current_data()
        win.resize(1400,1000)
        win.show(); app.processEvents()
        view._wiggle_timer.stop(); view._refresh_wiggle()
        def mouse(kind,t,ch,button=Qt.NoButton,buttons=Qt.NoButton,modifiers=Qt.ControlModifier):
            app.processEvents()
            # Hit inside the channel band, away from the axis boundary pixel.
            point = view.plot_widget.mapFromScene(view.plot_widget.getViewBox().mapViewToScene(QPointF(t,ch + .05)))
            event = QMouseEvent(kind,QPointF(point),button,buttons,modifiers)
            QApplication.sendEvent(view.plot_widget.viewport(),event)
            app.processEvents()
        # Keep a Ctrl drag active when the pointer briefly leaves the plot.
        mouse(QEvent.MouseButtonPress,.04,2,Qt.LeftButton,Qt.LeftButton)
        mouse(QEvent.MouseMove,.06,12,buttons=Qt.LeftButton)
        mouse(QEvent.MouseMove,.08,4,buttons=Qt.LeftButton)
        mouse(QEvent.MouseButtonRelease,.08,4,Qt.LeftButton)
        assert set(view.arrivals)=={2,3,4}, "Leaving/re-entering the plot lost the arrival gesture"
        view.clear_arrivals()
        mouse(QEvent.MouseButtonPress,.04,2,Qt.LeftButton,Qt.LeftButton)
        mouse(QEvent.MouseMove,.08,6,buttons=Qt.LeftButton)
        mouse(QEvent.MouseButtonRelease,.08,6,Qt.LeftButton)
        assert set(view.arrivals) == {2,3,4,5,6},view.arrivals
        assert all(item.isVisible() for item in view._arrival_items.values())
        assert view.arrivals[2] < view.arrivals[4] < view.arrivals[6]
        before = view.arrivals[4]
        mouse(QEvent.MouseMove,.06,4)
        QTest.keyClick(view.plot_widget,Qt.Key_Right,Qt.ControlModifier)
        assert np.isclose(view.arrivals[4],before+.001)
        QTest.keyClick(view.plot_widget,Qt.Key_Left,Qt.ControlModifier)
        assert np.isclose(view.arrivals[4],before)
        win.control_panel.display_mode_combo.setCurrentText("时空颜色图")
        assert all(not i.isVisible() for i in view._arrival_items.values())
        win.control_panel.display_mode_combo.setCurrentText("多道波形(Wiggle)")
        assert all(i.isVisible() for i in view._arrival_items.values())
        # A release at another channel must fill skipped move events as well.
        mouse(QEvent.MouseButtonPress,.03,0,Qt.LeftButton,Qt.LeftButton)
        mouse(QEvent.MouseButtonRelease,.05,1,Qt.LeftButton)
        assert {0,1}.issubset(view.arrivals), (view.arrivals,view.plot_widget.getViewBox().viewRange(),view._arrival_drag,view._hover_channel)
        saved = dict(view.arrivals)
        erase = Qt.ControlModifier | Qt.ShiftModifier
        # Deletion targets the channel irrespective of arrival time at the cursor.
        mouse(QEvent.MouseButtonPress,.15,4,Qt.LeftButton,Qt.LeftButton,erase)
        mouse(QEvent.MouseButtonRelease,.15,4,Qt.LeftButton,modifiers=erase)
        assert view.arrivals == {ch:t for ch,t in saved.items() if ch != 4}
        assert 4 not in view._arrival_items
        # Empty channels are harmless; reverse drag fills skipped mouse events.
        mouse(QEvent.MouseButtonPress,.15,6,Qt.LeftButton,Qt.LeftButton,erase)
        mouse(QEvent.MouseMove,.15,12,buttons=Qt.LeftButton,modifiers=erase)
        mouse(QEvent.MouseMove,.15,2,buttons=Qt.LeftButton,modifiers=erase)
        mouse(QEvent.MouseButtonRelease,.15,2,Qt.LeftButton,modifiers=erase)
        assert set(view.arrivals) == {0,1}
        assert set(view._arrival_items) == {0,1}
        # Release-only travel and missing marks must also work without creation.
        mouse(QEvent.MouseButtonPress,.15,0,Qt.LeftButton,Qt.LeftButton,erase)
        mouse(QEvent.MouseButtonRelease,.15,6,Qt.LeftButton,modifiers=erase)
        assert not view.arrivals and not view._arrival_items
        for ch,t in saved.items(): view.mark_arrival(ch,t)
        win.control_panel.display_mode_combo.setCurrentText("时空颜色图")
        mouse(QEvent.MouseButtonPress,.15,4,Qt.LeftButton,Qt.LeftButton,erase)
        mouse(QEvent.MouseButtonRelease,.15,4,Qt.LeftButton,modifiers=erase)
        assert view.arrivals == saved, "Color mode must not edit arrivals"
        win.control_panel.display_mode_combo.setCurrentText("多道波形(Wiggle)")
        # Filter/replot preserves picks for this file group.
        view.mark_arrival(7,.195)  # Outside the cropped time range and retained channels.
        retained = dict(view.arrivals)
        win._render_current_data()
        assert view.arrivals == retained
        view.set_x_range(.010,.181)
        view.set_y_range(2,8)
        out = root/"exports"; out.mkdir()
        win.control_panel.save_filtered_checkbox.setChecked(True)
        win.control_panel.export_format_combo.setCurrentText("NPZ")
        assert not win.control_panel.save_filtered_checkbox.isEnabled()
        win.control_panel.export_time_step_edit.setText("3")
        win.control_panel.export_space_step_edit.setText("2")
        request = win._capture_export_request()
        assert request.format == "npz" and request.time_step==3 and request.space_step==2
        # A worker uses the captured range and picks even if GUI state changes.
        view.set_x_range(0,.1); view.clear_arrivals()
        result = export_view(request,out)
        data,header = read_edas_npz(result['output_path'])
        expected = raw[request.frame_start:request.frame_end:3,2:8:2]
        np.testing.assert_array_equal(data,expected)
        assert data.dtype == np.int32 and not result['filtered']
        assert header['data_type']=='eDAS' and header['raw_unfiltered']
        assert header['channel_numbers']==[2,4,6] and header['channel_start']==2 and header['channel_step']==2
        assert header['sample_rate_hz']==1000/3
        assert header['start_time']==(group.start_time+timedelta(seconds=request.frame_start/1000)).isoformat()
        assert {p['channel'] for p in header['arrivals']}=={2,4,6}
        for p in header['arrivals']:
            assert np.isclose(p['time_s'],retained[p['channel']]-request.frame_start/1000)
        assert result['output_path'] in list_bin_files(out)
        meta = inspect_bin_file(result['output_path'])
        np.testing.assert_array_equal(read_bin_memmap(meta),expected)
        assert (meta.channel_start,meta.channel_end,meta.channel_step)==(2,6,2)
        # Fully reload through application, including absolute channel IDs and marks.
        payload = win._load_file_data(meta,FilterConfig(),[meta],win._file_load_generation)
        win._on_data_loaded(payload)
        app.processEvents()
        assert win._read_point_range()==(2,6)
        npz_picks = dict(view.arrivals)
        view.erase_arrival_segment(6, 2)
        assert not view.arrivals and not view._arrival_items
        for ch,t in npz_picks.items(): view.mark_arrival(ch,t)
        assert view._wiggle_stored_channels==[2,4,6]
        assert set(view.arrivals)=={2,4,6}
        view.update_waveform_point(4)
        np.testing.assert_array_equal(view.waveform_curve.getData()[1],read_phase_radians(meta)[:,1])
        win.channel_analysis_view.set_data(meta,win.current_data,[2,6],psd_nperseg=16)
        assert win._parse_channel_list("2,6")==[2,6]
        try: win._parse_channel_list("3")
        except ValueError: pass
        else: raise AssertionError("A removed spatial channel must not be accepted")
        # A second cropped export retains the actual source channel identities.
        view.set_x_range(0,meta.duration_s); view.set_y_range(4,8)
        second = win._capture_export_request(False,False,"npz",2,1)
        second_result = export_view(second,out)
        data2,head2 = read_edas_npz(second_result['output_path'])
        np.testing.assert_array_equal(data2,expected[::2,1:])
        assert head2['channel_numbers']==[4,6]
        # BIN/TXT use the same stride and raw selection, including across files.
        for fmt in ('bin','txt'):
            r=replace(request,format=fmt,processed=False)
            exported=export_view(r,out)
            if fmt=='bin':
                np.testing.assert_array_equal(np.fromfile(exported['output_path'],dtype=np.int32).reshape(expected.shape),expected)
                inspected=inspect_bin_file(exported['output_path'])
                assert np.isclose(inspected.sample_rate_hz,1000/3)
            else:
                from scripts.io.bin_reader import convert_raw_to_radians
                np.testing.assert_allclose(np.loadtxt(exported['output_path']),convert_raw_to_radians(expected),rtol=1e-6)
        # Reject generic, wrongly tagged, object-containing or inconsistent archives.
        for name, bad in [('wrong_type',dict(header,data_type='other')),('bad_shape',dict(header,channel_count=99)),
                          ('bad_rate',dict(header,sample_rate_hz=0)),('bad_pick',dict(header,arrivals=[{'channel':3,'time_s':0}]))]:
            path=root/(name+'.npz'); np.savez(path,data=data,header=np.array(json.dumps(bad)))
            try: inspect_bin_file(path)
            except ValueError: pass
            else: raise AssertionError(name+' was accepted')
        path=root/'generic.npz';np.savez(path,data=data)
        try: inspect_bin_file(path)
        except ValueError: pass
        else: raise AssertionError('Generic NPZ was accepted')
        # Session annotations remain associated with their source group.
        view.set_arrival_context(tuple(str(item.file_path) for item in sources),group)
        assert not view.arrivals  # The earlier explicit clear belongs only to this group.
        view.set_arrival_context((str(meta.file_path),),meta)
        assert set(view.arrivals)=={2,4,6}
        # Export options persist independently of the raw-only NPZ rule.
        win._save_parameters()
        store=QSettings(QSettings.IniFormat,QSettings.UserScope,"eDASread","eDASread")
        assert store.value("tab1/export_format")=="NPZ"
        assert str(store.value("tab1/export_time_step"))=="3"
        view._hover_channel=4
        view.mark_arrival(4,0);view.adjust_hovered_arrival(-1);assert view.arrivals[4]==0
        win.control_panel.clear_arrivals_button.click();assert not view.arrivals
        win.close();app.processEvents()
        app.removeEventFilter(win)
        win.deleteLater();app.processEvents()
    print('PASS: Ctrl click/drag/arrow, Ctrl+Shift click/drag erase and color isolation, replot, immutable export, raw stitched NPZ, crop/stride metadata, import/re-export, BIN/TXT and invalid NPZ rejection')

if __name__=='__main__':
    main()
