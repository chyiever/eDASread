"""Main window implementation for the FIPread application."""

from __future__ import annotations

import re
import tempfile
import wave
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QThreadPool, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import winsound
except ImportError:  # pragma: no cover - Windows-only playback backend
    winsound = None

from scripts.core.cache import LruByteCache
from scripts.core.models import FileMetadata, FilterConfig
from scripts.core.resources import asset_path
from scripts.core.settings import ParameterStore
from scripts.io.bin_reader import (
    batch_compress_bin_folder,
    build_export_filename,
    convert_radians_to_raw,
    convert_raw_to_radians,
    infer_points_from_filename,
    inspect_bin_file,
    list_raw_bin_files,
    list_bin_files,
    read_bin_memmap,
    read_phase_radians,
)
from scripts.io.text_export import write_phase_txt
from scripts.processing.denoise_pipeline import apply_denoise_step, compute_fk_spectrum_db
from scripts.processing.filters import apply_filter
from scripts.processing.time_frequency import compute_time_frequency_map
from scripts.render.analysis_views import ChannelAnalysisView, FkSpectrumView, TimeFrequencyAnalysisView
from scripts.render.space_time_view import SpaceTimeView
from scripts.processing.traffic_trajectory import build_vehicle_energy_image
from scripts.render.traffic_trajectory_view import TrafficTrajectoryView
from scripts.ui.control_panel import ControlPanel
from scripts.ui.file_browser import FileBrowserPanel
from scripts.workers.tasks import FunctionWorker


class MainWindow(QMainWindow):
    """Main desktop window coordinating file loading and plotting."""

    CACHE_MAX_BYTES = 1_000_000_000
    HORIZONTAL_KEY_STEP_RATIO = 0.4
    AUTO_POINT_CAP = 100
    WINDOW_TITLE = "分布式光纤传感数据回放软件(eDASread)"
    PREPROCESS_LABELS = {
        "spike": "去尖峰噪声",
        "common": "去共模噪声",
        "random": "去随机噪声",
        "fk": "FK滤波",
    }
    BUTTON_STYLE = """
        QPushButton {
            background-color: #2f6db2;
            color: white;
            border: 1px solid #224f82;
            border-radius: 4px;
            padding: 5px 12px;
            font-family: "Microsoft YaHei", "SimSun";
            font-size: 15px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #3a7dcb;
        }
        QPushButton:pressed {
            background-color: #234f82;
        }
        QPushButton:disabled {
            background-color: #9aa8b8;
            border-color: #8794a4;
            color: #eef3f8;
        }
        QTabWidget::pane {
            border: 1px solid #b7c1ce;
            top: -1px;
        }
        QTabBar {
            font-family: "Microsoft YaHei", "SimSun";
            font-size: 18px;
            font-weight: 600;
        }
        QTabBar::tab {
            background: #eef1f5;
            border: 1px solid #c4ccd6;
            padding: 8px 40px;
            color: #1f2937;
        }
        QTabBar::tab:selected {
            background: #2f6db2;
            color: white;
            border-color: #224f82;
        }
        QTabBar::tab:hover:!selected {
            background: #dbe7f5;
        }
    """

    def __init__(self, default_directory: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1500, 900)
        self._apply_window_icons()

        self.thread_pool = QThreadPool.globalInstance()
        self.processed_cache = LruByteCache(max_bytes=self.CACHE_MAX_BYTES)
        self.file_paths: list[Path] = []
        self.file_sort_mode = "name_asc"
        self.current_page = 0
        self.current_metadata: Optional[FileMetadata] = None
        self.current_data: Optional[np.ndarray] = None
        self.current_file_path: Optional[Path] = None
        self.current_file_paths: list[Path] = []
        self.loaded_file_paths: list[Path] = []
        self.current_filter_config = FilterConfig()
        self.current_preprocess_steps: list[str] = []
        self._active_workers: set[FunctionWorker] = set()
        self._point_end_manual_override = False
        self._bz_progress_dialog: Optional[QProgressDialog] = None
        self._audio_temp_paths: list[Path] = []
        self._file_load_generation = 0

        self.file_browser = FileBrowserPanel()
        self.control_panel = ControlPanel()
        self.space_time_view = SpaceTimeView()
        self.channel_analysis_view = ChannelAnalysisView()
        self.fk_spectrum_view = FkSpectrumView()
        self.time_frequency_view = TimeFrequencyAnalysisView()
        self.traffic_trajectory_view = TrafficTrajectoryView()
        self.traffic_trajectory_view.generate_button.setEnabled(False)
        self.plot_tabs = QTabWidget()
        self.horizontal_scroll = QScrollBar(Qt.Horizontal)
        self.vertical_scroll = QScrollBar(Qt.Vertical)
        self.title_label = QLabel(self.WINDOW_TITLE)

        self._is_syncing_scrollbars = False
        self._directory_load_timer = QTimer(self)
        self._directory_load_timer.setSingleShot(True)
        self._directory_load_timer.setInterval(400)
        self._directory_load_timer.timeout.connect(self._load_directory_from_timer)
        self._file_selection_timer = QTimer(self)
        self._file_selection_timer.setSingleShot(True)
        self._file_selection_timer.setInterval(180)
        self._file_selection_timer.timeout.connect(self._on_file_selected)
        self._parameter_store = ParameterStore()
        self._parameter_save_timer = QTimer(self)
        self._parameter_save_timer.setSingleShot(True)
        self._parameter_save_timer.setInterval(600)
        self._parameter_save_timer.timeout.connect(self._save_parameters)
        self._build_layout()
        self._connect_signals()
        self._connect_parameter_autosave()
        restored_keys = self._restore_persisted_parameters()
        if "tab1/point_end" in restored_keys:
            # Persisted point range belongs to the user: keep it across files.
            self._point_end_manual_override = True
        self._on_waveform_point_changed()
        self.setStyleSheet(self.BUTTON_STYLE)

        QApplication.instance().installEventFilter(self)

        status_bar = QStatusBar()
        status_bar.setFont(QFont("Times New Roman", 10))
        self.status_note_label = QLabel("\u4e2d\u56fd\u79d1\u5b66\u9662\u534a\u5bfc\u4f53\u6240\u5f00\u53d1")
        self.status_note_label.setFont(QFont("SimSun", 9))
        self.status_note_label.setStyleSheet("color: #555;")
        status_bar.addPermanentWidget(self.status_note_label)
        self.setStatusBar(status_bar)

        if not self.file_browser.path_edit.text().strip() and default_directory is not None:
            self.file_browser.path_edit.setText(str(default_directory))
        if not self.control_panel.export_path_edit.text().strip() and default_directory is not None:
            self.control_panel.export_path_edit.setText(str(default_directory))

        self._directory_load_timer.stop()
        self.load_directory(show_errors=False)

    def _build_layout(self) -> None:
        title_font = QFont("SimHei", 30)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: black;")
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.title_label.setFixedHeight(64)

        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(0, 0, 0, 0)
        title_row_layout.setSpacing(6)
        title_row_layout.addStretch(1)
        title_row_layout.addWidget(self.title_label, 0, Qt.AlignCenter)
        title_row_layout.addStretch(1)

        tab1_panel = QWidget()
        tab1_layout = QHBoxLayout(tab1_panel)
        tab1_layout.setContentsMargins(0, 0, 0, 0)

        plot_column = QVBoxLayout()
        plot_column.setContentsMargins(0, 0, 0, 0)
        plot_column.setSpacing(6)
        plot_column.addWidget(self.control_panel)
        plot_column.addWidget(self.space_time_view, 1)
        plot_column.addWidget(self.horizontal_scroll)

        tab1_layout.addLayout(plot_column, 1)
        tab1_layout.addWidget(self.vertical_scroll)

        self.plot_tabs.addTab(tab1_panel, "Tab1 数据预处理")
        self.plot_tabs.addTab(self.channel_analysis_view, "Tab2 通道时域/PSD")
        self.plot_tabs.addTab(self.fk_spectrum_view, "Tab3 f-k谱")
        self.plot_tabs.addTab(self.time_frequency_view, "Tab4 时域/时频")
        self.plot_tabs.addTab(self.traffic_trajectory_view, "Tab5 车辆轨迹标记")
        self.plot_tabs.tabBar().setElideMode(Qt.ElideNone)

        splitter = QSplitter()
        splitter.addWidget(self.file_browser)
        splitter.addWidget(self.plot_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 1100])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(title_row)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(container)

    def _connect_signals(self) -> None:
        self.file_browser.path_edit.returnPressed.connect(lambda: self.load_directory(show_errors=True))
        self.file_browser.path_edit.textChanged.connect(self._schedule_directory_load)
        self.file_browser.compress_bz_button.clicked.connect(self.compress_current_directory_to_bz)
        self.file_browser.sort_combo.currentIndexChanged.connect(self._on_file_sort_changed)
        self.file_browser.previous_button.clicked.connect(self.show_previous_page)
        self.file_browser.next_button.clicked.connect(self.show_next_page)
        self.file_browser.jump_button.clicked.connect(self.jump_to_page)
        self.file_browser.page_jump_edit.returnPressed.connect(self.jump_to_page)
        self.file_browser.file_list.itemSelectionChanged.connect(self._schedule_file_selection_load)

        self.control_panel.apply_button.clicked.connect(self.apply_current_settings)
        self.control_panel.reset_button.clicked.connect(self.reset_controls)
        self.control_panel.reset_view_button.clicked.connect(self.space_time_view.reset_view)
        self.control_panel.back_view_button.clicked.connect(self.space_time_view.back_view)
        self.control_panel.zoom_out_2x_button.clicked.connect(self._on_zoom_out_2x)
        self.control_panel.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.control_panel.export_button.clicked.connect(self.export_current_view)
        self.control_panel.point_start_edit.textChanged.connect(self._sync_waveform_point_bounds)
        self.control_panel.point_end_edit.textChanged.connect(self._sync_waveform_point_bounds)
        self.control_panel.point_end_edit.textEdited.connect(self._on_point_end_text_edited)
        self.control_panel.waveform_point_edit.editingFinished.connect(self._on_waveform_point_changed)
        self.control_panel.waveform_point_edit.wheelStepped.connect(self._on_waveform_point_wheel_stepped)
        self.control_panel.play_waveform_button.clicked.connect(self.play_current_waveform)
        self.control_panel.spike_button.clicked.connect(lambda: self.run_preprocess_step("spike"))
        self.control_panel.common_button.clicked.connect(lambda: self.run_preprocess_step("common"))
        self.control_panel.random_button.clicked.connect(lambda: self.run_preprocess_step("random"))
        self.control_panel.fk_filter_button.clicked.connect(lambda: self.run_preprocess_step("fk"))

        self.channel_analysis_view.plot_requested.connect(self.plot_selected_channels)
        self.fk_spectrum_view.plot_requested.connect(self.plot_fk_spectrum)
        self.time_frequency_view.plot_requested.connect(self.plot_time_frequency)
        self.traffic_trajectory_view.generate_requested.connect(self.generate_traffic_energy_image)

        self.space_time_view.view_range_changed.connect(self._sync_scrollbars_to_view)
        self.horizontal_scroll.valueChanged.connect(self._on_horizontal_scroll)
        self.vertical_scroll.valueChanged.connect(self._on_vertical_scroll)

    def eventFilter(self, watched, event):
        """Allow left/right arrows to drive horizontal scrolling outside text inputs."""
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Left, Qt.Key_Right):
            focus_widget = QApplication.focusWidget()
            if isinstance(focus_widget, (QLineEdit, QComboBox)):
                return super().eventFilter(watched, event)

            direction = -1 if event.key() == Qt.Key_Left else 1

            # Tab2 owns its own keyboard scrolling (the most recently clicked
            # time/PSD plot), so do not hijack left/right while Tab2 is active.
            if self.plot_tabs.currentWidget() is self.channel_analysis_view:
                self.channel_analysis_view.pan_active_plot(direction)
                return True
            if self.plot_tabs.currentWidget() is self.time_frequency_view:
                self.time_frequency_view.pan_active_plot(direction)
                return True

            self._step_horizontal_scroll(direction)
            return True

        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        """Persist the latest parameters so they reload on next launch."""
        self._directory_load_timer.stop()
        self._file_selection_timer.stop()
        self._parameter_save_timer.stop()
        try:
            self._save_parameters()
        finally:
            super().closeEvent(event)

    def _schedule_directory_load(self) -> None:
        """Debounce directory loading while the user edits the path."""
        self._directory_load_timer.start()

    def _load_directory_from_timer(self) -> None:
        self.load_directory(show_errors=False)

    def load_directory(self, show_errors: bool = True) -> None:
        """Load files from the directory typed into the file browser."""
        self._directory_load_timer.stop()
        directory = self.file_browser.path_edit.text().strip()
        if not directory:
            self.file_paths = []
            self._clear_file_listing()
            return

        try:
            self.file_paths = list_bin_files(directory)
            self._sort_file_paths()
        except Exception as exc:
            self.file_paths = []
            self._clear_file_listing()
            if show_errors:
                self._show_error(str(exc))
            return

        self.current_page = 0
        self._refresh_file_page()
        if self.file_browser.file_list.count() > 0:
            self.file_browser.file_list.setCurrentRow(0)

    def compress_current_directory_to_bz(self) -> None:
        """Batch-compress all bin files in the current folder to bz."""
        directory_text = self.file_browser.path_edit.text().strip()
        if not directory_text:
            self._show_error("Please choose a data folder first.")
            return

        source_folder = Path(directory_text)
        try:
            bin_files = list_raw_bin_files(source_folder)
        except Exception as exc:
            self._show_error(str(exc))
            return
        if not bin_files:
            self._show_error(f"No .bin files found in {source_folder}.")
            return

        export_folder_text = self.control_panel.export_path_edit.text().strip()
        output_folder = Path(export_folder_text) if export_folder_text else source_folder
        reply = QMessageBox.question(
            self,
            "Batch Bin -> bz",
            (
                f"Compress {len(bin_files)} bin files to bz?\n"
                f"Output folder: {output_folder}\n"
                "Existing .bz files will be skipped."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.file_browser.compress_bz_button.setEnabled(False)
        self._show_bz_progress_dialog(len(bin_files))
        self.statusBar().showMessage("Compressing bin files to bz ...")
        worker = FunctionWorker(
            batch_compress_bin_folder,
            source_folder,
            output_folder,
            overwrite=False,
            progress_callback_arg="progress_callback",
        )
        self._active_workers.add(worker)
        worker.signals.progress.connect(self._on_bz_compression_progress)
        worker.signals.result.connect(self._on_bz_compression_finished)
        worker.signals.error.connect(self._on_bz_compression_error)
        worker.signals.finished.connect(lambda: self.file_browser.compress_bz_button.setEnabled(True))
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _show_bz_progress_dialog(self, file_count: int) -> None:
        self._close_bz_progress_dialog()
        dialog = QProgressDialog(self)
        dialog.setWindowTitle("Bin -> bz")
        dialog.setLabelText(f"准备压缩 {file_count} 个文件...")
        dialog.setRange(0, 1000)
        dialog.setValue(0)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.resize(520, 120)
        dialog.show()
        self._bz_progress_dialog = dialog

    def _on_bz_compression_progress(self, payload: dict) -> None:
        dialog = self._bz_progress_dialog
        if dialog is None:
            return

        total_bytes = max(1, int(payload.get("total_bytes", 1)))
        current_bytes = max(0, min(int(payload.get("current_bytes", 0)), total_bytes))
        progress_value = int(round(current_bytes * 1000 / total_bytes))
        status = str(payload.get("status", "compressing"))
        file_name = str(payload.get("file_name", ""))
        file_index = int(payload.get("file_index", 0))
        file_count = int(payload.get("file_count", 0))

        if status == "skipped":
            action_text = "跳过已有 bz"
        elif status == "finished":
            action_text = "压缩完成"
        else:
            action_text = "正在压缩"

        label_parts = [
            f"{action_text}: {file_index}/{file_count}",
            f"{self._format_byte_size(current_bytes)} / {self._format_byte_size(total_bytes)}",
        ]
        if file_name:
            label_parts.insert(1, file_name)
        dialog.setLabelText("\n".join(label_parts))
        dialog.setValue(progress_value)
        self.statusBar().showMessage(" | ".join(label_parts))

    def _on_bz_compression_finished(self, payload: dict) -> None:
        self._close_bz_progress_dialog()
        compression_percent = float(payload["compression_percent"])
        elapsed_s = float(payload["elapsed_s"])
        throughput_mib_s = float(payload["throughput_mib_s"])
        self.statusBar().showMessage(
            (
                f"Bz compression finished | Converted: {payload['converted']} | "
                f"Skipped: {payload['skipped']} | "
                f"Compression: {compression_percent:.2f}% | "
                f"Speed: {throughput_mib_s:.2f} MB/s | Output: {payload['output_folder']}"
            )
        )
        self.load_directory(show_errors=False)
        QMessageBox.information(
            self,
            "压缩完成",
            (
                f"压缩了 {payload['converted']} 个文件。\n"
                f"跳过已有 bz 文件：{payload['skipped']} 个。\n"
                f"原始数据体积：{self._format_byte_size(payload['converted_raw_bytes'])}\n"
                f"压缩后体积：{self._format_byte_size(payload['converted_compressed_bytes'])}\n"
                f"压缩比：{compression_percent:.2f}%\n"
                f"用时：{elapsed_s:.3f}s\n"
                f"压缩效率：{throughput_mib_s:.2f} MB/s"
            ),
        )

    def _on_bz_compression_error(self, message: str) -> None:
        self._close_bz_progress_dialog()
        self._show_error(message)

    def _close_bz_progress_dialog(self) -> None:
        if self._bz_progress_dialog is None:
            return
        self._bz_progress_dialog.close()
        self._bz_progress_dialog.deleteLater()
        self._bz_progress_dialog = None

    def _format_byte_size(self, byte_count: int | float) -> str:
        size = float(byte_count)
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        while abs(size) >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        return f"{size:.2f} {units[unit_index]}"

    def _clear_file_listing(self) -> None:
        self.file_browser.file_list.clear()
        self.file_browser.page_label.setText("Page 0 / 0")
        self.file_browser.previous_button.setEnabled(False)
        self.file_browser.next_button.setEnabled(False)
        self.file_browser.page_jump_edit.setPlaceholderText("page / home / end")

    def _sort_file_paths(self) -> None:
        mode = self.file_sort_mode
        if mode == "mtime_desc":
            self.file_paths.sort(key=lambda path: (-self._file_mtime_ns(path), path.name.casefold()))
        elif mode == "mtime_asc":
            self.file_paths.sort(key=lambda path: (self._file_mtime_ns(path), path.name.casefold()))
        else:
            self.file_paths.sort(key=lambda path: path.name.casefold())

    def _file_mtime_ns(self, path: Path) -> int:
        try:
            return int(path.stat().st_mtime_ns)
        except OSError:
            return 0

    def _on_file_sort_changed(self) -> None:
        mode = self.file_browser.sort_combo.currentData() or "name_asc"
        self.file_sort_mode = str(mode)
        selected_path = self._selected_file_path() or self.current_file_path
        self._sort_file_paths()

        if selected_path in self.file_paths:
            selected_index = self.file_paths.index(selected_path)
            self.current_page = selected_index // self.file_browser.PAGE_SIZE
            selected_row = selected_index % self.file_browser.PAGE_SIZE
        else:
            self.current_page = 0
            selected_row = -1

        self.file_browser.file_list.blockSignals(True)
        try:
            self._refresh_file_page()
            if 0 <= selected_row < self.file_browser.file_list.count():
                self.file_browser.file_list.setCurrentRow(selected_row)
        finally:
            self.file_browser.file_list.blockSignals(False)

    def _refresh_file_page(self) -> None:
        if not self.file_paths:
            self._clear_file_listing()
            return

        page_size = self.file_browser.PAGE_SIZE
        total_pages = (len(self.file_paths) + page_size - 1) // page_size
        self.current_page = max(0, min(self.current_page, total_pages - 1))
        start = self.current_page * page_size
        end = start + page_size

        self.file_browser.file_list.clear()
        for path in self.file_paths[start:end]:
            self.file_browser.file_list.addItem(path.name)

        self.file_browser.page_label.setText(f"Page {self.current_page + 1} / {total_pages}")
        self.file_browser.previous_button.setEnabled(self.current_page > 0)
        self.file_browser.next_button.setEnabled(self.current_page < total_pages - 1)
        self.file_browser.page_jump_edit.setPlaceholderText(f"1-{total_pages} / home / end")

    def show_previous_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self._refresh_file_page()

    def show_next_page(self) -> None:
        page_size = self.file_browser.PAGE_SIZE
        if (self.current_page + 1) * page_size < len(self.file_paths):
            self.current_page += 1
            self._refresh_file_page()

    def jump_to_page(self) -> None:
        page_text = self.file_browser.page_jump_edit.text().strip().lower()
        if not page_text:
            return

        total_pages = max(1, (len(self.file_paths) + self.file_browser.PAGE_SIZE - 1) // self.file_browser.PAGE_SIZE)
        if page_text == "home":
            target_page = 0
        elif page_text == "end":
            target_page = total_pages - 1
        else:
            try:
                target_page = int(page_text) - 1
            except ValueError:
                self._show_error(f"Invalid page input: {page_text}")
                return
            if target_page < 0 or target_page >= total_pages:
                self._show_error(f"Page must be between 1 and {total_pages}.")
                return

        self.current_page = target_page
        self._refresh_file_page()
        if self.file_browser.file_list.count() > 0:
            self.file_browser.file_list.setCurrentRow(0)
        self.file_browser.page_jump_edit.clear()

    def _selected_file_path(self) -> Optional[Path]:
        selected_row = self.file_browser.file_list.currentRow()
        if selected_row < 0:
            return None
        index = self.current_page * self.file_browser.PAGE_SIZE + selected_row
        return self.file_paths[index] if 0 <= index < len(self.file_paths) else None

    def _selected_file_paths(self) -> list[Path]:
        selected_rows = sorted(index.row() for index in self.file_browser.file_list.selectedIndexes())
        paths: list[Path] = []
        for row in selected_rows:
            index = self.current_page * self.file_browser.PAGE_SIZE + row
            if 0 <= index < len(self.file_paths):
                paths.append(self.file_paths[index])
        return paths

    def _schedule_file_selection_load(self) -> None:
        """Coalesce the several selection events produced by Ctrl/Shift multi-select."""
        self._file_selection_timer.start()

    def _on_file_selected(self) -> None:
        file_paths = self._selected_file_paths()
        if not file_paths:
            return

        self._apply_auto_point_limit(file_path=file_paths[0])
        try:
            self._load_files(file_paths, self._build_filter_config())
        except Exception as exc:
            self._show_error(str(exc))

    def _load_file(self, file_path: Path, filter_config: FilterConfig) -> None:
        self._load_files([file_path], filter_config)

    def _load_files(self, file_paths: Sequence[Path], filter_config: FilterConfig) -> None:
        if not file_paths:
            raise ValueError("No file selected.")
        metadata, source_metadata = self._inspect_file_group(file_paths)
        self._file_load_generation += 1
        load_generation = self._file_load_generation
        self.current_file_paths = [item.file_path for item in source_metadata]
        self.traffic_trajectory_view.generate_button.setEnabled(False)
        self._apply_auto_point_limit(metadata=metadata)
        self.statusBar().showMessage(self._build_loading_status_text(metadata))
        worker = FunctionWorker(
            self._load_file_data, metadata, filter_config, source_metadata, load_generation
        )
        self._active_workers.add(worker)
        worker.signals.result.connect(self._on_data_loaded)
        worker.signals.error.connect(
            lambda message, generation=load_generation: self._on_file_load_error(message, generation)
        )
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _inspect_file_group(self, file_paths: Sequence[Path]) -> tuple[FileMetadata, list[FileMetadata]]:
        source_metadata = [inspect_bin_file(path) for path in file_paths]
        source_metadata.sort(key=self._metadata_sort_key)
        first = source_metadata[0]
        for metadata in source_metadata[1:]:
            if metadata.points != first.points:
                raise ValueError("多选拼接要求所有文件的位置点数一致。")
            if not np.isclose(metadata.sample_rate_hz, first.sample_rate_hz):
                raise ValueError("多选拼接要求所有文件的采样率一致。")

        if len(source_metadata) == 1:
            return first, source_metadata

        frame_count = sum(metadata.frame_count for metadata in source_metadata)
        file_size_bytes = sum(metadata.file_size_bytes for metadata in source_metadata)
        synthetic_name = f"{source_metadata[0].file_path.stem}+{len(source_metadata)}files{source_metadata[0].file_path.suffix}"
        group_metadata = FileMetadata(
            file_path=source_metadata[0].file_path.with_name(synthetic_name),
            sample_rate_hz=first.sample_rate_hz,
            points=first.points,
            frame_count=frame_count,
            duration_s=frame_count / first.sample_rate_hz,
            file_size_bytes=file_size_bytes,
            start_time=first.start_time,
        )
        return group_metadata, source_metadata

    def _metadata_sort_key(self, metadata: FileMetadata) -> tuple:
        if metadata.start_time is None:
            return (1, metadata.file_path.name.casefold())
        return (0, metadata.start_time, metadata.file_path.name.casefold())

    def _load_file_data(
        self,
        metadata: FileMetadata,
        filter_config: FilterConfig,
        source_metadata: Sequence[FileMetadata],
        load_generation: int,
    ) -> dict:
        source_paths = tuple(str(item.file_path) for item in source_metadata)
        cache_key = source_paths + filter_config.cache_key()
        processed = self.processed_cache.get(cache_key)
        if processed is None:
            if len(source_metadata) == 1:
                phase_data = read_phase_radians(source_metadata[0])
            else:
                chunks = [read_phase_radians(item) for item in source_metadata]
                phase_data = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
            processed = apply_filter(phase_data, metadata.sample_rate_hz, filter_config)
            self.processed_cache.put(cache_key, processed, int(processed.nbytes))

        return {
            "metadata": metadata,
            "data": processed,
            "file_path": metadata.file_path,
            "file_paths": [item.file_path for item in source_metadata],
            "filter_config": filter_config,
            "load_generation": load_generation,
        }

    def _on_data_loaded(self, payload: dict) -> None:
        if int(payload.get("load_generation", -1)) != self._file_load_generation:
            return
        self.current_metadata = payload["metadata"]
        self.current_data = payload["data"]
        self.current_file_path = payload["file_path"]
        self.current_file_paths = list(payload.get("file_paths", [self.current_file_path]))
        self.loaded_file_paths = list(self.current_file_paths)
        self.traffic_trajectory_view.generate_button.setEnabled(True)
        self.current_filter_config = payload["filter_config"]
        self.current_preprocess_steps = []
        self._apply_auto_point_limit(metadata=self.current_metadata)
        try:
            self._render_current_data()
        except Exception as exc:
            self._show_error(str(exc))

    def apply_current_settings(self) -> None:
        """Apply filter and display parameters to the currently selected file."""
        file_paths = self._selected_file_paths() or self.current_file_paths
        if not file_paths:
            return
        try:
            self._load_files(file_paths, self._build_filter_config())
        except Exception as exc:
            self._show_error(str(exc))

    def _on_file_load_error(self, message: str, load_generation: int) -> None:
        if load_generation != self._file_load_generation:
            return
        self.traffic_trajectory_view.generate_button.setEnabled(
            self.current_data is not None and self.loaded_file_paths == self.current_file_paths
        )
        self._show_error(message)

    def run_preprocess_step(self, step: str) -> None:
        """Run one DAS denoising step against the current processed data."""
        if self.current_metadata is None or self.current_data is None or self.current_file_path is None:
            self._show_error("No file loaded.")
            return
        if step not in self.PREPROCESS_LABELS:
            self._show_error(f"Unsupported preprocessing step: {step}")
            return

        try:
            spacing_m, fk_vmin_mps, fk_vmax_mps = self._read_tab1_fk_parameters()
        except Exception as exc:
            self._show_error(str(exc))
            return

        label = self.PREPROCESS_LABELS[step]
        self.control_panel.set_denoise_buttons_enabled(False)
        self.statusBar().showMessage(f"{label}处理中 ...")
        worker = FunctionWorker(
            self._apply_preprocess_step,
            self.current_data,
            step,
            self.current_metadata.sample_rate_hz,
            spacing_m,
            fk_vmin_mps,
            fk_vmax_mps,
            self.current_file_path,
        )
        self._active_workers.add(worker)
        worker.signals.result.connect(self._on_preprocess_finished)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda: self.control_panel.set_denoise_buttons_enabled(True))
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _apply_preprocess_step(
        self,
        data: np.ndarray,
        step: str,
        sample_rate_hz: float,
        spacing_m: float,
        fk_vmin_mps: float,
        fk_vmax_mps: float,
        file_path: Path,
    ) -> dict:
        processed = apply_denoise_step(
            data,
            step,
            sample_rate_hz=sample_rate_hz,
            channel_spacing_m=spacing_m,
            fk_vmin_mps=fk_vmin_mps,
            fk_vmax_mps=fk_vmax_mps,
        )
        return {
            "data": processed,
            "step": step,
            "file_path": file_path,
        }

    def _on_preprocess_finished(self, payload: dict) -> None:
        if self.current_file_path != payload["file_path"]:
            self.statusBar().showMessage("预处理结果已过期：当前文件已切换。")
            return

        step = str(payload["step"])
        self.current_data = payload["data"]
        self.current_preprocess_steps.append(step)
        try:
            self._render_current_data()
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.statusBar().showMessage(
            f"{self.PREPROCESS_LABELS.get(step, step)}完成 | 当前预处理：{self._format_preprocess_steps()}"
        )

    def plot_selected_channels(self) -> None:
        """Draw Tab2 selected-channel time traces and PSD curves."""
        if self.current_metadata is None or self.current_data is None:
            self._show_error("No file loaded.")
            return
        try:
            channels = self._parse_channel_list(self.channel_analysis_view.channels_edit.text())
            psd_options = self._read_tab2_psd_options()
            self.channel_analysis_view.set_data(
                self.current_metadata,
                self.current_data,
                channels,
                **psd_options,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.statusBar().showMessage(f"Tab2 已绘制通道：{', '.join(str(ch) for ch in channels)}")

    def plot_fk_spectrum(self) -> None:
        """Compute and draw Tab3 f-k spectrum from current processed data."""
        if self.current_metadata is None or self.current_data is None:
            self._show_error("No file loaded.")
            return

        try:
            channel_start, channel_end = self._parse_channel_range(
                self.fk_spectrum_view.channel_range_edit.text()
            )
            spacing_m = self._read_positive_float(self.fk_spectrum_view.spacing_edit, "Tab3 通道间距")
            fk_options = self._read_tab3_fk_options()
            levels = self._read_fk_db_levels()
        except Exception as exc:
            self._show_error(str(exc))
            return

        if channel_end <= channel_start:
            self._show_error("f-k 谱至少需要两个通道。")
            return

        self.statusBar().showMessage("Tab3 f-k 谱计算中 ...")
        worker = FunctionWorker(
            self._compute_fk_spectrum_for_view,
            self.current_data,
            self.current_metadata.sample_rate_hz,
            channel_start,
            channel_end,
            spacing_m,
            fk_options,
            levels,
            self.current_file_path,
        )
        self._active_workers.add(worker)
        worker.signals.result.connect(self._on_fk_spectrum_ready)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _compute_fk_spectrum_for_view(
        self,
        data: np.ndarray,
        sample_rate_hz: float,
        channel_start: int,
        channel_end: int,
        spacing_m: float,
        fk_options: dict,
        levels: tuple[float, float],
        file_path: Path | None,
    ) -> dict:
        channel_slice = slice(channel_start - 1, channel_end)
        gather = np.asarray(data[:, channel_slice], dtype=np.float32)
        spectrum = compute_fk_spectrum_db(
            gather,
            sample_rate_hz=sample_rate_hz,
            channel_spacing_m=spacing_m,
            taper_fraction=fk_options["taper_fraction"],
            remove_background=fk_options["remove_background"],
            nfft_channels=fk_options["nfft_channels"],
            nfft_samples=fk_options["nfft_samples"],
        )
        return {
            "spectrum": spectrum,
            "channel_start": channel_start,
            "channel_end": channel_end,
            "spacing_m": spacing_m,
            "frequency_range": fk_options["frequency_range"],
            "wavenumber_range": fk_options["wavenumber_range"],
            "levels": levels,
            "file_path": file_path,
        }

    def _on_fk_spectrum_ready(self, payload: dict) -> None:
        if self.current_file_path != payload["file_path"]:
            self.statusBar().showMessage("f-k 谱结果已过期：当前文件已切换。")
            return

        self.fk_spectrum_view.set_spectrum(
            payload["spectrum"],
            channel_start=payload["channel_start"],
            channel_end=payload["channel_end"],
            spacing_m=payload["spacing_m"],
            levels=payload["levels"],
            frequency_range=payload["frequency_range"],
            wavenumber_range=payload["wavenumber_range"],
        )
        self.statusBar().showMessage(
            f"Tab3 已绘制 f-k 谱：Ch {payload['channel_start']}-{payload['channel_end']}"
        )

    def plot_time_frequency(self) -> None:
        """Compute and draw Tab4 waveform and time-frequency map."""
        if self.current_metadata is None or self.current_data is None:
            self._show_error("No file loaded.")
            return

        try:
            channel = self._read_channel_number(self.time_frequency_view.channel_edit, "Tab4 通道")
            filter_config = self._build_tab4_filter_config()
            tf_options = self._read_tab4_time_frequency_options()
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.statusBar().showMessage("Tab4 时频图计算中 ...")
        worker = FunctionWorker(
            self._compute_time_frequency_for_view,
            self.current_data,
            self.current_metadata,
            channel,
            filter_config,
            tf_options,
            self.current_file_path,
        )
        self._active_workers.add(worker)
        worker.signals.result.connect(self._on_time_frequency_ready)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _compute_time_frequency_for_view(
        self,
        data: np.ndarray,
        metadata: FileMetadata,
        channel: int,
        filter_config: FilterConfig,
        tf_options: dict,
        file_path: Path | None,
    ) -> dict:
        trace = np.asarray(data[:, channel - 1], dtype=np.float32).reshape(-1, 1)
        if filter_config.is_enabled():
            trace = apply_filter(trace, metadata.sample_rate_hz, filter_config)
        trace_1d = np.asarray(trace[:, 0], dtype=np.float32)
        tf_result = compute_time_frequency_map(
            trace_1d,
            metadata.sample_rate_hz,
            window_seconds=tf_options["window_seconds"],
            overlap_percent=tf_options["overlap_percent"],
            mode=tf_options["mode"],
        )
        return {
            "metadata": metadata,
            "channel": channel,
            "trace": trace_1d,
            "tf_result": tf_result,
            "tf_options": tf_options,
            "file_path": file_path,
        }

    def _on_time_frequency_ready(self, payload: dict) -> None:
        if self.current_file_path != payload["file_path"]:
            self.statusBar().showMessage("时频图结果已过期：当前文件已切换。")
            return

        options = payload["tf_options"]
        self.time_frequency_view.set_result(
            metadata=payload["metadata"],
            channel=payload["channel"],
            trace=payload["trace"],
            tf_result=payload["tf_result"],
            frequency_range=options["frequency_range"],
            levels=options["levels"],
            auto_levels=options["auto_levels"],
            colormap_name=options["colormap_name"],
            value_scale=options["value_scale"],
            visible_window_seconds=options["visible_window_seconds"],
        )
        self.statusBar().showMessage(f"Tab4 已绘制时域/时频图：Ch {payload['channel']}")

    def generate_traffic_energy_image(self) -> None:
        """Build the Tab5 vehicle-energy image from the currently stitched files."""
        if self.current_data is None or self.current_metadata is None:
            self._show_error("请先在左侧选择一个或多个连续数据文件。")
            return
        if self.loaded_file_paths != self.current_file_paths:
            self._show_error("所选多文件仍在拼接读取中，请等待状态栏显示加载完成后再生成能量图。")
            return
        tab = self.traffic_trajectory_view
        display_mode = str(tab.display_mode_combo.currentData() or "energy")
        try:
            channel_start = int(tab.channel_start_edit.text().strip())
            channel_end = int(tab.channel_end_edit.text().strip())
            if channel_start > channel_end:
                channel_start, channel_end = channel_end, channel_start
            if channel_start < 1 or channel_end > self.current_metadata.points:
                raise ValueError(f"通道范围必须位于 1-{self.current_metadata.points}。")
            if display_mode == "energy":
                low_hz = float(tab.low_cut_edit.text().strip())
                high_hz = float(tab.high_cut_edit.text().strip())
                decimation = int(tab.decimation_edit.text().strip())
        except Exception as exc:
            self._show_error(str(exc) if str(exc) else "Tab5 参数格式无效。")
            return

        phase_slice = np.ascontiguousarray(
            self.current_data[:, channel_start - 1:channel_end], dtype=np.float32
        )
        # Use the exact same frames x points matrix already assembled for
        # Tab1.  Only slice columns here; never concatenate selected channels
        # or files again in Tab5.  Therefore axis 0 remains the full stitched
        # time dimension and axis 1 remains channel.
        source_files = [str(path) for path in self.loaded_file_paths]
        if display_mode == "phase":
            tab.set_phase(
                phase_slice,
                self.current_metadata.sample_rate_hz,
                channel_start,
                source_files,
                start_time=self.current_metadata.start_time,
            )
            self.statusBar().showMessage(
                f"Tab5 相位图完成：{phase_slice.shape[0]} 帧 × {phase_slice.shape[1]} 通道，"
                f"{len(source_files)} 个文件按时间拼接。"
            )
            return

        options = {
            "low_hz": low_hz,
            "high_hz": high_hz,
            "remove_common_mode": tab.common_mode_checkbox.isChecked(),
            "decimation": decimation,
            "balance": tab.balance_combo.currentText(),
        }
        tab.generate_button.setEnabled(False)
        self.statusBar().showMessage(
            f"Tab5 正在生成能量图：{len(self.current_file_paths) or 1} 个文件，"
            f"Ch {channel_start}-{channel_end} ..."
        )
        worker = FunctionWorker(
            build_vehicle_energy_image,
            phase_slice,
            self.current_metadata.sample_rate_hz,
            **options,
        )
        self._active_workers.add(worker)
        worker.signals.result.connect(
            lambda result, start=channel_start, files=source_files,
            expected_shape=(phase_slice.shape[0] // decimation, phase_slice.shape[1]): self._on_traffic_energy_ready(
                result, start, files, expected_shape
            )
        )
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(
            lambda: tab.generate_button.setEnabled(
                self.current_data is not None and self.loaded_file_paths == self.current_file_paths
            )
        )
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _on_traffic_energy_ready(
        self,
        result: tuple[np.ndarray, float],
        channel_start: int,
        source_files: list[str],
        expected_shape: tuple[int, int],
    ) -> None:
        if source_files != [str(path) for path in self.loaded_file_paths]:
            self.statusBar().showMessage("Tab5 能量图结果已过期：当前文件已切换。")
            return
        energy, energy_rate_hz = result
        if tuple(energy.shape) != tuple(expected_shape):
            self._show_error(
                "Tab5 能量图维度异常："
                f"期望 time×channel={expected_shape}，实际为 {tuple(energy.shape)}。"
            )
            return
        self.traffic_trajectory_view.set_energy(
            energy,
            energy_rate_hz,
            channel_start,
            source_files,
            start_time=self.current_metadata.start_time if self.current_metadata else None,
        )
        self.statusBar().showMessage(
            f"Tab5 能量图完成：{energy.shape[0]} × {energy.shape[1]} 像素，"
            f"等效采样率 {energy_rate_hz:g} Hz；可开始左键标记轨迹。"
        )

    def play_current_waveform(self) -> None:
        """Play the selected Tab1 waveform channel as scaled audio."""
        if winsound is None:
            self._show_error("当前运行环境不支持 winsound 播放。")
            return
        if self.current_metadata is None or self.current_data is None:
            self._show_error("No file loaded.")
            return

        try:
            start_point, end_point = self._read_point_range()
            waveform_point = self._read_waveform_point(start_point, end_point)
            waveform = np.asarray(self.current_data[:, waveform_point - 1], dtype=np.float32)
            audio_path, audio_rate = self._write_waveform_audio(waveform, self.current_metadata.sample_rate_hz)
        except Exception as exc:
            self._show_error(str(exc))
            return

        winsound.PlaySound(
            str(audio_path),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
        self.statusBar().showMessage(
            f"正在播放通道 {waveform_point} 波形 | 音频采样率：{audio_rate:g} Hz"
        )

    def _write_waveform_audio(self, waveform: np.ndarray, sample_rate_hz: float) -> tuple[Path, int]:
        audio, audio_rate = self._prepare_audio_samples(waveform, sample_rate_hz)
        pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype="<i2")

        temp_file = tempfile.NamedTemporaryFile(prefix="edas_waveform_", suffix=".wav", delete=False)
        audio_path = Path(temp_file.name)
        temp_file.close()
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio_rate)
            wav_file.writeframes(pcm.tobytes())

        self._audio_temp_paths.append(audio_path)
        self._cleanup_old_audio_files()
        return audio_path, audio_rate

    def _prepare_audio_samples(self, waveform: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, int]:
        if sample_rate_hz <= 0:
            raise ValueError("采样率必须大于 0。")

        trace = np.asarray(waveform, dtype=np.float32)
        if trace.size < 2:
            raise ValueError("波形长度不足，无法播放。")

        finite_values = trace[np.isfinite(trace)]
        if finite_values.size == 0:
            raise ValueError("波形不包含有效数值。")

        trace = np.nan_to_num(trace, nan=float(np.nanmedian(finite_values)), posinf=0.0, neginf=0.0)
        trace = trace - float(np.nanmedian(trace))
        scale = float(np.nanpercentile(np.abs(trace), 99.0))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanmax(np.abs(trace)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("波形幅值过小，无法播放。")

        audio = np.clip(trace / scale * 1.8, -1.0, 1.0) * 0.96
        source_rate = float(sample_rate_hz)
        audio_rate = int(min(max(round(source_rate), 8000), 44100))
        if not np.isclose(audio_rate, source_rate):
            duration_s = trace.size / source_rate
            target_count = max(2, int(round(duration_s * audio_rate)))
            source_x = np.arange(audio.size, dtype=np.float64) / source_rate
            target_x = np.arange(target_count, dtype=np.float64) / audio_rate
            audio = np.interp(target_x, source_x, audio).astype(np.float32)

        return np.asarray(audio, dtype=np.float32), audio_rate

    def _cleanup_old_audio_files(self) -> None:
        while len(self._audio_temp_paths) > 6:
            old_path = self._audio_temp_paths.pop(0)
            try:
                old_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _parameter_widgets(self) -> dict[str, QWidget]:
        panel = self.control_panel
        tab2 = self.channel_analysis_view
        tab3 = self.fk_spectrum_view
        tab4 = self.time_frequency_view
        tab5 = self.traffic_trajectory_view
        return {
            "data/browser_folder": self.file_browser.path_edit,
            "tab1/point_start": panel.point_start_edit,
            "tab1/point_end": panel.point_end_edit,
            "tab1/vmin": panel.vmin_edit,
            "tab1/vmax": panel.vmax_edit,
            "tab1/colormap": panel.colormap_combo,
            "tab1/filter_type": panel.filter_type_combo,
            "tab1/low_cut": panel.low_cut_edit,
            "tab1/high_cut": panel.high_cut_edit,
            "tab1/waveform_point": panel.waveform_point_edit,
            "tab1/fk_spacing": panel.fk_spacing_edit,
            "tab1/fk_vmin": panel.fk_vmin_edit,
            "tab1/fk_vmax": panel.fk_vmax_edit,
            "tab1/export_folder": panel.export_path_edit,
            "tab1/save_filtered": panel.save_filtered_checkbox,
            "tab1/save_as_txt": panel.save_as_txt_checkbox,
            "tab2/channels": tab2.channels_edit,
            "tab2/psd_nperseg": tab2.psd_nperseg_edit,
            "tab2/psd_overlap": tab2.psd_overlap_edit,
            "tab2/psd_fmin": tab2.psd_fmin_edit,
            "tab2/psd_fmax": tab2.psd_fmax_edit,
            "tab2/psd_ymin": tab2.psd_ymin_edit,
            "tab2/psd_ymax": tab2.psd_ymax_edit,
            "tab3/channel_range": tab3.channel_range_edit,
            "tab3/spacing": tab3.spacing_edit,
            "tab3/spatial_taper": tab3.spatial_taper_edit,
            "tab3/temporal_taper": tab3.temporal_taper_edit,
            "tab3/nfft_channels": tab3.nfft_channels_edit,
            "tab3/nfft_samples": tab3.nfft_samples_edit,
            "tab3/remove_background": tab3.remove_bg_checkbox,
            "tab3/fmin": tab3.fmin_edit,
            "tab3/fmax": tab3.fmax_edit,
            "tab3/kmin": tab3.kmin_edit,
            "tab3/kmax": tab3.kmax_edit,
            "tab3/vmin_db": tab3.vmin_db_edit,
            "tab3/vmax_db": tab3.vmax_db_edit,
            "tab4/channel": tab4.channel_edit,
            "tab4/filter_type": tab4.filter_type_combo,
            "tab4/low_cut": tab4.low_cut_edit,
            "tab4/high_cut": tab4.high_cut_edit,
            "tab4/mode": tab4.tf_mode_combo,
            "tab4/value_scale": tab4.tf_scale_combo,
            "tab4/window_seconds": tab4.window_seconds_edit,
            "tab4/visible_window_seconds": tab4.visible_window_seconds_edit,
            "tab4/overlap_percent": tab4.overlap_percent_edit,
            "tab4/fmin": tab4.fmin_edit,
            "tab4/fmax": tab4.fmax_edit,
            "tab4/vmin": tab4.vmin_edit,
            "tab4/vmax": tab4.vmax_edit,
            "tab4/colormap": tab4.colormap_combo,
            "tab4/auto_color": tab4.auto_color_checkbox,
            "tab5/channel_start": tab5.channel_start_edit,
            "tab5/channel_end": tab5.channel_end_edit,
            "tab5/display_mode": tab5.display_mode_combo,
            "tab5/low_cut": tab5.low_cut_edit,
            "tab5/high_cut": tab5.high_cut_edit,
            "tab5/decimation": tab5.decimation_edit,
            "tab5/common_mode": tab5.common_mode_checkbox,
            "tab5/balance": tab5.balance_combo,
            "tab5/colormap": tab5.colormap_combo,
            "tab5/vmin": tab5.vmin_edit,
            "tab5/vmax": tab5.vmax_edit,
            "tab5/auto_color": tab5.auto_color_checkbox,
        }

    def _connect_parameter_autosave(self) -> None:
        for widget in self._parameter_widgets().values():
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._schedule_parameter_save)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._schedule_parameter_save)
            else:
                widget.textEdited.connect(self._schedule_parameter_save)
        # Wheel stepping changes the waveform point without a textEdited event.
        self.control_panel.waveform_point_edit.wheelStepped.connect(self._schedule_parameter_save)

    def _schedule_parameter_save(self, *_args) -> None:
        self._parameter_save_timer.start()

    def _save_parameters(self) -> None:
        self._parameter_store.save_widgets(self._parameter_widgets())

    def _restore_persisted_parameters(self) -> set[str]:
        """Load the most recent parameter values into Tab1/Tab2/Tab3 controls."""
        try:
            return self._parameter_store.restore_widgets(self._parameter_widgets())
        except Exception:
            return {}

    def _parse_channel_list(self, text: str) -> list[int]:
        if self.current_metadata is None:
            raise ValueError("No file loaded.")
        parts = [part for part in re.split(r"[,，、;\s]+", text.strip()) if part]
        if not parts:
            raise ValueError("请输入通道号，例如 5,10,15。")

        channels = []
        for part in parts:
            try:
                channel = int(part)
            except ValueError as exc:
                raise ValueError(f"无效通道号：{part}") from exc
            if channel < 1 or channel > self.current_metadata.points:
                raise ValueError(f"通道 {channel} 超出有效范围 1-{self.current_metadata.points}。")
            channels.append(channel)
        return channels

    def _parse_channel_range(self, text: str) -> tuple[int, int]:
        if self.current_metadata is None:
            raise ValueError("No file loaded.")

        normalized = text.strip().replace("：", ":").replace("－", "-").replace("—", "-")
        match = re.fullmatch(r"(\d+)\s*[-:~]\s*(\d+)", normalized)
        if match is None:
            raise ValueError("通道范围格式应为 5-100。")

        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            start, end = end, start
        if start < 1 or end > self.current_metadata.points:
            raise ValueError(f"通道范围必须位于 1-{self.current_metadata.points}。")
        return start, end

    def _read_tab2_psd_options(self) -> dict:
        nperseg = self._read_optional_int(self.channel_analysis_view.psd_nperseg_edit, "PSD窗长")
        if nperseg is not None and nperseg < 8:
            raise ValueError("PSD窗长必须大于等于 8，或填写 auto。")

        overlap_percent = self._read_optional_float(
            self.channel_analysis_view.psd_overlap_edit,
            "PSD重叠比例",
            default=50.0,
        )
        if overlap_percent < 0 or overlap_percent >= 100:
            raise ValueError("PSD重叠比例必须位于 [0, 100)%。")

        psd_x_range = self._read_optional_range(
            self.channel_analysis_view.psd_fmin_edit,
            self.channel_analysis_view.psd_fmax_edit,
            "PSD频率范围",
            positive_or_zero=True,
        )
        psd_y_range = self._read_optional_range(
            self.channel_analysis_view.psd_ymin_edit,
            self.channel_analysis_view.psd_ymax_edit,
            "PSD幅值范围",
            positive_or_zero=False,
        )
        return {
            "psd_nperseg": nperseg,
            "psd_overlap_percent": overlap_percent,
            "psd_x_range": psd_x_range,
            "psd_y_range": psd_y_range,
        }

    def _read_tab3_fk_options(self) -> dict:
        spatial_taper = self._read_optional_float(
            self.fk_spectrum_view.spatial_taper_edit,
            "FK空间taper",
            default=0.0,
        )
        temporal_taper = self._read_optional_float(
            self.fk_spectrum_view.temporal_taper_edit,
            "FK时间taper",
            default=0.05,
        )
        for label, value in [("FK空间taper", spatial_taper), ("FK时间taper", temporal_taper)]:
            if value < 0 or value > 1:
                raise ValueError(f"{label} 必须位于 0-1。")

        nfft_channels = self._read_optional_int(self.fk_spectrum_view.nfft_channels_edit, "FK空间NFFT")
        nfft_samples = self._read_optional_int(self.fk_spectrum_view.nfft_samples_edit, "FK时间NFFT")
        if nfft_channels is not None and nfft_channels < 2:
            raise ValueError("FK空间NFFT必须大于等于 2，或填写 auto。")
        if nfft_samples is not None and nfft_samples < 2:
            raise ValueError("FK时间NFFT必须大于等于 2，或填写 auto。")

        frequency_range = self._read_optional_range(
            self.fk_spectrum_view.fmin_edit,
            self.fk_spectrum_view.fmax_edit,
            "FK频率范围",
            positive_or_zero=True,
        )
        wavenumber_range = self._read_optional_range(
            self.fk_spectrum_view.kmin_edit,
            self.fk_spectrum_view.kmax_edit,
            "FK波数范围",
            positive_or_zero=False,
        )
        return {
            "taper_fraction": (spatial_taper, temporal_taper),
            "remove_background": self.fk_spectrum_view.remove_bg_checkbox.isChecked(),
            "nfft_channels": nfft_channels,
            "nfft_samples": nfft_samples,
            "frequency_range": frequency_range,
            "wavenumber_range": wavenumber_range,
        }

    def _build_tab4_filter_config(self) -> FilterConfig:
        tab4 = self.time_frequency_view
        filter_type = tab4.filter_type_combo.currentText().strip().lower()
        low_text = tab4.low_cut_edit.text().strip()
        high_text = tab4.high_cut_edit.text().strip()
        return FilterConfig(
            filter_type=filter_type,
            low_cut_hz=float(low_text) if low_text else None,
            high_cut_hz=float(high_text) if high_text else None,
            order=4,
        )

    def _read_tab4_time_frequency_options(self) -> dict:
        tab4 = self.time_frequency_view
        window_seconds = self._read_positive_float(tab4.window_seconds_edit, "Tab4 时频窗长")
        overlap_percent = self._read_optional_float(
            tab4.overlap_percent_edit,
            "Tab4 时频重叠比例",
            default=50.0,
        )
        if overlap_percent < 0 or overlap_percent >= 100:
            raise ValueError("Tab4 时频重叠比例必须位于 [0, 100)%。")

        visible_window_seconds = self._read_optional_float(
            tab4.visible_window_seconds_edit,
            "Tab4 显示窗口长度",
            default=None,
        )
        if visible_window_seconds is not None and visible_window_seconds <= 0:
            raise ValueError("Tab4 显示窗口长度必须大于 0，或填写 auto/留空。")

        frequency_range = self._read_optional_range(
            tab4.fmin_edit,
            tab4.fmax_edit,
            "Tab4 频率范围",
            positive_or_zero=False,
        )
        if frequency_range[0] is not None and (frequency_range[0] <= 0 or frequency_range[1] <= 0):
            raise ValueError("Tab4 对数频率显示范围必须大于 0。")

        levels = self._read_optional_range(
            tab4.vmin_edit,
            tab4.vmax_edit,
            "Tab4 颜色范围",
            positive_or_zero=False,
        )
        auto_levels = tab4.auto_color_checkbox.isChecked()
        if not auto_levels and (levels[0] is None or levels[1] is None):
            raise ValueError("关闭颜色自动时，需要同时填写 Tab4 VMin 和 VMax。")

        return {
            "mode": tab4.tf_mode_combo.currentText().strip().lower(),
            "value_scale": tab4.tf_scale_combo.currentText().strip().lower(),
            "window_seconds": window_seconds,
            "visible_window_seconds": visible_window_seconds,
            "overlap_percent": overlap_percent,
            "frequency_range": frequency_range,
            "levels": levels,
            "auto_levels": auto_levels,
            "colormap_name": tab4.colormap_combo.currentText().strip().lower() or "jet",
        }

    def _read_tab1_fk_parameters(self) -> tuple[float, float, float]:
        spacing_m = self._read_positive_float(self.control_panel.fk_spacing_edit, "FK 通道间距")
        vmin_mps = self._read_positive_float(self.control_panel.fk_vmin_edit, "FK VMin")
        vmax_mps = self._read_positive_float(self.control_panel.fk_vmax_edit, "FK VMax")
        if np.isclose(vmin_mps, vmax_mps):
            raise ValueError("FK VMin 和 VMax 不能相等。")
        return spacing_m, vmin_mps, vmax_mps

    def _read_channel_number(self, line_edit: QLineEdit, name: str) -> int:
        if self.current_metadata is None:
            raise ValueError("No file loaded.")
        text = line_edit.text().strip()
        if not text:
            raise ValueError(f"{name}不能为空。")
        try:
            channel = int(text)
        except ValueError as exc:
            raise ValueError(f"{name}必须是整数。") from exc
        if channel < 1 or channel > self.current_metadata.points:
            raise ValueError(f"{name}必须位于 1-{self.current_metadata.points}。")
        return channel

    def _read_fk_db_levels(self) -> tuple[float, float]:
        vmin_db = self._read_float(self.fk_spectrum_view.vmin_db_edit, "f-k VMin(dB)")
        vmax_db = self._read_float(self.fk_spectrum_view.vmax_db_edit, "f-k VMax(dB)")
        if vmin_db >= vmax_db:
            raise ValueError("f-k VMin(dB) 必须小于 VMax(dB)。")
        return vmin_db, vmax_db

    def _read_positive_float(self, line_edit: QLineEdit, name: str) -> float:
        value = self._read_float(line_edit, name)
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0。")
        return value

    def _read_float(self, line_edit: QLineEdit, name: str) -> float:
        text = line_edit.text().strip()
        if not text:
            raise ValueError(f"{name} 不能为空。")
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字。") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} 必须是有限数字。")
        return value

    def _read_optional_float(
        self,
        line_edit: QLineEdit,
        name: str,
        *,
        default: float | None = None,
    ) -> float | None:
        text = line_edit.text().strip().lower()
        if not text or text == "auto":
            return default
        return self._read_float(line_edit, name)

    def _read_optional_int(self, line_edit: QLineEdit, name: str) -> int | None:
        text = line_edit.text().strip().lower()
        if not text or text == "auto":
            return None
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数或 auto。") from exc
        return value

    def _read_optional_range(
        self,
        min_edit: QLineEdit,
        max_edit: QLineEdit,
        name: str,
        *,
        positive_or_zero: bool,
    ) -> tuple[float | None, float | None]:
        min_text = min_edit.text().strip().lower()
        max_text = max_edit.text().strip().lower()
        if (not min_text or min_text == "auto") and (not max_text or max_text == "auto"):
            return None, None
        if not min_text or min_text == "auto" or not max_text or max_text == "auto":
            raise ValueError(f"{name} 需要同时填写最小值和最大值，或都留空/auto。")

        range_min = self._read_float(min_edit, f"{name}最小值")
        range_max = self._read_float(max_edit, f"{name}最大值")
        if positive_or_zero and (range_min < 0 or range_max < 0):
            raise ValueError(f"{name} 不能小于 0。")
        if range_min >= range_max:
            raise ValueError(f"{name} 最小值必须小于最大值。")
        return range_min, range_max

    def _format_preprocess_steps(self) -> str:
        if not self.current_preprocess_steps:
            return "无"
        return " -> ".join(self.PREPROCESS_LABELS.get(step, step) for step in self.current_preprocess_steps)

    def _render_current_data(self) -> None:
        if self.current_metadata is None or self.current_data is None:
            return

        start_point, end_point = self._read_point_range()
        self._sync_waveform_point_bounds()
        waveform_point = self._read_waveform_point(start_point, end_point)
        vmin, vmax = self._read_color_range()
        colormap_name = self.control_panel.colormap_combo.currentText().strip().lower() or "seismic"
        self.space_time_view.set_data(
            metadata=self.current_metadata,
            frame_data=self.current_data,
            start_point=start_point,
            end_point=end_point,
            waveform_point=waveform_point,
            vmin=vmin,
            vmax=vmax,
            colormap_name=colormap_name,
        )
        self._reset_scrollbars_for_current_view()
        self._update_status_text()

    def reset_controls(self) -> None:
        """Restore display and filter controls to default values."""
        self._point_end_manual_override = False
        selected_file = self._selected_file_path() or self.current_file_path
        metadata = self.current_metadata if self.current_metadata and self.current_metadata.file_path == selected_file else None

        self.control_panel.point_start_edit.setText("1")
        self.control_panel.point_end_edit.setText(self._default_point_end_value(file_path=selected_file, metadata=metadata))
        self.control_panel.vmin_edit.setText("-0.4")
        self.control_panel.vmax_edit.setText("0.4")
        self.control_panel.colormap_combo.setCurrentText("seismic")
        self.control_panel.filter_type_combo.setCurrentText("highpass")
        self.control_panel.low_cut_edit.setText("10")
        self.control_panel.high_cut_edit.clear()
        self.control_panel.save_filtered_checkbox.setChecked(False)
        self.control_panel.save_as_txt_checkbox.setChecked(False)
        self.control_panel.waveform_point_edit.setText(self.control_panel.point_start_edit.text().strip() or "1")
        self.control_panel.fk_spacing_edit.setText("5")
        self.control_panel.fk_vmin_edit.setText("800")
        self.control_panel.fk_vmax_edit.setText("1200")
        self._sync_waveform_point_bounds()
        self._schedule_parameter_save()

        if selected_file is not None:
            try:
                self._load_file(selected_file, self._build_filter_config())
            except Exception as exc:
                self._show_error(str(exc))

    def _build_filter_config(self) -> FilterConfig:
        filter_type = self.control_panel.filter_type_combo.currentText().strip().lower()
        low_text = self.control_panel.low_cut_edit.text().strip()
        high_text = self.control_panel.high_cut_edit.text().strip()

        return FilterConfig(
            filter_type=filter_type,
            low_cut_hz=float(low_text) if low_text else None,
            high_cut_hz=float(high_text) if high_text else None,
            order=4,
        )

    def _read_point_range(self) -> tuple[int, int]:
        if self.current_metadata is None:
            raise ValueError("No file loaded.")

        start_point = int(self.control_panel.point_start_edit.text().strip())
        end_point = int(self.control_panel.point_end_edit.text().strip())
        if start_point < 1 or end_point > self.current_metadata.points or start_point > end_point:
            raise ValueError("Invalid point range.")
        return start_point, end_point

    def _read_waveform_point(self, start_point: int, end_point: int) -> int:
        point_text = self.control_panel.waveform_point_edit.text().strip()
        if not point_text:
            self.control_panel.waveform_point_edit.setText(str(start_point))
            return start_point

        try:
            waveform_point = int(point_text)
        except ValueError as exc:
            raise ValueError("指定位置点必须是整数。") from exc

        if waveform_point < start_point or waveform_point > end_point:
            raise ValueError(f"指定位置点必须位于 Point Start 和 Point End 之间（{start_point}-{end_point}）。")

        return waveform_point

    def _read_color_range(self) -> tuple[float | None, float | None]:
        vmin_text = self.control_panel.vmin_edit.text().strip()
        vmax_text = self.control_panel.vmax_edit.text().strip()
        vmin = float(vmin_text) if vmin_text else None
        vmax = float(vmax_text) if vmax_text else None
        if (vmin is None) ^ (vmax is None):
            raise ValueError("Both VMin and VMax must be provided together.")
        if vmin is not None and vmax is not None and vmin >= vmax:
            raise ValueError("VMin must be smaller than VMax.")
        return vmin, vmax

    def _default_point_end_value(
        self,
        file_path: Path | None = None,
        metadata: FileMetadata | None = None,
    ) -> str:
        points = metadata.points if metadata is not None else None
        if points is None and file_path is not None:
            points = infer_points_from_filename(file_path)
        if points is None:
            return str(self.AUTO_POINT_CAP)
        return str(max(1, min(points, self.AUTO_POINT_CAP)))

    def _apply_auto_point_limit(
        self,
        file_path: Path | None = None,
        metadata: FileMetadata | None = None,
    ) -> None:
        if self._point_end_manual_override:
            points = metadata.points if metadata is not None else None
            if points is None and file_path is not None:
                points = infer_points_from_filename(file_path)
            self._clamp_point_range_to_points(points)
            return
        self.control_panel.point_end_edit.setText(self._default_point_end_value(file_path=file_path, metadata=metadata))
        self._sync_waveform_point_bounds()

    def _clamp_point_range_to_points(self, points: int | None) -> None:
        """Keep the user-owned point range across files; clamp only out-of-range values."""
        if points is None or points < 1:
            return
        start_text = self.control_panel.point_start_edit.text().strip()
        end_text = self.control_panel.point_end_edit.text().strip()
        try:
            start_value = int(start_text)
            end_value = int(end_text)
        except ValueError:
            return
        new_start = max(1, min(start_value, points))
        new_end = max(new_start, min(end_value, points))
        if new_start != start_value:
            self.control_panel.point_start_edit.setText(str(new_start))
        if new_end != end_value:
            self.control_panel.point_end_edit.setText(str(new_end))

    def _sync_waveform_point_bounds(self) -> None:
        start_text = self.control_panel.point_start_edit.text().strip()
        end_text = self.control_panel.point_end_edit.text().strip()
        try:
            start_point = int(start_text)
            end_point = int(end_text)
        except ValueError:
            return
        if start_point > end_point:
            return
        self.control_panel.set_waveform_point_bounds(start_point, end_point)

    def _on_point_end_text_edited(self, text: str) -> None:
        self._point_end_manual_override = bool(text.strip())
        self._sync_waveform_point_bounds()

    def _on_waveform_point_wheel_stepped(self, _value: int) -> None:
        self._on_waveform_point_changed()

    def _on_waveform_point_changed(self) -> None:
        if self.current_metadata is None:
            return

        self._sync_waveform_point_bounds()

        try:
            start_point, end_point = self._read_point_range()
            waveform_point = self._read_waveform_point(start_point, end_point)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.control_panel.waveform_point_edit.setText(str(waveform_point))
        self.space_time_view.update_waveform_point(waveform_point)

    def _reset_scrollbars_for_current_view(self) -> None:
        if self.current_metadata is None:
            return
        self._is_syncing_scrollbars = True
        try:
            self.horizontal_scroll.setRange(0, 1000)
            self.horizontal_scroll.setPageStep(1000)
            self.horizontal_scroll.setValue(0)
            self.vertical_scroll.setRange(0, 1000)
            self.vertical_scroll.setPageStep(1000)
            self.vertical_scroll.setValue(0)
        finally:
            self._is_syncing_scrollbars = False

    def _sync_scrollbars_to_view(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        if self.current_metadata is None or self._is_syncing_scrollbars:
            return

        self._is_syncing_scrollbars = True
        try:
            duration = max(self.current_metadata.duration_s, 1e-9)
            x_span = max(x_range[1] - x_range[0], 1e-9)
            x_max_start = max(duration - x_span, 1e-9)
            x_start = min(max(x_range[0], 0.0), x_max_start)

            point_count = max(self.current_metadata.points, 1)
            y_span = max(y_range[1] - y_range[0], 1.0)
            y_min = 1.0
            y_max_start = max(point_count + 1 - y_span, y_min)
            y_start = min(max(y_range[0], y_min), y_max_start)

            x_page = max(1, int(round(1000 * (x_span / duration))))
            x_value = int(round(1000 * (x_start / max(x_max_start, 1e-9)))) if x_max_start > 1e-9 else 0
            self.horizontal_scroll.setPageStep(min(x_page, 1000))
            self.horizontal_scroll.setValue(min(max(x_value, 0), 1000))

            y_page = max(1, int(round(1000 * (y_span / point_count))))
            y_value = int(
                round(1000 * ((y_start - y_min) / max(y_max_start - y_min, 1e-9)))
            ) if y_max_start > y_min else 0
            self.vertical_scroll.setPageStep(min(y_page, 1000))
            self.vertical_scroll.setValue(min(max(y_value, 0), 1000))
        finally:
            self._is_syncing_scrollbars = False

    def _on_horizontal_scroll(self, value: int) -> None:
        if self._is_syncing_scrollbars or self.current_metadata is None:
            return
        x_range, _ = self.space_time_view.current_ranges()
        span = max(x_range[1] - x_range[0], 1e-9)
        max_start = max(self.current_metadata.duration_s - span, 0.0)
        start = (value / 1000.0) * max_start if max_start > 0 else 0.0
        self.space_time_view.set_x_range(start, start + span)

    def _on_vertical_scroll(self, value: int) -> None:
        if self._is_syncing_scrollbars or self.current_metadata is None:
            return
        _, y_range = self.space_time_view.current_ranges()
        span = max(y_range[1] - y_range[0], 1.0)
        max_start = max((self.current_metadata.points + 1) - span, 1.0)
        start = 1.0 + (value / 1000.0) * max(max_start - 1.0, 0.0)
        self.space_time_view.set_y_range(start, start + span)

    def _on_colormap_changed(self, colormap_name: str) -> None:
        if self.current_data is None:
            return
        self.space_time_view.update_colormap(colormap_name.strip().lower() or "seismic")

    def _on_zoom_out_2x(self) -> None:
        self.space_time_view.zoom_out_horizontal(2.0)

    def _step_horizontal_scroll(self, direction: int) -> None:
        if self.current_metadata is None:
            return

        x_range, _ = self.space_time_view.current_ranges()
        span = max(x_range[1] - x_range[0], 1e-9)
        delta = span * self.HORIZONTAL_KEY_STEP_RATIO * direction
        max_start = max(self.current_metadata.duration_s - span, 0.0)
        new_start = min(max(x_range[0] + delta, 0.0), max_start)
        self.space_time_view.set_x_range(new_start, new_start + span)

    def _on_worker_finished(self, worker: FunctionWorker) -> None:
        self._active_workers.discard(worker)

    def _format_sample_rate(self, sample_rate_hz: float) -> str:
        return f"{sample_rate_hz:g}"

    def _build_loading_status_text(self, metadata: FileMetadata) -> str:
        file_count = len(self.current_file_paths) if self.current_file_paths else 1
        file_label = metadata.file_path.name if file_count <= 1 else f"{file_count} files stitched"
        return (
            f"Loading {file_label} | "
            f"采样率：{self._format_sample_rate(metadata.sample_rate_hz)}Hz | "
            f"位置点数：{metadata.points} points | "
            f"数据时长：{metadata.duration_s:.2f} s"
        )

    def _update_status_text(self) -> None:
        if self.current_metadata is None or self.current_file_path is None:
            return
        file_count = len(self.current_file_paths) if self.current_file_paths else 1
        file_label = self.current_file_path.name if file_count <= 1 else f"{file_count} 个文件按时间拼接"
        self.statusBar().showMessage(
            (
                f"当前文件：{file_label} | "
                f"采样率：{self._format_sample_rate(self.current_metadata.sample_rate_hz)}Hz | "
                f"位置点数：{self.current_metadata.points} points | "
                f"数据时长：{self.current_metadata.duration_s:.2f} s | "
                f"预处理：{self._format_preprocess_steps()}"
            )
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def _apply_window_icons(self) -> None:
        logo_path = self._resolve_logo_path()
        if logo_path is None:
            return

        icon = QIcon(str(logo_path))
        if icon.isNull():
            return

        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _resolve_logo_path(self) -> Optional[Path]:
        candidates = [
            asset_path("eDASread_logo.png"),
            Path(__file__).resolve().parents[2] / "assets" / "eDASread_logo.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def export_current_view(self) -> None:
        if self.current_metadata is None or self.current_file_path is None:
            self._show_error("No file loaded.")
            return

        export_folder_text = self.control_panel.export_path_edit.text().strip()
        if not export_folder_text:
            self._show_error("Please choose an export folder.")
            return

        export_folder = Path(export_folder_text)
        export_folder.mkdir(parents=True, exist_ok=True)

        save_filtered = self.control_panel.save_filtered_checkbox.isChecked()
        save_as_txt = self.control_panel.save_as_txt_checkbox.isChecked()

        self.statusBar().showMessage("Exporting current view ...")
        worker = FunctionWorker(
            self._export_current_view_data,
            export_folder,
            save_filtered,
            save_as_txt,
        )
        self._active_workers.add(worker)
        worker.signals.result.connect(self._on_export_finished)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _export_current_view_data(
        self,
        export_folder: Path,
        save_processed: bool,
        save_as_txt: bool,
    ) -> dict:
        if self.current_metadata is None:
            raise ValueError("No file loaded.")

        metadata = self.current_metadata
        x_range, y_range = self.space_time_view.current_ranges()
        frame_start = max(0, int(np.floor(x_range[0] * metadata.sample_rate_hz)))
        frame_end = min(metadata.frame_count, int(np.ceil(x_range[1] * metadata.sample_rate_hz)))
        point_start = max(1, int(np.floor(y_range[0])))
        point_end = min(metadata.points, int(np.ceil(y_range[1]) - 1))

        if frame_end <= frame_start or point_end < point_start:
            raise ValueError("Current view is empty and cannot be exported.")

        point_slice = slice(point_start - 1, point_end)
        frame_slice = slice(frame_start, frame_end)

        if save_processed or len(self.current_file_paths) > 1:
            if self.current_data is None:
                raise ValueError("No processed data loaded.")
            export_phase_slice = np.ascontiguousarray(self.current_data[frame_slice, point_slice], dtype=np.float32)
            export_raw_slice = None
        else:
            raw_data = read_bin_memmap(metadata)
            export_raw_slice = np.ascontiguousarray(raw_data[frame_slice, point_slice], dtype=np.int32)
            export_phase_slice = convert_raw_to_radians(export_raw_slice)

        filename = build_export_filename(
            metadata=metadata,
            point_count=point_end - point_start + 1,
            time_offset_s=frame_start / metadata.sample_rate_hz,
            suffix=".txt" if save_as_txt else ".bin",
            preserve_leading_token=save_as_txt,
        )
        output_path = export_folder / filename
        if save_as_txt:
            write_phase_txt(output_path, export_phase_slice)
            export_dtype = str(export_phase_slice.dtype)
            export_format = "txt"
        else:
            if export_raw_slice is None:
                export_slice = np.ascontiguousarray(convert_radians_to_raw(export_phase_slice))
            else:
                export_slice = export_raw_slice
            export_slice.tofile(output_path)
            export_dtype = str(export_slice.dtype)
            export_format = "bin"
        return {
            "output_path": output_path,
            "frame_count": export_phase_slice.shape[0],
            "point_count": export_phase_slice.shape[1],
            "filtered": save_processed,
            "dtype": export_dtype,
            "format": export_format,
        }

    def _on_export_finished(self, payload: dict) -> None:
        filtered_text = "filtered" if payload["filtered"] else "unfiltered"
        self.statusBar().showMessage(
            (
                f"Exported {filtered_text} {payload['format']} view to {payload['output_path']} | "
                f"Frames: {payload['frame_count']} | Points: {payload['point_count']} | "
                f"Type: {payload['dtype']}"
            )
        )


