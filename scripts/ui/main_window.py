"""Main window implementation for the FIPread application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QThreadPool, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from scripts.core.cache import LruByteCache
from scripts.core.models import FileMetadata, FilterConfig
from scripts.core.resources import asset_path
from scripts.io.bin_reader import (
    build_export_filename,
    convert_radians_to_raw,
    convert_raw_to_radians,
    infer_points_from_filename,
    inspect_bin_file,
    list_bin_files,
    read_bin_memmap,
)
from scripts.io.text_export import write_phase_txt
from scripts.processing.filters import apply_filter
from scripts.processing.view_data import prepare_image_matrix
from scripts.render.space_time_view import SpaceTimeView
from scripts.ui.control_panel import ControlPanel
from scripts.ui.file_browser import FileBrowserPanel
from scripts.workers.tasks import FunctionWorker


class MainWindow(QMainWindow):
    """Main desktop window coordinating file loading and plotting."""

    CACHE_MAX_BYTES = 1_000_000_000
    HORIZONTAL_KEY_STEP_RATIO = 0.4
    AUTO_POINT_CAP = 100
    WINDOW_TITLE = "分布式光纤传感数据回放软件(eDASread)"
    BUTTON_STYLE = """
        QPushButton {
            background-color: #2f6db2;
            color: white;
            border: 1px solid #224f82;
            border-radius: 4px;
            padding: 5px 12px;
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
    """

    def __init__(self, default_directory: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1500, 900)
        self._apply_window_icons()

        self.thread_pool = QThreadPool.globalInstance()
        self.processed_cache = LruByteCache(max_bytes=self.CACHE_MAX_BYTES)
        self.file_paths: list[Path] = []
        self.current_page = 0
        self.current_metadata: Optional[FileMetadata] = None
        self.current_data: Optional[np.ndarray] = None
        self.current_file_path: Optional[Path] = None
        self._active_workers: set[FunctionWorker] = set()
        self._point_end_manual_override = False

        self.file_browser = FileBrowserPanel()
        self.control_panel = ControlPanel()
        self.space_time_view = SpaceTimeView()
        self.horizontal_scroll = QScrollBar(Qt.Horizontal)
        self.vertical_scroll = QScrollBar(Qt.Vertical)
        self.title_label = QLabel(self.WINDOW_TITLE)
        self.title_logo_label = QLabel()

        self._is_syncing_scrollbars = False
        self._directory_load_timer = QTimer(self)
        self._directory_load_timer.setSingleShot(True)
        self._directory_load_timer.setInterval(400)
        self._directory_load_timer.timeout.connect(self._load_directory_from_timer)
        self._build_layout()
        self._connect_signals()
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

        if default_directory is not None:
            self.file_browser.path_edit.setText(str(default_directory))
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

        self.title_logo_label.setFixedSize(54, 54)
        self.title_logo_label.setAlignment(Qt.AlignCenter)
        logo_path = self._resolve_logo_path()
        if logo_path is not None:
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                self.title_logo_label.setPixmap(
                    pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(0, 0, 0, 0)
        title_row_layout.setSpacing(6)
        title_row_layout.addWidget(self.title_logo_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        title_row_layout.addStretch(1)
        title_row_layout.addWidget(self.title_label, 0, Qt.AlignCenter)
        title_row_layout.addStretch(1)
        right_spacer = QWidget()
        right_spacer.setFixedSize(54, 54)
        title_row_layout.addWidget(right_spacer, 0, Qt.AlignRight | Qt.AlignVCenter)

        plot_panel = QWidget()
        plot_layout = QHBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        plot_column = QVBoxLayout()
        plot_column.setContentsMargins(0, 0, 0, 0)
        plot_column.setSpacing(6)
        plot_column.addWidget(self.control_panel)
        plot_column.addWidget(self.space_time_view, 1)
        plot_column.addWidget(self.horizontal_scroll)

        plot_layout.addLayout(plot_column, 1)
        plot_layout.addWidget(self.vertical_scroll)

        splitter = QSplitter()
        splitter.addWidget(self.file_browser)
        splitter.addWidget(plot_panel)
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
        self.file_browser.previous_button.clicked.connect(self.show_previous_page)
        self.file_browser.next_button.clicked.connect(self.show_next_page)
        self.file_browser.jump_button.clicked.connect(self.jump_to_page)
        self.file_browser.page_jump_edit.returnPressed.connect(self.jump_to_page)
        self.file_browser.file_list.itemSelectionChanged.connect(self._on_file_selected)

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
            self._step_horizontal_scroll(direction)
            return True

        return super().eventFilter(watched, event)

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

    def _clear_file_listing(self) -> None:
        self.file_browser.file_list.clear()
        self.file_browser.page_label.setText("Page 0 / 0")
        self.file_browser.previous_button.setEnabled(False)
        self.file_browser.next_button.setEnabled(False)
        self.file_browser.page_jump_edit.setPlaceholderText("page / home / end")

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

    def _on_file_selected(self) -> None:
        file_path = self._selected_file_path()
        if file_path is None:
            return

        self._apply_auto_point_limit(file_path=file_path)
        try:
            self._load_file(file_path, self._build_filter_config())
        except Exception as exc:
            self._show_error(str(exc))

    def _load_file(self, file_path: Path, filter_config: FilterConfig) -> None:
        metadata = inspect_bin_file(file_path)
        self._apply_auto_point_limit(metadata=metadata)
        self.statusBar().showMessage(self._build_loading_status_text(metadata))
        worker = FunctionWorker(self._load_file_data, metadata, filter_config)
        self._active_workers.add(worker)
        worker.signals.result.connect(self._on_data_loaded)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.thread_pool.start(worker)

    def _load_file_data(self, metadata: FileMetadata, filter_config: FilterConfig) -> dict:
        cache_key = (str(metadata.file_path),) + filter_config.cache_key()
        processed = self.processed_cache.get(cache_key)
        if processed is None:
            raw = read_bin_memmap(metadata)
            phase_data = convert_raw_to_radians(raw)
            processed = apply_filter(phase_data, metadata.sample_rate_hz, filter_config)
            self.processed_cache.put(cache_key, processed, int(processed.nbytes))

        return {
            "metadata": metadata,
            "data": processed,
            "file_path": metadata.file_path,
            "filter_config": filter_config,
        }

    def _on_data_loaded(self, payload: dict) -> None:
        self.current_metadata = payload["metadata"]
        self.current_data = payload["data"]
        self.current_file_path = payload["file_path"]
        self._apply_auto_point_limit(metadata=self.current_metadata)
        try:
            self._render_current_data()
        except Exception as exc:
            self._show_error(str(exc))

    def apply_current_settings(self) -> None:
        """Apply filter and display parameters to the currently selected file."""
        file_path = self._selected_file_path() or self.current_file_path
        if file_path is None:
            return
        try:
            self._load_file(file_path, self._build_filter_config())
        except Exception as exc:
            self._show_error(str(exc))

    def _render_current_data(self) -> None:
        if self.current_metadata is None or self.current_data is None:
            return

        start_point, end_point = self._read_point_range()
        self._sync_waveform_point_bounds()
        waveform_point = self._read_waveform_point(start_point, end_point)
        vmin, vmax = self._read_color_range()
        colormap_name = self.control_panel.colormap_combo.currentText().strip().lower() or "seismic"
        image_data = prepare_image_matrix(self.current_data, start_point, end_point)
        self.space_time_view.set_data(
            metadata=self.current_metadata,
            image_data=image_data,
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
        self._sync_waveform_point_bounds()

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
            return
        self.control_panel.point_end_edit.setText(self._default_point_end_value(file_path=file_path, metadata=metadata))
        self._sync_waveform_point_bounds()

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
        return (
            f"Loading {metadata.file_path.name} | "
            f"采样率：{self._format_sample_rate(metadata.sample_rate_hz)}Hz | "
            f"位置点数：{metadata.points} points | "
            f"单个数据文件时长：{metadata.duration_s:.2f} s"
        )

    def _update_status_text(self) -> None:
        if self.current_metadata is None or self.current_file_path is None:
            return
        self.statusBar().showMessage(
            (
                f"当前文件：{self.current_file_path.name} | "
                f"采样率：{self._format_sample_rate(self.current_metadata.sample_rate_hz)}Hz | "
                f"位置点数：{self.current_metadata.points} points | "
                f"单个数据文件时长：{self.current_metadata.duration_s:.2f} s"
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
            Path(__file__).resolve().parents[1] / "eDASread_logo.png",
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
        filter_config = self._build_filter_config() if save_filtered else FilterConfig()

        self.statusBar().showMessage("Exporting current view ...")
        worker = FunctionWorker(
            self._export_current_view_data,
            export_folder,
            filter_config,
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
        filter_config: FilterConfig,
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

        if filter_config.is_enabled():
            cache_key = (str(metadata.file_path),) + filter_config.cache_key()
            export_data = self.processed_cache.get(cache_key)
            if export_data is None:
                raw = read_bin_memmap(metadata)
                phase_data = convert_raw_to_radians(raw)
                export_data = apply_filter(phase_data, metadata.sample_rate_hz, filter_config)
                self.processed_cache.put(cache_key, export_data, int(export_data.nbytes))

            export_phase_slice = np.ascontiguousarray(export_data[frame_slice, point_slice], dtype=np.float32)
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
            "filtered": filter_config.is_enabled(),
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


