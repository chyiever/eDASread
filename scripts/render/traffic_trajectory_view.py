"""Interactive energy-map view for manual traffic trajectory annotation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QTransform
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QShortcut, QSplitter, QVBoxLayout, QWidget,
)

from scripts.render.plot_style import style_axis
from scripts.render.space_time_view import COLOR_MAPS
from scripts.render.time_axis import TimeAxisItem
from scripts.processing.view_data import MAX_IMAGE_DISPLAY_COLUMNS, downsample_image_for_display


def _uniform_param_cell(label_text: str, widget: QWidget, *, width: int | None = None) -> QWidget:
    """Build a compact label-above-input cell with a uniform input width."""
    cell = QWidget()
    layout = QVBoxLayout(cell)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    label = QLabel(label_text)
    label.setFont(widget.font())
    label.setAlignment(Qt.AlignCenter)

    if width is None:
        width = 110 if isinstance(widget, QComboBox) else 96
    widget.setFixedWidth(width)
    widget.setMinimumWidth(width)

    layout.addWidget(label)
    layout.addWidget(widget, 0, Qt.AlignHCenter)
    return cell


def _labeled_pair(label_text: str, widget: QWidget) -> QWidget:
    """Label-left, input-right compact pair for a single-row parameter strip."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    label = QLabel(label_text)
    label.setFont(widget.font())
    layout.addWidget(label)
    layout.addWidget(widget)
    return container


class TrafficTrajectoryView(QWidget):
    """Tab5 controls, energy image, and multi-polyline annotation state."""

    generate_requested = pyqtSignal()
    COLORS = ["#00e5ff", "#ffeb3b", "#ff4081", "#69f0ae", "#ff9100", "#b388ff"]

    def __init__(self) -> None:
        super().__init__()
        self.energy: np.ndarray | None = None
        self.image_data: np.ndarray | None = None
        self.display_mode = "energy"
        self.energy_rate_hz = 1.0
        self.point_start = 1
        self.source_files: list[str] = []
        self.trajectories: list[list[tuple[float, float]]] = []
        self.trajectory_visible: list[bool] = []
        self.current_points: list[tuple[float, float]] = []
        self._items: list[pg.PlotDataItem] = []
        self._build_ui()

    def _build_ui(self) -> None:
        controls = QGroupBox("交通 DAS 能量图与轨迹标记参数")
        row = QHBoxLayout(controls)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)
        self.channel_start_edit = QLineEdit("200")
        self.channel_end_edit = QLineEdit("820")
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem("能量图", "energy")
        self.display_mode_combo.addItem("相位图", "phase")
        self.low_cut_edit = QLineEdit("0.05")
        self.high_cut_edit = QLineEdit("10")
        self.decimation_edit = QLineEdit("10")
        self.common_mode_checkbox = QCheckBox("去共模")
        self.common_mode_checkbox.setChecked(True)
        self.balance_combo = QComboBox()
        self.balance_combo.addItems(["p999", "mad", "none"])
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "inferno", "viridis", "plasma", "magma", "jet", "turbo", "seismic", "hsv"
        ])
        self.vmin_edit = QLineEdit()
        self.vmin_edit.setPlaceholderText("auto")
        self.vmax_edit = QLineEdit()
        self.vmax_edit.setPlaceholderText("auto")
        self.auto_color_checkbox = QCheckBox("颜色自动")
        self.auto_color_checkbox.setChecked(True)
        self.apply_color_button = QPushButton("应用颜色条")
        self.generate_button = QPushButton("生成能量图")
        for widget in [
            self.channel_start_edit, self.channel_end_edit, self.low_cut_edit,
            self.high_cut_edit, self.decimation_edit, self.vmin_edit, self.vmax_edit,
            self.display_mode_combo, self.balance_combo, self.colormap_combo,
            self.common_mode_checkbox, self.auto_color_checkbox,
            self.apply_color_button, self.generate_button,
        ]:
            widget.setFont(QFont("Times New Roman", 10))
        for edit, width in [
            (self.channel_start_edit, 56),
            (self.channel_end_edit, 56),
            (self.low_cut_edit, 52),
            (self.high_cut_edit, 52),
            (self.decimation_edit, 48),
            (self.vmin_edit, 48),
            (self.vmax_edit, 48),
        ]:
            edit.setFixedWidth(width)
        self.display_mode_combo.setFixedWidth(84)
        self.balance_combo.setFixedWidth(76)
        self.colormap_combo.setFixedWidth(84)
        self.apply_color_button.setFixedWidth(92)
        self.generate_button.setFixedWidth(92)

        row.addWidget(_labeled_pair("图像类型", self.display_mode_combo))
        row.addWidget(_labeled_pair("通道起点", self.channel_start_edit))
        row.addWidget(_labeled_pair("通道终点", self.channel_end_edit))
        row.addWidget(_labeled_pair("低频(Hz)", self.low_cut_edit))
        row.addWidget(_labeled_pair("高频(Hz)", self.high_cut_edit))
        row.addWidget(_labeled_pair("降采样", self.decimation_edit))
        row.addWidget(_labeled_pair("通道均衡", self.balance_combo))
        row.addWidget(_labeled_pair("Colormap", self.colormap_combo))
        row.addWidget(_labeled_pair("VMin", self.vmin_edit))
        row.addWidget(_labeled_pair("VMax", self.vmax_edit))
        row.addWidget(self.auto_color_checkbox)
        row.addWidget(self.common_mode_checkbox)
        row.addWidget(self.apply_color_button)
        row.addWidget(self.generate_button)
        row.addStretch(1)

        self.time_axis = TimeAxisItem(orientation="bottom")
        self.channel_axis = pg.AxisItem(orientation="left")
        # Keep the same standard PlotWidget/ViewBox construction used by Tab1.
        # Replacing its ViewBox caused axis-layout differences on some Qt/PG
        # versions, including labels/ticks being allocated no visible space.
        self.plot = pg.PlotWidget(
            axisItems={"bottom": self.time_axis, "left": self.channel_axis}
        )
        self.plot.setBackground("w")
        self.plot.plotItem.showAxis("bottom", show=True)
        self.plot.plotItem.showAxis("left", show=True)
        self.time_axis.setLabel(text="Time", units="s")
        self.channel_axis.setLabel(text="Channel Index")
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        self.plot.setTitle("选择连续文件后生成能量图；左键依次添加轨迹点，Ctrl+拖动框选放大")
        font = QFont("Times New Roman", 18)
        for name in ("left", "bottom"):
            axis_item = self.plot.getAxis(name)
            style_axis(axis_item, font, font)
            axis_item.setStyle(tickLength=8, showValues=True)
            axis_item.enableAutoSIPrefix(False)
            axis_item.setVisible(True)
            axis_item.label.setVisible(True)
            axis_item.setZValue(20)
        self.plot.getAxis("left").setWidth(110)
        self.plot.getAxis("bottom").setHeight(68)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        self.histogram_widget = pg.HistogramLUTWidget(orientation="vertical")
        self.histogram_widget.setImageItem(self.image_item)
        self.histogram_widget.setFixedWidth(150)
        self.histogram_widget.setBackground("w")
        self.histogram_widget.item.axis.setLabel(text="Energy", units="")
        style_axis(self.histogram_widget.item.axis, font, font)

        plot_area = QWidget()
        plot_area_layout = QHBoxLayout(plot_area)
        plot_area_layout.setContentsMargins(0, 0, 0, 0)
        plot_area_layout.setSpacing(6)
        plot_area_layout.addWidget(self.plot, 1)
        plot_area_layout.addWidget(self.histogram_widget)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("轨迹列表"))
        self.trajectory_list = QListWidget()
        side_layout.addWidget(self.trajectory_list, 1)
        self.finish_button = QPushButton("完成当前轨迹")
        self.undo_button = QPushButton("撤销当前点")
        self.delete_button = QPushButton("删除所选轨迹")
        self.clear_button = QPushButton("清空全部轨迹")
        self.save_button = QPushButton("导出标注 JSON/CSV")
        self.load_button = QPushButton("载入标注 JSON")
        for button in (self.finish_button, self.undo_button, self.delete_button,
                       self.clear_button, self.save_button, self.load_button):
            side_layout.addWidget(button)
        help_label = QLabel("左键：添加顶点\n起点 → 可选中间点 → 结束点\nCtrl+左键拖动：框选放大\n右键拖动：平移；滚轮：缩放")
        help_label.setWordWrap(True)
        side_layout.addWidget(help_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(plot_area)
        splitter.addWidget(side)
        splitter.setSizes([1100, 250])
        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(splitter, 1)

        self.generate_button.clicked.connect(self.generate_requested)
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self.finish_button.clicked.connect(self.finish_current)
        self.undo_button.clicked.connect(self.undo_point)
        self.delete_button.clicked.connect(self.delete_selected)
        self.clear_button.clicked.connect(self.clear_trajectories)
        self.save_button.clicked.connect(self.choose_and_save)
        self.load_button.clicked.connect(self.choose_and_load)
        self.colormap_combo.currentTextChanged.connect(self._apply_colormap)
        self.apply_color_button.clicked.connect(self.apply_color_controls)
        self.trajectory_list.itemChanged.connect(self._on_trajectory_check_changed)
        self.undo_shortcut = QShortcut(QKeySequence.Undo, self)
        self.undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self.undo_point)

    def _on_display_mode_changed(self) -> None:
        is_energy = self.display_mode_combo.currentData() == "energy"
        self.generate_button.setText("生成能量图" if is_energy else "绘制相位图")
        for widget in (
            self.low_cut_edit, self.high_cut_edit, self.decimation_edit,
            self.common_mode_checkbox, self.balance_combo,
        ):
            widget.setEnabled(is_energy)

    def set_energy(self, energy: np.ndarray, energy_rate_hz: float, point_start: int,
                   source_files: list[str], start_time=None) -> None:
        self.energy = np.asarray(energy, dtype=np.float32)
        self.image_data = self.energy
        self.display_mode = "energy"
        self.energy_rate_hz = float(energy_rate_hz)
        self.point_start = int(point_start)
        self.source_files = list(source_files)
        self.time_axis.set_reference_time(start_time)
        self.clear_trajectories()
        self._set_image_matrix(self.energy, self.energy_rate_hz)
        self.apply_color_controls()
        duration_s = self.energy.shape[0] / self.energy_rate_hz
        self._set_plot_ranges(duration_s, point_start, self.energy.shape[1])
        self.histogram_widget.item.axis.setLabel(text="Energy", units="")
        self.plot.setTitle(
            f"车辆轨迹能量图 | 时间 {duration_s:.3f} s | "
            f"通道 {point_start}-{point_start + self.energy.shape[1] - 1} | "
            f"{len(source_files)} 个文件"
        )

    def set_phase(self, phase: np.ndarray, sample_rate_hz: float, point_start: int,
                  source_files: list[str], start_time=None) -> None:
        """Display the Tab1-stitched phase matrix with time on the x axis."""
        self.energy = None
        self.image_data = np.asarray(phase, dtype=np.float32)
        self.display_mode = "phase"
        self.energy_rate_hz = float(sample_rate_hz)
        self.point_start = int(point_start)
        self.source_files = list(source_files)
        self.time_axis.set_reference_time(start_time)
        self.clear_trajectories()
        self._set_image_matrix(self.image_data, self.energy_rate_hz)
        self.apply_color_controls()
        duration_s = self.image_data.shape[0] / self.energy_rate_hz
        point_end = point_start + self.image_data.shape[1] - 1
        self._set_plot_ranges(duration_s, point_start, self.image_data.shape[1])
        self.histogram_widget.item.axis.setLabel(text="Phase", units="rad")
        self.plot.setTitle(
            f"相位瀑布图 | 时间 {duration_s:.3f} s | 通道 {point_start}-{point_end} | "
            f"{len(source_files)} 个文件"
        )

    def _set_image_matrix(self, data: np.ndarray, sample_rate_hz: float) -> None:
        """Use Tab1's channel×time display orientation and time LOD."""
        image_channel_time = np.asarray(data, dtype=np.float32).T
        display, samples_per_column = downsample_image_for_display(
            image_channel_time, MAX_IMAGE_DISPLAY_COLUMNS
        )
        self.image_item.setImage(display, autoLevels=False)
        transform = QTransform()
        transform.translate(0.0, float(self.point_start))
        transform.scale(samples_per_column / sample_rate_hz, 1.0)
        self.image_item.setTransform(transform)

    def _set_plot_ranges(self, duration_s: float, point_start: int, point_count: int) -> None:
        self.plot.setLimits(
            xMin=0.0, xMax=duration_s,
            yMin=float(point_start), yMax=float(point_start + point_count),
        )
        self.plot.setXRange(0.0, duration_s, padding=0)
        self.plot.setYRange(float(point_start), float(point_start + point_count), padding=0)

    def _apply_colormap(self) -> None:
        name = self.colormap_combo.currentText().lower()
        cmap = self._resolve_colormap(name)
        self.image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.histogram_widget.gradient.setColorMap(cmap)

    @staticmethod
    def _resolve_colormap(name: str) -> pg.ColorMap:
        """Resolve project and PyQtGraph maps across old/new PG versions."""
        normalized = name.strip().lower() or "jet"
        # jet/seismic/hsv are already fully constructed ColorMap objects.
        # They are not color arrays and must never be passed to len()/ColorMap.
        if normalized in COLOR_MAPS:
            return COLOR_MAPS[normalized]
        try:
            return pg.colormap.get(normalized)
        except Exception:
            # Some releases expose these maps only through matplotlib.
            try:
                get_from_matplotlib = getattr(pg.colormap, "getFromMatplotlib", None)
                if get_from_matplotlib is not None:
                    return get_from_matplotlib(normalized)
                return pg.colormap.get(normalized, source="matplotlib")
            except Exception:
                return COLOR_MAPS["jet"]

    def _on_plot_clicked(self, event) -> None:
        """Add an annotation vertex while retaining Tab1's native ViewBox."""
        if event.button() != Qt.LeftButton or event.modifiers() & Qt.ControlModifier:
            return
        if not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.getViewBox().mapSceneToView(event.scenePos())
        self._add_point(float(point.x()), float(point.y()))

    def apply_color_controls(self) -> None:
        """Apply the Tab5 color range without rebuilding the energy image."""
        self._apply_colormap()
        if self.image_data is None or not self.image_data.size:
            return
        if self.auto_color_checkbox.isChecked():
            if self.display_mode == "phase":
                bound = max(float(np.percentile(np.abs(self.image_data), 99.0)), 1e-12)
                level_min, level_max = -bound, bound
            else:
                level_min = float(np.percentile(self.image_data, 0.0))
                level_max = float(np.percentile(self.image_data, 99.5))
        else:
            try:
                level_min = float(self.vmin_edit.text().strip())
                level_max = float(self.vmax_edit.text().strip())
            except ValueError:
                QMessageBox.warning(self, "颜色范围", "关闭颜色自动后，请填写有效的 VMin 和 VMax。")
                return
        if not np.isfinite(level_min) or not np.isfinite(level_max) or level_min >= level_max:
            QMessageBox.warning(self, "颜色范围", "颜色范围需要满足 VMin < VMax。")
            return
        self.image_item.setLevels((level_min, level_max))
        self.histogram_widget.item.setLevels(level_min, level_max)

    def _add_point(self, t: float, channel: float) -> None:
        if self.image_data is None:
            return
        duration = self.image_data.shape[0] / self.energy_rate_hz
        last_channel = self.point_start + self.image_data.shape[1] - 1
        if not (0 <= t <= duration and self.point_start <= channel <= last_channel + 1):
            return
        self.current_points.append((float(t), float(channel)))
        self._redraw()

    def finish_current(self) -> None:
        if len(self.current_points) < 2:
            QMessageBox.information(self, "轨迹标记", "一条轨迹至少需要起点和结束点。")
            return
        self.trajectories.append(list(self.current_points))
        self.trajectory_visible.append(True)
        self.current_points.clear()
        self._redraw()

    def undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self._redraw()

    def delete_selected(self) -> None:
        row = self.trajectory_list.currentRow()
        if 0 <= row < len(self.trajectories):
            self.trajectories.pop(row)
            self.trajectory_visible.pop(row)
            self._redraw()

    def clear_trajectories(self) -> None:
        self.trajectories.clear()
        self.trajectory_visible.clear()
        self.current_points.clear()
        self._redraw()

    def _redraw(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items.clear()
        self.trajectory_list.blockSignals(True)
        self.trajectory_list.clear()
        for idx, points in enumerate(self.trajectories):
            color = self.COLORS[idx % len(self.COLORS)]
            visible = self.trajectory_visible[idx] if idx < len(self.trajectory_visible) else True
            if visible:
                x, y = zip(*points)
                plot_item = self.plot.plot(x, y, pen=pg.mkPen(color, width=3), symbol="o",
                                           symbolSize=7, symbolBrush=color)
                self._items.append(plot_item)
            list_item = QListWidgetItem(f"轨迹 {idx + 1}（{len(points)} 点）")
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
            self.trajectory_list.addItem(list_item)
        if self.current_points:
            x, y = zip(*self.current_points)
            item = self.plot.plot(x, y, pen=pg.mkPen("#ffffff", width=2, style=Qt.DashLine),
                                  symbol="o", symbolSize=8, symbolBrush="#ffffff")
            self._items.append(item)
        self.trajectory_list.blockSignals(False)

    def _on_trajectory_check_changed(self, item: QListWidgetItem) -> None:
        row = self.trajectory_list.row(item)
        if 0 <= row < len(self.trajectories):
            self.trajectory_visible[row] = item.checkState() == Qt.Checked
            self._redraw()

    def annotation_document(self) -> dict:
        return {
            "format": "edasread-traffic-trajectories-v1",
            "coordinate_system": {"x": "seconds_from_first_frame", "y": "1_based_channel"},
            "source_files": self.source_files,
            "display_mode": self.display_mode,
            "display_sample_rate_hz": self.energy_rate_hz,
            "energy_rate_hz": self.energy_rate_hz if self.display_mode == "energy" else None,
            "channel_range": [self.point_start, self.point_start + (self.image_data.shape[1] - 1 if self.image_data is not None else 0)],
            "trajectories": [
                {"id": i + 1, "visible": self.trajectory_visible[i],
                 "points": [{"time_s": t, "channel": ch} for t, ch in points]}
                for i, points in enumerate(self.trajectories)
            ],
        }

    def choose_and_save(self) -> None:
        if not self.trajectories:
            QMessageBox.information(self, "导出标注", "当前没有已完成的轨迹。")
            return
        path, selected = QFileDialog.getSaveFileName(self, "导出轨迹标注", "traffic_trajectories.json",
                                                     "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() == ".csv" or "CSV" in selected:
            if target.suffix.lower() != ".csv": target = target.with_suffix(".csv")
            with target.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["trajectory_id", "vertex_index", "time_s", "channel"])
                for tid, points in enumerate(self.trajectories, 1):
                    for vertex, (t, ch) in enumerate(points): writer.writerow([tid, vertex, f"{t:.9f}", f"{ch:.6f}"])
        else:
            if target.suffix.lower() != ".json": target = target.with_suffix(".json")
            target.write_text(json.dumps(self.annotation_document(), ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "导出完成", f"轨迹标注已保存到：\n{target}")

    def choose_and_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入轨迹标注", "", "JSON (*.json)")
        if not path: return
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            loaded = [[(float(p["time_s"]), float(p["channel"])) for p in tr["points"]]
                      for tr in doc.get("trajectories", [])]
            if any(len(points) < 2 for points in loaded): raise ValueError("存在少于两个顶点的轨迹。")
            self.trajectories = loaded
            self.trajectory_visible = [bool(tr.get("visible", True)) for tr in doc.get("trajectories", [])]
            self.current_points.clear()
            self._redraw()
        except Exception as exc:
            QMessageBox.critical(self, "载入失败", str(exc))
