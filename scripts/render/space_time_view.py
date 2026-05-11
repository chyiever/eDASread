"""PyQtGraph-based combined waveform and space-time plotting widget."""

from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont, QTransform
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from scripts.core.models import FileMetadata
from scripts.render.time_axis import TimeAxisItem


def _build_color_map(color_stops: list[tuple[float, tuple[int, int, int]]]) -> pg.ColorMap:
    positions = np.array([stop[0] for stop in color_stops], dtype=float)
    colors = np.array([[r, g, b, 255] for _, (r, g, b) in color_stops], dtype=np.ubyte)
    return pg.ColorMap(positions, colors)


def _build_hsv_colormap() -> pg.ColorMap:
    positions = np.linspace(0.0, 1.0, 13)
    colors = []
    for position in positions:
        red, green, blue = colorsys.hsv_to_rgb(position, 1.0, 1.0)
        colors.append((int(red * 255), int(green * 255), int(blue * 255)))
    return _build_color_map(list(zip(positions, colors)))


COLOR_MAPS = {
    "jet": _build_color_map(
        [
            (0.00, (0, 0, 131)),
            (0.13, (0, 60, 170)),
            (0.38, (5, 255, 255)),
            (0.50, (255, 255, 0)),
            (0.63, (250, 0, 0)),
            (0.88, (128, 0, 0)),
            (1.00, (80, 0, 0)),
        ]
    ),
    "seismic": _build_color_map(
        [
            (0.00, (0, 0, 76)),
            (0.20, (0, 56, 168)),
            (0.40, (127, 197, 255)),
            (0.50, (255, 255, 255)),
            (0.60, (255, 179, 179)),
            (0.80, (214, 39, 40)),
            (1.00, (103, 0, 31)),
        ]
    ),
    "hsv": _build_hsv_colormap(),
}


class SpaceTimeView(QWidget):
    """Widget responsible for displaying linked waveform and space-time plots."""

    view_range_changed = pyqtSignal(tuple, tuple)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._metadata: Optional[FileMetadata] = None
        self._image_data: Optional[np.ndarray] = None
        self._y_start_point = 1
        self._y_end_point = 1
        self._waveform_point = 1
        self._colormap_name = "jet"
        self._view_history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._default_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._last_view_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._tracking_enabled = True

        axis_font = QFont("Times New Roman", 11)
        label_font = QFont("Times New Roman", 12)
        label_font.setBold(True)

        self.time_axis = TimeAxisItem(orientation="bottom")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": self.time_axis})
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel("left", "Point Index")
        self.plot_widget.setLabel("bottom", "Time")
        self.plot_widget.getAxis("left").setStyle(tickFont=axis_font)
        self.plot_widget.getAxis("bottom").setStyle(tickFont=axis_font)
        self.plot_widget.getAxis("left").label.setFont(label_font)
        self.plot_widget.getAxis("bottom").label.setFont(label_font)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_widget.addItem(self.image_item)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._emit_view_range_changed)
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(self._record_manual_view)

        self.waveform_axis = TimeAxisItem(orientation="bottom")
        self.waveform_plot = pg.PlotWidget(axisItems={"bottom": self.waveform_axis})
        self.waveform_plot.setBackground("w")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.15)
        self.waveform_plot.setLabel("left", "Phase", units="rad")
        self.waveform_plot.setLabel("bottom", "Time")
        self.waveform_plot.setTitle("指定位置点时域波形")
        self.waveform_plot.getAxis("left").setStyle(tickFont=axis_font)
        self.waveform_plot.getAxis("bottom").setStyle(tickFont=axis_font)
        self.waveform_plot.getAxis("left").label.setFont(label_font)
        self.waveform_plot.getAxis("bottom").label.setFont(label_font)
        self.waveform_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.waveform_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen(color="#0f5ea8", width=1.8))

        self.waveform_plot.setXLink(self.plot_widget)

        waveform_row_widget = QWidget()
        waveform_row_layout = QHBoxLayout(waveform_row_widget)
        waveform_row_layout.setContentsMargins(0, 0, 0, 0)
        waveform_row_layout.setSpacing(6)
        waveform_row_layout.addWidget(self.waveform_plot, 1)
        waveform_right_spacer = QWidget()
        waveform_right_spacer.setFixedWidth(150)
        waveform_row_layout.addWidget(waveform_right_spacer)

        self.histogram_widget = pg.HistogramLUTWidget(orientation="vertical")
        self.histogram_widget.setImageItem(self.image_item)
        self.histogram_widget.setFixedWidth(150)
        self.histogram_widget.setBackground("w")
        self.histogram_widget.item.axis.setLabel(text="Amplitude", units="rad")
        self.histogram_widget.item.axis.setStyle(tickFont=axis_font)
        self.histogram_widget.item.axis.label.setFont(label_font)
        self._apply_colormap(self._colormap_name)

        lower_row_widget = QWidget()
        lower_row_layout = QHBoxLayout(lower_row_widget)
        lower_row_layout.setContentsMargins(0, 0, 0, 0)
        lower_row_layout.setSpacing(6)
        lower_row_layout.addWidget(self.plot_widget, 1)
        lower_row_layout.addWidget(self.histogram_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(waveform_row_widget, 2)
        layout.addWidget(lower_row_widget, 6)

    def set_data(
        self,
        metadata: FileMetadata,
        image_data: np.ndarray,
        start_point: int,
        end_point: int,
        waveform_point: int,
        vmin: float | None = None,
        vmax: float | None = None,
        colormap_name: str = "jet",
    ) -> None:
        """Update the displayed image and linked waveform view."""
        self._metadata = metadata
        self._image_data = image_data
        self._y_start_point = start_point
        self._y_end_point = end_point
        self._waveform_point = waveform_point
        self._colormap_name = colormap_name
        self._view_history.clear()
        self._default_range = None
        self._last_view_range = None

        levels = self._resolve_levels(image_data, vmin, vmax)
        self._apply_colormap(colormap_name)
        self.time_axis.set_reference_time(metadata.start_time)
        self.waveform_axis.set_reference_time(metadata.start_time)
        self.image_item.setImage(image_data, autoLevels=False, levels=levels)
        self.histogram_widget.item.setLevels(*levels)

        transform = QTransform()
        transform.scale(1.0 / metadata.sample_rate_hz, 1.0)
        transform.translate(0.0, start_point)
        self.image_item.setTransform(transform)

        self.plot_widget.setLimits(
            xMin=0.0,
            xMax=metadata.duration_s,
            yMin=start_point,
            yMax=end_point + 1,
        )
        self.waveform_plot.setLimits(xMin=0.0, xMax=metadata.duration_s)
        self._update_waveform_curve()
        self.reset_view(clear_history=True)

    def update_waveform_point(self, waveform_point: int) -> None:
        """Update the waveform subplot to show a different point."""
        self._waveform_point = waveform_point
        self._update_waveform_curve()

    def update_colormap(self, colormap_name: str) -> None:
        """Update the colormap immediately without reloading file data."""
        self._colormap_name = colormap_name
        self._apply_colormap(colormap_name)

    def reset_view(self, clear_history: bool = False) -> None:
        """Reset the view to show the full visible image extent."""
        if self._metadata is None or self._image_data is None:
            return

        if clear_history:
            self._view_history.clear()

        self._tracking_enabled = False
        try:
            self.plot_widget.setXRange(0.0, self._metadata.duration_s, padding=0.0)
            self.plot_widget.setYRange(self._y_start_point, self._y_end_point + 1, padding=0.0)
            current = self.current_ranges()
            self._default_range = current
            self._last_view_range = current
        finally:
            self._tracking_enabled = True

    def back_view(self) -> None:
        """Restore the previous view range if one exists."""
        if not self._view_history:
            return

        x_range, y_range = self._view_history.pop()
        self._tracking_enabled = False
        try:
            self.plot_widget.setXRange(x_range[0], x_range[1], padding=0.0)
            self.plot_widget.setYRange(y_range[0], y_range[1], padding=0.0)
            self._last_view_range = (x_range, y_range)
        finally:
            self._tracking_enabled = True

    def zoom_out_horizontal(self, factor: float = 2.0) -> None:
        """Expand the current horizontal view around its center."""
        if self._metadata is None:
            return

        current_range = self.current_ranges()
        x_range, y_range = current_range
        current_span = max(x_range[1] - x_range[0], 1e-9)
        center = (x_range[0] + x_range[1]) / 2.0
        new_span = min(self._metadata.duration_s, current_span * factor)
        new_start = max(0.0, center - new_span / 2.0)
        new_end = min(self._metadata.duration_s, new_start + new_span)
        new_start = max(0.0, new_end - new_span)

        self._append_history(current_range)
        self._tracking_enabled = False
        try:
            self.plot_widget.setXRange(new_start, new_end, padding=0.0)
            self._last_view_range = ((new_start, new_end), y_range)
        finally:
            self._tracking_enabled = True

    def current_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the current visible ranges of the space-time plot."""
        view_box = self.plot_widget.getViewBox()
        x_range, y_range = view_box.viewRange()
        return tuple(x_range), tuple(y_range)

    def set_x_range(self, x_min: float, x_max: float) -> None:
        """Update the horizontal view range."""
        self.plot_widget.setXRange(x_min, x_max, padding=0.0)

    def set_y_range(self, y_min: float, y_max: float) -> None:
        """Update the vertical view range."""
        self.plot_widget.setYRange(y_min, y_max, padding=0.0)

    def _update_waveform_curve(self) -> None:
        if self._metadata is None or self._image_data is None:
            self.waveform_curve.clear()
            return

        waveform_index = self._waveform_point - self._y_start_point
        if waveform_index < 0 or waveform_index >= self._image_data.shape[0]:
            self.waveform_curve.clear()
            return

        waveform = np.asarray(self._image_data[waveform_index], dtype=np.float32)
        x_values = np.arange(waveform.shape[0], dtype=np.float32) / self._metadata.sample_rate_hz
        self.waveform_curve.setData(x=x_values, y=waveform)
        self.waveform_plot.setTitle(f"指定位置点 {self._waveform_point} 时域波形")
        self._reset_waveform_y_range(waveform)

    def _reset_waveform_y_range(self, waveform: np.ndarray) -> None:
        if waveform.size == 0:
            return
        y_min = float(np.nanmin(waveform))
        y_max = float(np.nanmax(waveform))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        if y_min == y_max:
            delta = abs(y_min) * 0.05 or 1e-6
            y_min -= delta
            y_max += delta
        else:
            padding = (y_max - y_min) * 0.08
            y_min -= padding
            y_max += padding
        self.waveform_plot.setYRange(y_min, y_max, padding=0.0)

    def _resolve_levels(
        self,
        image_data: np.ndarray,
        vmin: float | None,
        vmax: float | None,
    ) -> tuple[float, float]:
        if vmin is None or vmax is None:
            auto_min = float(np.nanmin(image_data))
            auto_max = float(np.nanmax(image_data))
            if auto_min == auto_max:
                auto_max = auto_min + 1e-6
            return auto_min, auto_max
        return vmin, vmax

    def _apply_colormap(self, colormap_name: str) -> None:
        color_map = COLOR_MAPS.get(colormap_name, COLOR_MAPS["jet"])
        self.image_item.setColorMap(color_map)
        self.histogram_widget.gradient.setColorMap(color_map)

    def _record_manual_view(self, *args) -> None:
        if not self._tracking_enabled:
            return

        current_range = self.current_ranges()
        previous_range = self._last_view_range or self._default_range
        if previous_range is not None and not self._ranges_close(previous_range, current_range):
            self._append_history(previous_range)
        self._last_view_range = current_range

    def _append_history(self, range_value: tuple[tuple[float, float], tuple[float, float]]) -> None:
        if self._view_history and self._ranges_close(self._view_history[-1], range_value):
            return
        self._view_history.append(range_value)
        if len(self._view_history) > 20:
            self._view_history.pop(0)

    def _ranges_close(
        self,
        a: tuple[tuple[float, float], tuple[float, float]],
        b: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        return (
            abs(a[0][0] - b[0][0]) < 1e-6
            and abs(a[0][1] - b[0][1]) < 1e-6
            and abs(a[1][0] - b[1][0]) < 1e-6
            and abs(a[1][1] - b[1][1]) < 1e-6
        )

    def _emit_view_range_changed(self, _view_box, _ranges) -> None:
        self.view_range_changed.emit(*self.current_ranges())
