"""PyQtGraph-based combined waveform and space-time plotting widget."""

from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QTransform
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from scripts.core.models import FileMetadata
from scripts.processing.view_data import (
    MAX_IMAGE_DISPLAY_COLUMNS,
    downsample_image_for_display,
    robust_min_max,
)
from scripts.render.plot_style import AXIS_TEXT_COLOR, style_axis
from scripts.render.time_axis import TimeAxisItem


def _decimate_peaks(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace to ``<= max_points`` samples keeping the min/max envelope.

    Returns ``(x_indices, y_values)`` where both have the same length; the
    ``x`` values stay non-decreasing (dx = segment length).  The original
    final sample is always appended so the trace still reaches the very end of
    the time axis.  Used so long traces are only uploaded to pyqtgraph at
    display resolution, keeping wiggle rebuilds fast even for million-frame
    files.
    """
    n = values.shape[0]
    if n <= max_points:
        return np.arange(n, dtype=np.float64), values
    segments = max_points
    seg = max(1, n // segments)
    block = values[: segments * seg].reshape(segments, seg)
    out_y = np.empty(segments * 2 + 1, dtype=values.dtype)
    out_y[0 : segments * 2 : 2] = block.max(axis=1)
    out_y[1 : segments * 2 : 2] = block.min(axis=1)
    out_y[-1] = values[-1]
    out_x = np.empty(segments * 2 + 1, dtype=np.float64)
    out_x[0 : segments * 2 : 2] = np.arange(segments, dtype=np.float64) * seg
    out_x[1 : segments * 2 : 2] = np.arange(segments, dtype=np.float64) * seg
    out_x[-1] = n - 1
    return out_x, out_y


class DistanceAxisItem(pg.AxisItem):
    """Left axis that shows tick labels as physical distance or channel number.

    The underlying view coordinates stay in 1-based channel space (so zoom,
    pan, scrollbars and export logic are unchanged); only the tick placement
    and label formatting are switched.  In distance mode the tick positions
    are chosen so they fall on round distances computed from the trace
    spacing and the position of the first trace.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._spacing_m = 5.0
        self._first_pos_m = 0.0
        self._distance_mode = False
        self._unit = "m"
        self.enableAutoSIPrefix(False)

    def configure(self, spacing_m: float, first_pos_m: float, distance_mode: bool, unit: str) -> None:
        self._spacing_m = float(spacing_m) if float(spacing_m) > 0 else 5.0
        self._first_pos_m = float(first_pos_m)
        self._distance_mode = bool(distance_mode)
        self._unit = unit if unit in ("m", "km") else "m"
        self.picture = None
        self.update()

    def _to_distance(self, channel_value: float) -> float:
        """Distance in metres for a 1-based channel coordinate."""
        return self._first_pos_m + (channel_value - 1.0) * self._spacing_m

    def _display_value(self, distance_m: float) -> float:
        return distance_m / 1000.0 if self._unit == "km" else distance_m

    def tickSpacing(self, minVal, maxVal, size):  # noqa: N802
        if not self._distance_mode or self._spacing_m <= 0:
            return super().tickSpacing(minVal, maxVal, size)
        unit_factor = 1000.0 if self._unit == "km" else 1.0
        dmin = self._to_distance(minVal) / unit_factor
        dmax = self._to_distance(maxVal) / unit_factor
        levels = super().tickSpacing(dmin, dmax, size)
        converted = []
        for spacing_display, _offset in levels:
            spacing_dist_m = spacing_display * unit_factor
            channel_spacing = spacing_dist_m / self._spacing_m
            anchor_distance = (-self._first_pos_m) % spacing_dist_m
            channel_offset = 1.0 + anchor_distance / self._spacing_m
            converted.append((channel_spacing, channel_offset))
        return converted

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        if not self._distance_mode:
            return super().tickStrings(values, scale, spacing)
        labels = []
        for value in values:
            if not np.isfinite(value):
                labels.append("")
                continue
            display = self._display_value(self._to_distance(value))
            if abs(display - round(display)) < 1e-3:
                labels.append(f"{int(round(display))}")
            else:
                labels.append(f"{display:.3g}")
        return labels


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
    cursor_info = pyqtSignal(str)

    WIGGLE_SCALE_MODES = [
        "全局最大值",
        "三道最大值",
        "单道最大值",
        "全局均方根",
        "三道均方根",
        "单道均方根",
        "全局平均值",
    ]
    MAX_WIGGLE_TRACES = 150
    MAX_WIGGLE_POINTS = 2000

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._metadata: Optional[FileMetadata] = None
        self._frame_data: Optional[np.ndarray] = None
        self._y_start_point = 1
        self._y_end_point = 1
        self._waveform_point = 1
        self._colormap_name = "jet"
        self._vmin: Optional[float] = None
        self._vmax: Optional[float] = None
        self._applied_image_key: Optional[tuple] = None
        self._auto_levels: Optional[tuple[float, float]] = None
        self._view_history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._default_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._last_view_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._tracking_enabled = True
        self._plot_mode = "image"
        self._wiggle_items: list[pg.PlotDataItem] = []
        self._wiggle_fills: list[pg.PlotDataItem] = []
        self._wiggle_baselines: list[pg.PlotDataItem] = []
        self._wiggle_stored_channels: list[int] = []
        self._wiggle_stored_fill = False
        self._wiggle_stored_fill_negative = False
        self._wiggle_display_cache: list = []
        self._w_stats_key: tuple | None = None
        self._w_stats: dict | None = None
        self._wiggle_params: dict = {
            "scale_mode": "全局最大值",
            "gain_db": 0.0,
            "flip": False,
            "clip": None,
            "fill": True,
            "fill_negative": False,
            "spacing_mode": "地震道",
            "spacing_value": 1,
        }

        axis_font = QFont("Times New Roman", 18)
        label_font = QFont("Times New Roman", 18)

        self.time_axis = TimeAxisItem(orientation="bottom")
        self.distance_axis = DistanceAxisItem(orientation="left")
        self.plot_widget = pg.PlotWidget(
            axisItems={"bottom": self.time_axis, "left": self.distance_axis}
        )
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel("left", "Point Index")
        self.plot_widget.setLabel("bottom", "Time")
        style_axis(self.distance_axis, axis_font, label_font)
        style_axis(self.plot_widget.getAxis("bottom"), axis_font, label_font)
        # Fix the left-axis width so PlotWidget and WaveformPlot stay aligned:
        # when the visible point range grows (e.g. huge point counts), the
        # Point Index tick labels get wider and would otherwise shift the
        # lower plot area to the right of the waveform plot.
        self.distance_axis.setWidth(96)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_widget.addItem(self.image_item)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._emit_view_range_changed)
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._schedule_lod_refresh)
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._schedule_wiggle_refresh)
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(self._record_manual_view)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        self._lod_timer = QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.setInterval(30)
        self._lod_timer.timeout.connect(self._refresh_image_lod)

        self._wiggle_timer = QTimer(self)
        self._wiggle_timer.setSingleShot(True)
        self._wiggle_timer.setInterval(25)
        self._wiggle_timer.timeout.connect(self._refresh_wiggle)

        self.waveform_axis = TimeAxisItem(orientation="bottom")
        self.waveform_plot = pg.PlotWidget(axisItems={"bottom": self.waveform_axis})
        self.waveform_plot.setBackground("w")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.15)
        self.waveform_plot.setLabel("left", "Phase", units="rad")
        self.waveform_plot.setLabel("bottom", "Time")
        self.waveform_plot.setTitle("指定位置点时域波形", color=AXIS_TEXT_COLOR)
        style_axis(self.waveform_plot.getAxis("left"), axis_font, label_font)
        style_axis(self.waveform_plot.getAxis("bottom"), axis_font, label_font)
        # Same fixed left-axis width as the space-time plot to keep the two
        # subplot time axes aligned on the left edge.
        self.waveform_plot.getAxis("left").setWidth(96)
        self.waveform_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.waveform_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen(color="#0f5ea8", width=1.8))

        self.waveform_plot.setXLink(self.plot_widget)

        self.waveform_row_widget = QWidget()
        waveform_row_layout = QHBoxLayout(self.waveform_row_widget)
        waveform_row_layout.setContentsMargins(0, 0, 0, 0)
        waveform_row_layout.setSpacing(6)
        waveform_row_layout.addWidget(self.waveform_plot, 1)
        self.waveform_right_spacer = QWidget()
        self.waveform_right_spacer.setFixedWidth(150)
        waveform_row_layout.addWidget(self.waveform_right_spacer)

        self.histogram_widget = pg.HistogramLUTWidget(orientation="vertical")
        self.histogram_widget.setImageItem(self.image_item)
        self.histogram_widget.setFixedWidth(150)
        self.histogram_widget.setBackground("w")
        self.histogram_widget.item.axis.setLabel(text="Amplitude", units="rad")
        style_axis(self.histogram_widget.item.axis, axis_font, label_font)
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
        layout.addWidget(self.waveform_row_widget, 2)
        layout.addWidget(lower_row_widget, 6)

    def set_waveform_visible(self, visible: bool) -> None:
        """Show/hide the single-channel time-domain plot above the main plot."""
        self.waveform_row_widget.setVisible(bool(visible))

    def set_data(
        self,
        metadata: FileMetadata,
        frame_data: np.ndarray,
        start_point: int,
        end_point: int,
        waveform_point: int,
        vmin: float | None = None,
        vmax: float | None = None,
        colormap_name: str = "jet",
    ) -> None:
        """Update the displayed image and linked waveform view.

        ``frame_data`` is the full ``frames x points`` processed matrix; the
        visible point slice is a view into it (no copy), and the image is
        decimated on the fly for display while keeping full resolution for zoom.
        """
        self._metadata = metadata
        self._frame_data = frame_data
        self._y_start_point = start_point
        self._y_end_point = end_point
        self._waveform_point = waveform_point
        self._colormap_name = colormap_name
        self._vmin = vmin
        self._vmax = vmax
        self._view_history.clear()
        self._default_range = None
        self._last_view_range = None
        self._applied_image_key = None
        self._auto_levels = self._compute_auto_levels() if (vmin is None or vmax is None) else None

        self._apply_colormap(colormap_name)
        self.time_axis.set_reference_time(metadata.start_time)
        self.waveform_axis.set_reference_time(metadata.start_time)

        self.plot_widget.setLimits(
            xMin=0.0,
            xMax=metadata.duration_s,
            yMin=start_point,
            yMax=end_point + 1,
        )
        self.waveform_plot.setLimits(xMin=0.0, xMax=metadata.duration_s)

        self._refresh_image(full=True)
        self._update_waveform_curve()
        self.reset_view(clear_history=True)
        self._sync_visible_items()

    def update_waveform_point(self, waveform_point: int) -> None:
        """Update the waveform subplot to show a different point."""
        self._waveform_point = waveform_point
        self._update_waveform_curve()

    def update_colormap(self, colormap_name: str) -> None:
        """Update the colormap immediately without reloading file data."""
        self._colormap_name = colormap_name
        self._apply_colormap(colormap_name)

    def set_y_axis_mode(
        self,
        spacing_m: float,
        first_pos_m: float,
        mode: str,
        unit: str,
    ) -> None:
        """Switch the left axis between channel number and physical distance."""
        distance_mode = str(mode).strip() == "距离"
        self.distance_axis.configure(
            spacing_m=spacing_m,
            first_pos_m=first_pos_m,
            distance_mode=distance_mode,
            unit=unit,
        )
        if distance_mode:
            self.distance_axis.setLabel(text="Distance", units=unit)
        else:
            self.distance_axis.setLabel(text="Point Index", units="")

    def reset_view(self, clear_history: bool = False) -> None:
        """Reset the view to show the full visible image extent."""
        if self._metadata is None or self._frame_data is None:
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

    def set_plot_mode(self, mode: str) -> None:
        """Switch Tab1 lower plot between the time-space color map and wiggle."""
        new_mode = "image" if str(mode).strip() == "时空颜色图" else "wiggle"
        if new_mode == self._plot_mode:
            return
        self._plot_mode = new_mode
        self._sync_visible_items()
        # Show the full time/channel extent on every mode switch so a zoom
        # window left over from the other mode does not make the Wiggle (or
        # the color map) look truncated on the right.
        self.reset_view(clear_history=False)

    def set_wiggle_params(self, params: dict) -> None:
        """Update wiggle rendering parameters and refresh (debounced)."""
        self._wiggle_params.update(params)
        if self._plot_mode == "wiggle" and not self._wiggle_timer.isActive():
            self._wiggle_timer.start()

    def _sync_visible_items(self) -> None:
        image_mode = self._plot_mode == "image"
        self.image_item.setVisible(image_mode)
        self.histogram_widget.setVisible(image_mode)
        # Keep the single-trace plot and the lower plot the same width: the
        # noise map hides the histogram, so drop the matching right spacer too.
        self.waveform_right_spacer.setVisible(image_mode)
        if image_mode:
            for item in [*self._wiggle_items, *self._wiggle_fills, *self._wiggle_baselines]:
                item.setVisible(False)
        else:
            self._refresh_wiggle()

    def _schedule_wiggle_refresh(self, *_args) -> None:
        if self._plot_mode != "wiggle":
            return
        if self._wiggle_params.get("spacing_mode") == "像素点" and not self._wiggle_timer.isActive():
            self._wiggle_timer.start()

    def _wiggle_spacing_step(self) -> int:
        """Return the channel step between two drawn traces."""
        params = self._wiggle_params
        if params.get("spacing_mode") == "像素点":
            pixels = float(params.get("spacing_value") or 20.0)
            view_box = self.plot_widget.getViewBox()
            rect = view_box.viewRect()
            viewbox_height_px = float(view_box.height())
            if viewbox_height_px < 100:
                viewbox_height_px = 600.0
            y_span = max(rect.height(), 1e-9)
            channels_per_pixel = y_span / viewbox_height_px
            return max(1, int(round(pixels * channels_per_pixel)))
        return max(1, int(round(params.get("spacing_value") or 1)))

    def _refresh_wiggle(self) -> None:
        """Refresh the multi-channel wiggle profile.

        Amplitude-only changes (gain/flip/clip/scale mode) update the existing
        items in place so dragging the gain slider stays smooth; only changes
        that alter the trace set (spacing, fill toggles) trigger a rebuild.
        """
        if self._metadata is None or self._frame_data is None or self._plot_mode != "wiggle":
            self._clear_wiggle_items()
            return

        params = self._wiggle_params
        channels = self._compute_wiggle_channels()
        needs_rebuild = (
            not self._wiggle_items
            or self._wiggle_stored_channels != list(channels)
            or self._wiggle_stored_fill != bool(params.get("fill"))
            or self._wiggle_stored_fill_negative != bool(params.get("fill_negative"))
        )
        if needs_rebuild:
            self._rebuild_wiggle(channels)
        else:
            self._update_wiggle_data()

    def _clear_wiggle_items(self) -> None:
        for item in [*self._wiggle_items, *self._wiggle_fills, *self._wiggle_baselines]:
            self.plot_widget.removeItem(item)
        self._wiggle_items.clear()
        self._wiggle_fills.clear()
        self._wiggle_baselines.clear()
        self._wiggle_stored_channels = []

    def _compute_wiggle_channels(self) -> list[int]:
        step = self._wiggle_spacing_step()
        channels = np.arange(self._y_start_point, self._y_end_point + 1, step, dtype=int)
        if channels.size == 0:
            return []
        if channels.size > self.MAX_WIGGLE_TRACES:
            extra = int(np.ceil(channels.size / self.MAX_WIGGLE_TRACES)) * step
            channels = np.arange(self._y_start_point, self._y_end_point + 1, extra, dtype=int)
        return [int(c) for c in channels]

    def _compute_wiggle_stats(self):
        """Bounded (subsampled) nan-safe statistics for the current point slice.

        Statistics only depend on the loaded data and the point range, so they
        are cached and reused across amplitude-only wiggle updates.
        """
        data = self._frame_data
        if data is None:
            return None
        key = (id(data), self._y_start_point, self._y_end_point)
        if self._w_stats_key == key and self._w_stats is not None:
            self._w_stats["matrix"] = data[:, slice(self._y_start_point - 1, self._y_end_point)]
            return self._w_stats
        point_slice = slice(self._y_start_point - 1, self._y_end_point)
        matrix = data[:, point_slice]
        if matrix.size == 0:
            return None
        rows, cols = matrix.shape
        target_samples = 2_000_000
        stat_step = max(1, int(np.ceil(rows * cols / target_samples)))
        abs_sub = np.abs(matrix[::stat_step, :])
        col_amp = np.nanmax(abs_sub, axis=0)
        sub_len = abs_sub.shape[0]
        col_sqsum = np.nansum(np.square(abs_sub), axis=0)
        slice_amp = float(np.nanmax(col_amp)) if col_amp.size else 0.0
        slice_rms = float(np.sqrt(col_sqsum.sum() / max(sub_len * cols, 1)))
        slice_abs_mean = float(np.nanmean(abs_sub))
        self._w_stats = {
            "matrix": matrix,
            "rows": rows,
            "cols": cols,
            "col_amp": col_amp,
            "col_sqsum": col_sqsum,
            "sub_len": sub_len,
            "slice_amp": slice_amp,
            "slice_rms": slice_rms,
            "slice_abs_mean": slice_abs_mean,
        }
        self._w_stats_key = key
        return self._w_stats

    def _select_wiggle_scale(self, column: int, stats: dict) -> float:
        scale_mode = str(self._wiggle_params.get("scale_mode") or "全局最大值")
        col_amp = stats["col_amp"]
        col_sqsum = stats["col_sqsum"]
        sub_len = stats["sub_len"]
        cols = stats["cols"]
        lo = max(0, column - 1)
        hi = min(cols, column + 2)
        tri_amp = float(np.nanmax(col_amp[lo:hi])) if lo < hi else 0.0
        tri_count = max((hi - lo) * sub_len, 1)
        tri_rms = float(np.sqrt(col_sqsum[lo:hi].sum() / tri_count))
        col_max = float(col_amp[column]) if col_amp.size else 0.0
        col_rms = float(np.sqrt(col_sqsum[column] / max(sub_len, 1)))
        if scale_mode == "全局最大值":
            scale = 1.0 / stats["slice_amp"] if stats["slice_amp"] > 1e-12 else 1.0
        elif scale_mode == "三道最大值":
            scale = 1.0 / tri_amp if tri_amp > 1e-12 else 1.0
        elif scale_mode == "单道最大值":
            scale = 1.0 / col_max if col_max > 1e-12 else 1.0
        elif scale_mode == "全局均方根":
            scale = 1.0 / stats["slice_rms"] if stats["slice_rms"] > 1e-12 else 1.0
        elif scale_mode == "三道均方根":
            scale = 1.0 / tri_rms if tri_rms > 1e-12 else 1.0
        elif scale_mode == "单道均方根":
            scale = 1.0 / col_rms if col_rms > 1e-12 else 1.0
        elif scale_mode == "全局平均值":
            scale = 1.0 / stats["slice_abs_mean"] if stats["slice_abs_mean"] > 1e-12 else 1.0
        else:
            scale = 1.0
        return scale

    def _decimate_raw_trace(self, trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decimate a raw trace once (per rebuild); returns (x_time, envelope)."""
        sample_rate = self._metadata.sample_rate_hz
        if trace.size > self.MAX_WIGGLE_POINTS:
            x_idx, envelope = _decimate_peaks(trace, self.MAX_WIGGLE_POINTS)
            x_time = x_idx / sample_rate
        else:
            x_time = np.arange(trace.size, dtype=np.float64) / sample_rate
            envelope = np.ascontiguousarray(trace)
        return x_time, envelope

    def _channel_display_arrays(self, index: int, stats: dict):
        """Compute (x, y, fill) display arrays for a cached trace envelope."""
        x_time, envelope, baseline = self._wiggle_display_cache[index]
        if envelope is None:
            return None, None, None
        params = self._wiggle_params
        channel = self._wiggle_stored_channels[index]
        column = int(channel - self._y_start_point)
        scale = self._select_wiggle_scale(column, stats)
        gain = 10.0 ** (float(params.get("gain_db") or 0.0) / 20.0)
        scaled = (envelope * (scale * gain)).astype(np.float32, copy=False)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        clip = params.get("clip")
        if clip is not None and clip > 0:
            np.clip(scaled, -clip, clip, out=scaled)
        if params.get("flip"):
            scaled = -scaled
        y_plot = baseline + scaled
        if params.get("fill"):
            if params.get("fill_negative"):
                fill_y = y_plot
            else:
                fill_y = baseline + np.maximum(scaled, 0.0)
            return x_time, np.ascontiguousarray(y_plot), np.ascontiguousarray(fill_y)
        return x_time, np.ascontiguousarray(y_plot), None

    def _rebuild_wiggle(self, channels: list[int]) -> None:
        self._clear_wiggle_items()
        params = self._wiggle_params
        stats = self._compute_wiggle_stats()
        if stats is None or not channels:
            return

        matrix = stats["matrix"]
        self._wiggle_display_cache = []
        for channel in channels:
            column = int(channel - self._y_start_point)
            trace = matrix[:, column]
            if trace.shape[0] == 0:
                self._wiggle_display_cache.append((None, None, float(channel)))
                continue
            x_time, envelope = self._decimate_raw_trace(trace)
            # Mark envelopes that have no finite content so updates skip them.
            if not np.isfinite(envelope).any():
                envelope = np.zeros_like(envelope)
            self._wiggle_display_cache.append((x_time, envelope, float(channel)))
        self._wiggle_stored_channels = list(channels)

        duration_s = self._metadata.duration_s
        x_axis_display = [0.0, duration_s]
        black_pen = pg.mkPen(color="#000000", width=1.0)
        gray_pen = pg.mkPen(color="#9aa0a6", width=0.8)
        fill_brush = QBrush(QColor("#000000"))
        fill = bool(params.get("fill"))

        for index in range(len(channels)):
            x_time, y_plot, fill_y = self._channel_display_arrays(index, stats)
            if x_time is None:
                continue
            baseline = self._wiggle_display_cache[index][2]

            trace_item = pg.PlotDataItem(x=x_time, y=y_plot, pen=black_pen)
            self.plot_widget.addItem(trace_item)
            self._wiggle_items.append(trace_item)

            baseline_item = pg.PlotDataItem(x=x_axis_display, y=[baseline, baseline], pen=gray_pen)
            self.plot_widget.addItem(baseline_item)
            self._wiggle_baselines.append(baseline_item)

            if fill and fill_y is not None:
                fill_item = pg.PlotDataItem(
                    x=x_time, y=fill_y, pen=None, brush=fill_brush, fillLevel=baseline
                )
                self.plot_widget.addItem(fill_item)
                fill_item.setZValue(-1)
                self._wiggle_fills.append(fill_item)

        self._wiggle_stored_fill = fill
        self._wiggle_stored_fill_negative = bool(params.get("fill_negative"))

    def _update_wiggle_data(self) -> None:
        """Update existing wiggle items in place (gain/flip/clip/scale changes)."""
        stats = self._compute_wiggle_stats()
        if stats is None or not self._wiggle_stored_channels:
            return
        for index in range(len(self._wiggle_stored_channels)):
            if index >= len(self._wiggle_items):
                break
            x_time, y_plot, fill_y = self._channel_display_arrays(index, stats)
            if x_time is None:
                continue
            self._wiggle_items[index].setData(x=x_time, y=y_plot)
            if self._wiggle_fills and index < len(self._wiggle_fills):
                if fill_y is not None:
                    self._wiggle_fills[index].setData(x=x_time, y=fill_y)
                else:
                    self._wiggle_fills[index].clear()

    def _on_scene_mouse_moved(self, pos) -> None:
        """Report (channel, distance, time, amplitude) on hover in wiggle mode."""
        if self._plot_mode != "wiggle":
            return
        if self._metadata is None or self._frame_data is None:
            return
        view_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
        channel = int(round(view_point.y()))
        if channel < 1 or channel > self._frame_data.shape[1]:
            return
        clamped_time = min(max(view_point.x(), 0.0), self._metadata.duration_s)
        frame = int(clamped_time * self._metadata.sample_rate_hz)
        frame = min(max(frame, 0), self._frame_data.shape[0] - 1)
        amplitude = float(self._frame_data[frame, channel - 1])
        spacing_m = getattr(self.distance_axis, "_spacing_m", 5.0) or 5.0
        first_pos_m = getattr(self.distance_axis, "_first_pos_m", 0.0) or 0.0
        distance = first_pos_m + (channel - 1) * spacing_m
        self.cursor_info.emit(
            f"道号 {channel:g} | 距离 {distance:g} m | 时间 {clamped_time:.3f} s | 振幅 {amplitude:.4g} rad"
        )

    def _update_waveform_curve(self) -> None:
        if self._metadata is None or self._frame_data is None:
            self.waveform_curve.clear()
            return

        point_index = self._y_start_point - 1 + (self._waveform_point - self._y_start_point)
        if point_index < 0 or point_index >= self._frame_data.shape[1]:
            self.waveform_curve.clear()
            return

        waveform = np.ascontiguousarray(self._frame_data[:, point_index], dtype=np.float32)
        x_values = np.arange(waveform.shape[0], dtype=np.float32) / self._metadata.sample_rate_hz
        self.waveform_curve.setData(x=x_values, y=waveform)
        # Display-level downsampling only: 'peak' mode keeps the min/max envelope
        # of every bin, so extreme values are never dropped while GPU upload stays
        # small. The full-resolution array is unchanged.
        self.waveform_curve.setDownsampling(auto=True, method="peak")
        self.waveform_curve.setClipToView(True)
        self.waveform_plot.setTitle(f"指定位置点 {self._waveform_point} 时域波形")
        self._reset_waveform_y_range(waveform)

    def _frame_slice_for_view(self) -> slice:
        """Return the frame slice to upload for the current visible window.

        When the visible window is small enough to display at full resolution,
        the slice is the visible window plus a small safety margin so that short
        scrolls never reveal a blank area. When zoomed far out, the whole time
        axis is returned as one coarse (downsampled) image so that horizontal
        scrolling never reveals blank areas either.
        """
        if self._metadata is None or self._frame_data is None:
            return slice(0, 1)

        x_range, _ = self.current_ranges()
        sample_rate = self._metadata.sample_rate_hz
        frame_count = self._frame_data.shape[0]
        frame_start = max(0, int(np.floor(x_range[0] * sample_rate)))
        frame_end = min(frame_count, int(np.ceil(x_range[1] * sample_rate)))
        frame_end = max(frame_end, frame_start + 1)

        span = frame_end - frame_start
        if span <= MAX_IMAGE_DISPLAY_COLUMNS:
            margin = max(1, int(round(span * 0.2)))
            return slice(max(0, frame_start - margin), min(frame_count, frame_end + margin))
        return slice(0, frame_count)

    def _refresh_image(self, full: bool = False) -> None:
        """(Re)slice the full-resolution data and upload a decimated image."""
        if self._metadata is None or self._frame_data is None:
            return

        npoints = self._y_end_point - self._y_start_point + 1
        if npoints < 1:
            return

        frame_slice = slice(0, self._frame_data.shape[0]) if full else self._frame_slice_for_view()
        point_slice = slice(self._y_start_point - 1, self._y_end_point)
        frame0 = max(0, frame_slice.start or 0)

        image_pt = self._frame_data[frame_slice, point_slice].T
        display, samples_per_col = downsample_image_for_display(image_pt, MAX_IMAGE_DISPLAY_COLUMNS)

        key = (
            self._y_start_point,
            self._y_end_point,
            frame_slice.start,
            frame_slice.stop,
            display.shape[1],
        )
        if key == self._applied_image_key:
            return

        levels = self._resolve_levels(image_pt, self._vmin, self._vmax)
        self.image_item.setImage(display, autoLevels=False, levels=levels)
        self.histogram_widget.item.setLevels(*levels)

        transform = QTransform()
        transform.translate(frame0 / self._metadata.sample_rate_hz, self._y_start_point)
        transform.scale(samples_per_col / self._metadata.sample_rate_hz, 1.0)
        self.image_item.setTransform(transform)
        self._applied_image_key = key

    def _schedule_lod_refresh(self, *_args) -> None:
        if not self._lod_timer.isActive():
            self._lod_timer.start()

    def _refresh_image_lod(self) -> None:
        self._refresh_image(full=False)

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
            if self._auto_levels is not None:
                auto_min, auto_max = self._auto_levels
            else:
                auto_min, auto_max = robust_min_max(image_data)
            if not np.isfinite(auto_min) or not np.isfinite(auto_max):
                return -1.0, 1.0
            if auto_min == auto_max:
                auto_max = auto_min + 1e-6
            return auto_min, auto_max
        return vmin, vmax

    def _compute_auto_levels(self) -> Optional[tuple[float, float]]:
        """Estimate colour levels once from the full point range.

        Computing the levels from the full time axis (instead of the currently
        visible slice) keeps the colormap stable while zooming/panning, so the
        image does not appear lighter when zoomed in and darker when zoomed out.
        """
        if self._metadata is None or self._frame_data is None:
            return None

        point_slice = slice(self._y_start_point - 1, self._y_end_point)
        data = self._frame_data[:, point_slice]
        if data.size == 0:
            return None

        max_samples = 1_000_000
        if data.size <= max_samples:
            auto_min = float(np.nanmin(data))
            auto_max = float(np.nanmax(data))
        else:
            npoints = max(data.shape[1], 1)
            per_point = max(1, int(max_samples // npoints))
            step = max(1, int(np.ceil(data.shape[0] / per_point)))
            subset = data[::step, :]
            auto_min = float(np.nanmin(subset))
            auto_max = float(np.nanmax(subset))

        if not np.isfinite(auto_min) or not np.isfinite(auto_max):
            return None
        if auto_min == auto_max:
            auto_max = auto_min + 1e-6
        return auto_min, auto_max

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
