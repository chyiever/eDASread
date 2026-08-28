"""Extra analysis tabs for channel traces, PSD and f-k spectra."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from scipy import signal

from scripts.core.models import FileMetadata
from scripts.processing.view_data import robust_abs_percentile, robust_median
from scripts.processing.time_frequency import build_log_frequency_display_grid
from scripts.render.plot_style import AXIS_TEXT_COLOR, style_axis
from scripts.render.space_time_view import COLOR_MAPS
from scripts.render.time_axis import TimeAxisItem


CURVE_COLORS = [
    "#0f5ea8",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#0072b2",
    "#e69f00",
    "#56b4e9",
    "#7f7f7f",
]


class TimeSelectionViewBox(pg.ViewBox):
    """ViewBox that supports Ctrl+left-drag time-range selection.

    A plain left-drag keeps the default ``RectMode`` zoom behaviour. Holding
    ``Ctrl`` while dragging instead selects a time range (without zooming),
    emitting ``region_dragged`` continuously and ``region_selected`` on release.
    """

    region_dragged = pyqtSignal(float, float)
    region_selected = pyqtSignal(float, float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._selecting = False
        self._select_start_x: float | None = None
        self.setMouseMode(pg.ViewBox.RectMode)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and (ev.modifiers() & Qt.ControlModifier):
            self._selecting = True
            point = self.mapToView(ev.pos())
            self._select_start_x = float(point.x())
            self.region_dragged.emit(self._select_start_x, self._select_start_x)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if self._selecting:
            if ev.button() & Qt.LeftButton:
                point = self.mapToView(ev.pos())
                self.region_dragged.emit(self._select_start_x, float(point.x()))
            ev.accept()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseReleaseEvent(self, ev) -> None:
        if self._selecting:
            self._selecting = False
            point = self.mapToView(ev.pos())
            x0 = min(self._select_start_x, float(point.x()))
            x1 = max(self._select_start_x, float(point.x()))
            if x1 - x0 > 1e-9:
                self.region_selected.emit(x0, x1)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class FrequencyLogAxis(pg.AxisItem):
    """Log-frequency axis with plain decimal tick labels.

    pyqtgraph's default log tick formatter uses ``%0.1g`` which renders values
    such as 20/100 as ``2·10¹`` / ``10²``. This subclass instead renders plain
    ``20`` / ``100`` labels so the Hz scale is readable and does not crowd.
    """

    def logTickStrings(self, values, scale, spacing):
        strings = []
        for value in values:
            number = float(10.0 ** value * scale)
            if number >= 1.0 and abs(number - round(number)) < 1e-6 * max(1.0, number):
                strings.append(str(int(round(number))))
            else:
                strings.append(f"{number:g}")
        return strings


class ChannelAnalysisView(QWidget):
    """Tab2 widget for selected-channel time traces and PSD curves."""

    plot_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channels_edit = QLineEdit("5,10,15")
        self.psd_nperseg_edit = QLineEdit("auto")
        self.psd_overlap_edit = QLineEdit("50")
        self.psd_fmin_edit = QLineEdit()
        self.psd_fmax_edit = QLineEdit()
        self.psd_ymin_edit = QLineEdit()
        self.psd_ymax_edit = QLineEdit()
        self.plot_button = QPushButton("绘制")

        ui_font = QFont("Times New Roman", 10)
        axis_font = QFont("Times New Roman", 18)
        label_font = QFont("Times New Roman", 18)
        for widget in [
            self.channels_edit,
            self.psd_nperseg_edit,
            self.psd_overlap_edit,
            self.psd_fmin_edit,
            self.psd_fmax_edit,
            self.psd_ymin_edit,
            self.psd_ymax_edit,
            self.plot_button,
        ]:
            widget.setFont(ui_font)
        self.psd_fmin_edit.setPlaceholderText("auto")
        self.psd_fmax_edit.setPlaceholderText("auto")
        self.psd_ymin_edit.setPlaceholderText("auto")
        self.psd_ymax_edit.setPlaceholderText("auto")

        controls = QGroupBox("通道分析")
        controls.setFont(ui_font)
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(6)
        controls_layout.addWidget(self._labeled_widget("通道", self.channels_edit), 0, 0, 1, 2)
        controls_layout.addWidget(self._labeled_widget("PSD窗长", self.psd_nperseg_edit), 0, 2)
        controls_layout.addWidget(self._labeled_widget("重叠(%)", self.psd_overlap_edit), 0, 3)
        controls_layout.addWidget(self._labeled_widget("FMin", self.psd_fmin_edit), 1, 0)
        controls_layout.addWidget(self._labeled_widget("FMax", self.psd_fmax_edit), 1, 1)
        controls_layout.addWidget(self._labeled_widget("PSD Min", self.psd_ymin_edit), 1, 2)
        controls_layout.addWidget(self._labeled_widget("PSD Max", self.psd_ymax_edit), 1, 3)
        controls_layout.addWidget(self.plot_button, 0, 4, 2, 1)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 1)
        controls_layout.setColumnStretch(3, 1)

        self.time_axis = TimeAxisItem(orientation="bottom")
        self._time_view_box = TimeSelectionViewBox()
        self.time_plot = pg.PlotWidget(axisItems={"bottom": self.time_axis}, viewBox=self._time_view_box)
        self.psd_plot = pg.PlotWidget(axisItems={"bottom": FrequencyLogAxis(orientation="bottom")})

        self.time_plot.setLabel("left", "Amplitude", units="rad")
        self.time_plot.setLabel("bottom", "Time")
        self.time_plot.setTitle("指定通道时域波形", color=AXIS_TEXT_COLOR)
        self._style_plot(self.time_plot, axis_font, label_font)
        self.time_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.time_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.time_legend = self.time_plot.addLegend(offset=(10, 10))

        self.psd_region = pg.LinearRegionItem(
            values=(0.0, 1.0),
            brush=pg.mkBrush(255, 230, 60, 70),
            pen=pg.mkPen(200, 170, 0, 150),
            movable=False,
        )
        self.psd_region.setZValue(-20)
        self.time_plot.addItem(self.psd_region)
        self.psd_region.hide()
        self._time_view_box.region_dragged.connect(self._on_region_dragged)
        self._time_view_box.region_selected.connect(self._on_region_selected)

        self._metadata: FileMetadata | None = None
        self._data: np.ndarray | None = None
        self._channels: list[int] = []
        self._traces: list[np.ndarray] = []
        self._offset_step = 1.0
        self._psd_nperseg: int | None = None
        self._psd_overlap_percent = 50.0
        self._psd_x_range: tuple[float | None, float | None] = (None, None)
        self._psd_y_range: tuple[float | None, float | None] = (None, None)
        self._psd_time_range: tuple[float, float] | None = None

        self.psd_plot.setLabel("left", "PSD", units="dB")
        self.psd_plot.setLabel("bottom", "Frequency", units="Hz")
        self.psd_plot.setTitle("功率谱密度", color=AXIS_TEXT_COLOR)
        self._style_plot(self.psd_plot, axis_font, label_font)
        self.psd_plot.setLogMode(x=True)
        self.psd_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.psd_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.psd_legend = self.psd_plot.addLegend(offset=(10, 10))

        # Log-frequency axis: label one tick per decade (10, 100, ...) and drop
        # the crowded 1..9 minor labels so the Hz scale never overlaps.
        self.psd_x_axis = self.psd_plot.getAxis("bottom")
        self.psd_x_axis.setTickSpacing(major=1.0, minor=0.25)
        self.psd_x_axis.setStyle(maxTextLevel=0)
        # Keep the unit label as "Hz": pyqtgraph otherwise auto-prefixes the
        # label to "kHz" once the range reaches a few kilohertz.
        self.psd_x_axis.enableAutoSIPrefix(False)
        self.psd_x_axis.labelUnitPrefix = ""

        self._active_plot: pg.PlotWidget | None = None
        self.time_plot.scene().sigMouseClicked.connect(self._mark_time_active)
        self.psd_plot.scene().sigMouseClicked.connect(self._mark_psd_active)

        plot_splitter = QSplitter(Qt.Vertical)
        plot_splitter.setChildrenCollapsible(False)
        plot_splitter.addWidget(self.time_plot)
        plot_splitter.addWidget(self.psd_plot)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 1)
        plot_splitter.setHandleWidth(8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(controls)
        layout.addWidget(plot_splitter, 1)

        self.plot_button.clicked.connect(self.plot_requested)
        self.channels_edit.returnPressed.connect(self.plot_requested)
        self.psd_nperseg_edit.returnPressed.connect(self.plot_requested)
        self.psd_overlap_edit.returnPressed.connect(self.plot_requested)

    def set_data(
        self,
        metadata: FileMetadata,
        data: np.ndarray,
        channels: Iterable[int],
        *,
        psd_nperseg: int | None = None,
        psd_overlap_percent: float = 50.0,
        psd_x_range: tuple[float | None, float | None] = (None, None),
        psd_y_range: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """Draw selected channels from a ``frames x channels`` data matrix."""
        channel_list = list(channels)
        if not channel_list:
            raise ValueError("请至少输入一个通道。")

        matrix = np.asarray(data, dtype=np.float32)
        for channel in channel_list:
            if channel < 1 or channel > matrix.shape[1]:
                raise ValueError(f"通道 {channel} 超出有效范围 1-{matrix.shape[1]}。")

        self._metadata = metadata
        self._data = matrix
        self._channels = channel_list
        self._psd_nperseg = psd_nperseg
        self._psd_overlap_percent = psd_overlap_percent
        self._psd_x_range = psd_x_range
        self._psd_y_range = psd_y_range
        self._psd_time_range = None

        self._prepare_traces()
        self._render_time_plot()
        self._render_psd()

    def _prepare_traces(self) -> None:
        self._traces = []
        for channel in self._channels:
            trace = np.asarray(self._data[:, channel - 1], dtype=np.float32)
            self._traces.append(trace - robust_median(trace))
        self._offset_step = self._resolve_offset_step(self._traces)

    def _render_time_plot(self) -> None:
        self.time_plot.clear()
        self.time_legend.clear()
        self.time_axis.set_reference_time(self._metadata.start_time)

        frames = self._data.shape[0]
        x_values = np.arange(frames, dtype=np.float32) / self._metadata.sample_rate_hz

        for index, (channel, trace) in enumerate(zip(self._channels, self._traces)):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            pen = pg.mkPen(color=color, width=1.6)
            curve = self.time_plot.plot(
                x_values,
                trace + self._offset_step * index,
                pen=pen,
                name=f"CH {channel}",
            )
            # Display-level decimation only: 'peak' keeps min/max envelopes so no
            # extreme sample is dropped, while GPU upload stays small for very long
            # traces. The stored data array is untouched.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)

        self.time_plot.setTitle(
            "指定通道时域波形：" + ", ".join(str(ch) for ch in self._channels),
            color=AXIS_TEXT_COLOR,
        )

        # ``clear()`` also removes the selection region, so re-add it and restore
        # its visibility (always hidden after a fresh ``set_data`` call).
        self.time_plot.addItem(self.psd_region)
        if self._psd_time_range is not None:
            self.psd_region.setRegion(self._psd_time_range)
            self.psd_region.show()
        else:
            self.psd_region.hide()

        self.time_plot.enableAutoRange()

    def _render_psd(self) -> None:
        self.psd_plot.clear()
        self.psd_legend.clear()

        for index, (channel, trace) in enumerate(zip(self._channels, self._traces)):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            pen = pg.mkPen(color=color, width=1.6)
            psd_trace = self._slice_psd_trace(trace)
            freq_hz, psd_db = self._compute_psd(
                psd_trace,
                self._metadata.sample_rate_hz,
                nperseg=self._psd_nperseg,
                overlap_percent=self._psd_overlap_percent,
            )
            self.psd_plot.plot(freq_hz, psd_db, pen=pen, name=f"CH {channel}")

        title = "功率谱密度"
        if self._psd_time_range is not None:
            title += f"（{self._psd_time_range[0]:.2f}-{self._psd_time_range[1]:.2f}s）"
        self.psd_plot.setTitle(title, color=AXIS_TEXT_COLOR)
        self.psd_plot.enableAutoRange()
        self._apply_psd_ranges(self._psd_x_range, self._psd_y_range)

    def _slice_psd_trace(self, trace: np.ndarray) -> np.ndarray:
        if self._psd_time_range is None or self._metadata is None:
            return trace
        t0, t1 = self._psd_time_range
        sample_rate = self._metadata.sample_rate_hz
        index0 = max(0, int(np.floor(t0 * sample_rate)))
        index1 = min(trace.size, int(np.ceil(t1 * sample_rate)))
        index1 = max(index1, index0 + 1)
        return trace[index0:index1]

    def _on_region_dragged(self, x0: float, x1: float) -> None:
        if self._data is None:
            return
        left = min(x0, x1)
        right = max(x0, x1)
        self.psd_region.setRegion((left, right))
        self.psd_region.show()

    def _on_region_selected(self, x0: float, x1: float) -> None:
        if self._data is None:
            return
        self._psd_time_range = (min(x0, x1), max(x0, x1))
        self.psd_region.setRegion(self._psd_time_range)
        self.psd_region.show()

        sample_rate = self._metadata.sample_rate_hz
        duration = self._psd_time_range[1] - self._psd_time_range[0]
        if int(np.ceil(duration * sample_rate)) < 8:
            return
        self._render_psd()

    def _mark_time_active(self, _ev) -> None:
        self._active_plot = self.time_plot

    def _mark_psd_active(self, _ev) -> None:
        self._active_plot = self.psd_plot

    def pan_active_plot(self, direction: int) -> None:
        """Pan the most recently clicked plot horizontally (keyboard scroll)."""
        plot = self._active_plot or self.time_plot
        if plot is None:
            return

        view_box = plot.getViewBox()
        x_min, x_max = view_box.viewRange()[0]
        span = x_max - x_min
        if span <= 0:
            return

        delta = span * 0.3 * direction
        if plot is self.time_plot:
            if self._metadata is not None and self._data is not None:
                x_limit = (self._data.shape[0] - 1) / self._metadata.sample_rate_hz
                new_min, new_max = self._clamp_pan(x_min, x_max, delta, 0.0, max(x_limit, 1e-9))
            else:
                new_min, new_max = x_min + delta, x_max + delta
        else:
            # PSD x-axis is logarithmic: viewRange returns log10(Hz), so clamp in
            # log domain to keep the view within [10, Nyquist] Hz.
            if self._metadata is not None:
                nyquist = self._metadata.sample_rate_hz / 2.0
                low = np.log10(10.0)
                high = np.log10(max(nyquist, 10.0))
                new_min, new_max = self._clamp_pan(x_min, x_max, delta, low, high)
            else:
                new_min, new_max = x_min + delta, x_max + delta

        view_box.setXRange(new_min, new_max, padding=0.0)

    @staticmethod
    def _clamp_pan(
        x_min: float,
        x_max: float,
        delta: float,
        low: float,
        high: float,
    ) -> tuple[float, float]:
        span = x_max - x_min
        new_min = x_min + delta
        new_max = x_max + delta
        if new_max > high:
            new_max = high
            new_min = high - span
        if new_min < low:
            new_min = low
            new_max = low + span
        return new_min, new_max

    def _compute_psd(
        self,
        trace: np.ndarray,
        sample_rate_hz: float,
        *,
        nperseg: int | None,
        overlap_percent: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if nperseg is None:
            nperseg = min(max(256, int(sample_rate_hz)), 4096, trace.size)
        else:
            nperseg = min(int(nperseg), trace.size)
        if nperseg < 8:
            raise ValueError("PSD 至少需要 8 个采样点。")
        if overlap_percent < 0 or overlap_percent >= 100:
            raise ValueError("PSD 重叠比例必须位于 [0, 100)%。")
        noverlap = int(round(nperseg * overlap_percent / 100.0))
        noverlap = max(0, min(noverlap, nperseg - 1))
        freq_hz, power = signal.welch(
            trace,
            fs=sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        psd_db = 10.0 * np.log10(power + np.finfo(np.float32).eps)
        return np.asarray(freq_hz, dtype=np.float32), np.asarray(psd_db, dtype=np.float32)

    def _resolve_offset_step(self, traces: list[np.ndarray]) -> float:
        robust_ranges = []
        for trace in traces:
            spread = robust_abs_percentile(trace, 95.0)
            if np.isfinite(spread) and spread > 0:
                robust_ranges.append(spread)
        if not robust_ranges:
            return 1.0
        return max(robust_ranges) * 2.4

    def _style_plot(self, plot_widget: pg.PlotWidget, axis_font: QFont, label_font: QFont) -> None:
        plot_widget.setBackground("w")
        plot_widget.showGrid(x=True, y=True, alpha=0.15)
        for axis_name in ("left", "bottom"):
            axis = plot_widget.getAxis(axis_name)
            style_axis(axis, axis_font, label_font)
            tick_length = -8 if axis_name == "left" else 8
            axis.setStyle(tickLength=tick_length)
        plot_widget.getViewBox().setMouseEnabled(x=True, y=True)

    def _apply_psd_ranges(
        self,
        x_range: tuple[float | None, float | None],
        y_range: tuple[float | None, float | None],
    ) -> None:
        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_min is not None and x_max is not None:
            # FMin/FMax are in Hz; the log axis works in log10 domain and needs
            # strictly positive bounds (log10(0) would be -inf).
            x_min = max(float(x_min), 1e-9)
            x_max = max(float(x_max), x_min)
            self.psd_plot.setXRange(np.log10(x_min), np.log10(x_max), padding=0.0)
        elif self._metadata is not None:
            nyquist = self._metadata.sample_rate_hz / 2.0
            if nyquist > 10.0:
                self.psd_plot.setXRange(np.log10(10.0), np.log10(nyquist), padding=0.0)
        if y_min is not None and y_max is not None:
            self.psd_plot.setYRange(y_min, y_max, padding=0.0)

    def _build_control_row(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    def _labeled_widget(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(label_text)
        label.setFont(widget.font())
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return container


class FkSpectrumView(QWidget):
    """Tab3 widget for f-k spectrum display."""

    plot_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel_range_edit = QLineEdit("5-100")
        self.spacing_edit = QLineEdit("5")
        self.spatial_taper_edit = QLineEdit("0")
        self.temporal_taper_edit = QLineEdit("0.05")
        self.nfft_channels_edit = QLineEdit("auto")
        self.nfft_samples_edit = QLineEdit("auto")
        self.remove_bg_checkbox = QCheckBox("扣除背景")
        self.remove_bg_checkbox.setChecked(True)
        self.fmin_edit = QLineEdit()
        self.fmax_edit = QLineEdit()
        self.kmin_edit = QLineEdit()
        self.kmax_edit = QLineEdit()
        self.vmin_db_edit = QLineEdit("-20")
        self.vmax_db_edit = QLineEdit("0")
        self.plot_button = QPushButton("绘制 f-k 谱")

        ui_font = QFont("Times New Roman", 10)
        axis_font = QFont("Times New Roman", 18)
        label_font = QFont("Times New Roman", 18)
        for widget in [
            self.channel_range_edit,
            self.spacing_edit,
            self.spatial_taper_edit,
            self.temporal_taper_edit,
            self.nfft_channels_edit,
            self.nfft_samples_edit,
            self.remove_bg_checkbox,
            self.fmin_edit,
            self.fmax_edit,
            self.kmin_edit,
            self.kmax_edit,
            self.vmin_db_edit,
            self.vmax_db_edit,
            self.plot_button,
        ]:
            widget.setFont(ui_font)
        for edit in [self.fmin_edit, self.fmax_edit, self.kmin_edit, self.kmax_edit]:
            edit.setPlaceholderText("auto")

        controls = QGroupBox("f-k 谱")
        controls.setFont(ui_font)
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)
        controls_layout.addWidget(self._labeled_widget("通道范围", self.channel_range_edit), 0, 0, 1, 2)
        controls_layout.addWidget(self._labeled_widget("间距(m)", self.spacing_edit), 0, 2)
        controls_layout.addWidget(self._labeled_widget("空间taper", self.spatial_taper_edit), 0, 3)
        controls_layout.addWidget(self._labeled_widget("时间taper", self.temporal_taper_edit), 0, 4)
        controls_layout.addWidget(self._labeled_widget("NFFT-K", self.nfft_channels_edit), 1, 0)
        controls_layout.addWidget(self._labeled_widget("NFFT-T", self.nfft_samples_edit), 1, 1)
        controls_layout.addWidget(self._labeled_widget("FMin", self.fmin_edit), 1, 2)
        controls_layout.addWidget(self._labeled_widget("FMax", self.fmax_edit), 1, 3)
        controls_layout.addWidget(self.remove_bg_checkbox, 1, 4)
        controls_layout.addWidget(self._labeled_widget("KMin", self.kmin_edit), 2, 0)
        controls_layout.addWidget(self._labeled_widget("KMax", self.kmax_edit), 2, 1)
        controls_layout.addWidget(self._labeled_widget("VMin(dB)", self.vmin_db_edit), 2, 2)
        controls_layout.addWidget(self._labeled_widget("VMax(dB)", self.vmax_db_edit), 2, 3)
        controls_layout.addWidget(self.plot_button, 2, 4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel("left", "Frequency", units="Hz")
        self.plot_widget.setLabel("bottom", "Spatial frequency", units="cycles/m")
        self.plot_widget.setTitle("f-k 谱", color=AXIS_TEXT_COLOR)
        self.plot_widget.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        for axis_name in ("left", "bottom"):
            style_axis(self.plot_widget.getAxis(axis_name), axis_font, label_font)
        # Keep unit labels as-is ("Hz", "cycles/m"): pyqtgraph would otherwise
        # auto-prefix them to "kHz" / "mcycles/m" based on the data range.
        for axis_name in ("left", "bottom"):
            axis = self.plot_widget.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.labelUnitPrefix = ""

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_widget.addItem(self.image_item)
        self.image_item.setColorMap(COLOR_MAPS["jet"])

        self.histogram_widget = pg.HistogramLUTWidget(orientation="vertical")
        self.histogram_widget.setImageItem(self.image_item)
        self.histogram_widget.setFixedWidth(150)
        self.histogram_widget.setBackground("w")
        self.histogram_widget.item.axis.setLabel(text="Amplitude", units="dB")
        style_axis(self.histogram_widget.item.axis, axis_font, label_font)
        self.histogram_widget.gradient.setColorMap(COLOR_MAPS["jet"])

        plot_row = QWidget()
        plot_row_layout = QHBoxLayout(plot_row)
        plot_row_layout.setContentsMargins(0, 0, 0, 0)
        plot_row_layout.setSpacing(6)
        plot_row_layout.addWidget(self.plot_widget, 1)
        plot_row_layout.addWidget(self.histogram_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(controls)
        layout.addWidget(plot_row, 1)

        self.plot_button.clicked.connect(self.plot_requested)
        self.channel_range_edit.returnPressed.connect(self.plot_requested)
        self.spacing_edit.returnPressed.connect(self.plot_requested)
        self.spatial_taper_edit.returnPressed.connect(self.plot_requested)
        self.temporal_taper_edit.returnPressed.connect(self.plot_requested)
        self.nfft_channels_edit.returnPressed.connect(self.plot_requested)
        self.nfft_samples_edit.returnPressed.connect(self.plot_requested)

    def set_spectrum(
        self,
        spectrum: dict[str, np.ndarray],
        *,
        channel_start: int,
        channel_end: int,
        spacing_m: float,
        levels: tuple[float, float],
        frequency_range: tuple[float | None, float | None] = (None, None),
        wavenumber_range: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """Display f-k spectrum computed from selected channels."""
        db = np.asarray(spectrum["db"], dtype=np.float32)
        frequency_hz = np.asarray(spectrum["frequency_hz"], dtype=np.float32)
        wavenumber_cpm = np.asarray(spectrum["wavenumber_cpm"], dtype=np.float32)
        if db.size == 0 or frequency_hz.size == 0 or wavenumber_cpm.size == 0:
            raise ValueError("f-k 谱数据为空。")

        image_data = np.ascontiguousarray(db.T)
        self.image_item.setImage(image_data, autoLevels=False, levels=levels)
        self.histogram_widget.item.setLevels(*levels)

        k_min = float(wavenumber_cpm[0])
        k_max = float(wavenumber_cpm[-1])
        f_min = float(frequency_hz[0])
        f_max = float(frequency_hz[-1])
        width = k_max - k_min if k_max > k_min else 1.0
        height = f_max - f_min if f_max > f_min else 1.0
        self.image_item.setRect(QRectF(k_min, f_min, width, height))

        self.plot_widget.setLimits(xMin=k_min, xMax=k_max, yMin=f_min, yMax=f_max)
        display_k_min, display_k_max = self._resolve_display_range(wavenumber_range, k_min, k_max)
        display_f_min, display_f_max = self._resolve_display_range(frequency_range, f_min, f_max)
        self.plot_widget.setXRange(display_k_min, display_k_max, padding=0.0)
        self.plot_widget.setYRange(display_f_min, display_f_max, padding=0.0)
        self.plot_widget.setTitle(
            f"f-k 谱：Ch {channel_start}-{channel_end}, dx={spacing_m:g} m",
            color=AXIS_TEXT_COLOR,
        )

    def _resolve_display_range(
        self,
        requested_range: tuple[float | None, float | None],
        fallback_min: float,
        fallback_max: float,
    ) -> tuple[float, float]:
        range_min, range_max = requested_range
        if range_min is None or range_max is None:
            return fallback_min, fallback_max
        return range_min, range_max

    def _labeled_widget(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(label_text)
        label.setFont(widget.font())
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return container


class LogFrequencyAxis(pg.AxisItem):
    """Axis for y values stored as log10(Hz), labelled as real Hz."""

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        labels = []
        for value in values:
            decade = float(value)
            rounded = round(decade)
            if abs(decade - rounded) > 1e-6:
                labels.append("")
                continue
            frequency = 10.0 ** rounded
            labels.append(f"{frequency:.0f}")
        return labels


class TimeFrequencyAnalysisView(QWidget):
    """Tab4 widget for one channel waveform and short-time spectrum."""

    plot_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel_edit = QLineEdit("1")
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["none", "lowpass", "highpass", "bandpass"])
        self.filter_type_combo.setCurrentText("none")
        self.low_cut_edit = QLineEdit()
        self.high_cut_edit = QLineEdit()
        self.tf_mode_combo = QComboBox()
        self.tf_mode_combo.addItems(["PSD", "Amplitude"])
        self.tf_scale_combo = QComboBox()
        self.tf_scale_combo.addItems(["Log", "Linear"])
        self.window_seconds_edit = QLineEdit("0.01")
        self.visible_window_seconds_edit = QLineEdit("0.1")
        self.overlap_percent_edit = QLineEdit("50")
        self.fmin_edit = QLineEdit()
        self.fmax_edit = QLineEdit()
        self.vmin_edit = QLineEdit()
        self.vmax_edit = QLineEdit()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["jet", "hsv", "seismic"])
        self.colormap_combo.setCurrentText("jet")
        self.auto_color_checkbox = QCheckBox("颜色自动")
        self.auto_color_checkbox.setChecked(True)
        self.plot_button = QPushButton("绘制时频图")

        ui_font = QFont("Times New Roman", 10)
        axis_font = QFont("Times New Roman", 18)
        label_font = QFont("Times New Roman", 18)
        compact_edits = [
            self.channel_edit, self.low_cut_edit, self.high_cut_edit,
            self.window_seconds_edit, self.visible_window_seconds_edit,
            self.overlap_percent_edit, self.fmin_edit, self.fmax_edit,
            self.vmin_edit, self.vmax_edit,
        ]
        compact_combos = [
            self.filter_type_combo, self.tf_mode_combo, self.tf_scale_combo,
            self.colormap_combo,
        ]
        for widget in compact_edits + compact_combos:
            widget.setFont(ui_font)
        for widget in compact_edits:
            widget.setMaximumWidth(80)
        for widget in compact_combos:
            widget.setMaximumWidth(100)
        self.auto_color_checkbox.setFont(ui_font)
        self.plot_button.setFont(ui_font)
        for edit in compact_edits:
            edit.setPlaceholderText("auto")

        controls = QGroupBox("Tab4 时域/时频参数")
        controls.setFont(ui_font)
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(6, 4, 6, 4)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(2)
        controls_layout.addWidget(self._labeled_widget("通道", self.channel_edit), 0, 0)
        controls_layout.addWidget(self._labeled_widget("滤波", self.filter_type_combo), 0, 1)
        controls_layout.addWidget(self._labeled_widget("Low Cut", self.low_cut_edit), 0, 2)
        controls_layout.addWidget(self._labeled_widget("High Cut", self.high_cut_edit), 0, 3)
        controls_layout.addWidget(self._labeled_widget("模式", self.tf_mode_combo), 0, 4)
        controls_layout.addWidget(self._labeled_widget("数值", self.tf_scale_combo), 0, 5)
        controls_layout.addWidget(self._labeled_widget("窗长(s)", self.window_seconds_edit), 0, 6)
        controls_layout.addWidget(self._labeled_widget("重叠(%)", self.overlap_percent_edit), 0, 7)
        controls_layout.addWidget(self._labeled_widget("FMin", self.fmin_edit), 1, 0)
        controls_layout.addWidget(self._labeled_widget("FMax", self.fmax_edit), 1, 1)
        controls_layout.addWidget(self._labeled_widget("VMin", self.vmin_edit), 1, 2)
        controls_layout.addWidget(self._labeled_widget("VMax", self.vmax_edit), 1, 3)
        controls_layout.addWidget(self._labeled_widget("显示窗口(s)", self.visible_window_seconds_edit), 1, 4)
        controls_layout.addWidget(self._labeled_widget("Colormap", self.colormap_combo), 1, 5)
        controls_layout.addWidget(self.auto_color_checkbox, 1, 6)
        controls_layout.addWidget(self.plot_button, 1, 7)

        self._metadata: FileMetadata | None = None
        self._channel = 1
        self._trace: np.ndarray | None = None
        self._tf_log_bounds: tuple[float, float] | None = None
        self._active_plot: pg.PlotWidget | None = None
        self._syncing_x_ranges = False

        self.time_axis = TimeAxisItem(orientation="bottom")
        self.time_plot = pg.PlotWidget(axisItems={"bottom": self.time_axis})
        self.time_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.time_plot.setMinimumHeight(180)
        self.time_plot.setLabel("left", "Phase", units="rad")
        self.time_plot.setLabel("bottom", "Time")
        self.time_plot.setTitle("指定通道时域波形", color=AXIS_TEXT_COLOR)
        self._style_plot(self.time_plot, axis_font, label_font)
        # Fix the left-axis width so the plot area does not shift when the
        # time-frequency y tick labels change digit count while zooming.
        self.time_plot.getAxis("left").setWidth(96)
        self.time_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.time_curve = self.time_plot.plot(pen=pg.mkPen("#cc2222", width=1.2))
        self.time_curve.setClipToView(True)
        self.time_curve.setDownsampling(auto=True, method="peak")
        self.time_right_spacer = QWidget()
        # 150 px matches the histogram width (150) including the 6px layout
        # spacing so the two plot areas stay left/right aligned.
        self.time_right_spacer.setFixedWidth(150)

        time_row = QWidget()
        time_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        time_row.setMinimumHeight(200)
        time_row_layout = QHBoxLayout(time_row)
        time_row_layout.setContentsMargins(0, 0, 0, 0)
        time_row_layout.setSpacing(6)
        time_row_layout.addWidget(self.time_plot, 1)
        time_row_layout.addWidget(self.time_right_spacer)

        self.tf_axis = TimeAxisItem(orientation="bottom")
        self.tf_plot = pg.PlotWidget(axisItems={"bottom": self.tf_axis, "left": LogFrequencyAxis(orientation="left")})
        self.tf_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tf_plot.setMinimumHeight(360)
        self.tf_plot.setLabel("left", "Frequency", units="Hz")
        self.tf_plot.setLabel("bottom", "Time")
        self.tf_plot.setTitle("短时时频图", color=AXIS_TEXT_COLOR)
        self._style_plot(self.tf_plot, axis_font, label_font)
        # Same fixed left-axis width as time_plot to keep time axes aligned.
        self.tf_plot.getAxis("left").setWidth(96)
        self.tf_plot.showGrid(x=True, y=False, alpha=0.22)
        self.tf_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.tf_plot.getViewBox().setAspectLocked(False)
        self.tf_plot.setAutoVisible(y=False)
        self.tf_plot.enableAutoRange(axis="y", enable=False)
        self.time_plot.getViewBox().sigXRangeChanged.connect(self._on_time_x_range_changed)
        self.tf_plot.getViewBox().sigXRangeChanged.connect(self._on_tf_x_range_changed)
        self.tf_plot.getViewBox().sigYRangeChanged.connect(self._update_frequency_axis_ticks)
        self.tf_image_item = pg.ImageItem(axisOrder="row-major")
        self.tf_plot.addItem(self.tf_image_item)
        self.tf_image_item.setColorMap(COLOR_MAPS["jet"])

        self.histogram_widget = pg.HistogramLUTWidget(orientation="vertical")
        self.histogram_widget.setImageItem(self.tf_image_item)
        self.histogram_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.histogram_widget.setFixedWidth(150)
        self.histogram_widget.setBackground("w")
        self.histogram_widget.item.axis.setLabel(text="Amplitude", units="")
        style_axis(self.histogram_widget.item.axis, axis_font, label_font)
        self.histogram_widget.gradient.setColorMap(COLOR_MAPS["jet"])

        tf_row = QWidget()
        tf_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tf_row.setMinimumHeight(380)
        tf_row_layout = QHBoxLayout(tf_row)
        tf_row_layout.setContentsMargins(0, 0, 0, 0)
        tf_row_layout.setSpacing(6)
        tf_row_layout.addWidget(self.tf_plot, 1)
        tf_row_layout.addWidget(self.histogram_widget)

        plot_splitter = QSplitter(Qt.Vertical)
        plot_splitter.setChildrenCollapsible(False)
        plot_splitter.addWidget(time_row)
        plot_splitter.addWidget(tf_row)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 4)
        plot_splitter.setHandleWidth(8)
        plot_splitter.setSizes([220, 520])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(controls)
        layout.addWidget(plot_splitter, 1)

        self._active_plot: pg.PlotWidget | None = self.time_plot

        self.plot_button.clicked.connect(self.plot_requested)
        for edit in [
            self.channel_edit,
            self.low_cut_edit,
            self.high_cut_edit,
            self.window_seconds_edit,
            self.visible_window_seconds_edit,
            self.overlap_percent_edit,
            self.fmin_edit,
            self.fmax_edit,
            self.vmin_edit,
            self.vmax_edit,
        ]:
            edit.returnPressed.connect(self.plot_requested)
        self.time_plot.scene().sigMouseClicked.connect(self._mark_time_active)
        self.tf_plot.scene().sigMouseClicked.connect(self._mark_tf_active)

    def set_result(
        self,
        *,
        metadata: FileMetadata,
        channel: int,
        trace: np.ndarray,
        tf_result: dict[str, np.ndarray],
        frequency_range: tuple[float | None, float | None],
        levels: tuple[float | None, float | None],
        auto_levels: bool,
        colormap_name: str,
        value_scale: str,
        visible_window_seconds: float | None = None,
    ) -> None:
        self._metadata = metadata
        self._channel = int(channel)
        self._trace = np.asarray(trace, dtype=np.float32)
        self.time_axis.set_reference_time(metadata.start_time)
        self.tf_axis.set_reference_time(metadata.start_time)
        x_values = np.arange(self._trace.size, dtype=np.float32) / metadata.sample_rate_hz
        centered_trace = self._trace - robust_median(self._trace)
        self.time_curve.setData(x_values, centered_trace)
        self.time_plot.setTitle(f"指定通道时域波形：CH {channel}", color=AXIS_TEXT_COLOR)
        duration_s = self._time_x_limit()
        self.time_plot.setLimits(xMin=0.0, xMax=duration_s)
        self.time_plot.enableAutoRange()

        display_values, log_freq = build_log_frequency_display_grid(
            tf_result["frequency_hz"],
            tf_result["values"],
            mode=str(tf_result.get("mode", "psd")),
            value_scale=value_scale,
        )
        if display_values.size == 0 or log_freq.size == 0:
            raise ValueError("时频图没有可显示的正频率数据。")

        self._tf_log_bounds = (float(log_freq[0]), float(log_freq[-1]))
        self.tf_image_item.setImage(display_values, autoLevels=False)
        color_map = COLOR_MAPS.get(colormap_name.lower(), COLOR_MAPS["jet"])
        self.tf_image_item.setColorMap(color_map)
        self.histogram_widget.gradient.setColorMap(color_map)
        if value_scale.strip().lower() == "log":
            self.histogram_widget.item.axis.setLabel(text="Amplitude", units="dB")
        else:
            self.histogram_widget.item.axis.setLabel(text="Amplitude", units="")

        times = np.asarray(tf_result["time_s"], dtype=np.float32)
        x0 = float(times[0])
        x1 = float(times[-1])
        dx = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        dy = float(np.median(np.diff(log_freq))) if log_freq.size > 1 else 0.01
        self.tf_image_item.setRect(QRectF(x0 - 0.5 * dx, float(log_freq[0]) - 0.5 * dy, max(dx, x1 - x0 + dx), max(dy, float(log_freq[-1] - log_freq[0]) + dy)))
        self.tf_plot.getViewBox().setAspectLocked(False)
        self.tf_plot.enableAutoRange(axis="y", enable=False)
        self.tf_plot.setLimits(xMin=0.0, xMax=duration_s, yMin=float(log_freq[0]) - dy, yMax=float(log_freq[-1]) + 1.2 * dy)
        self._apply_frequency_range(frequency_range)
        self._update_frequency_axis_ticks()
        self._apply_color_levels(display_values, levels, auto_levels)
        display_duration = duration_s
        if visible_window_seconds is not None:
            display_duration = min(duration_s, max(float(visible_window_seconds), 1e-9))
        self._set_linked_x_range(0.0, display_duration)
        mode_label = "PSD" if str(tf_result.get("mode", "psd")) == "psd" else "Amplitude"
        self.tf_plot.setTitle(f"短时时频图：CH {channel}, {mode_label}", color=AXIS_TEXT_COLOR)

    def pan_active_plot(self, direction: int) -> None:
        plot = self._active_plot or self.time_plot
        view_box = plot.getViewBox()
        x_min, x_max = view_box.viewRange()[0]
        span = x_max - x_min
        if span <= 0:
            return
        delta = span * 0.3 * direction
        low, high = self._common_x_limit()
        new_min, new_max = ChannelAnalysisView._clamp_pan(x_min, x_max, delta, low, high)
        view_box.setXRange(new_min, new_max, padding=0.0)

    def _time_x_limit(self) -> float:
        if self._metadata is None or self._trace is None:
            return 1.0
        return max(self._trace.size / self._metadata.sample_rate_hz, 1e-9)

    def _tf_x_limit(self) -> tuple[float, float]:
        rect = self.tf_image_item.boundingRect()
        return float(rect.left()), float(rect.right())

    def _common_x_limit(self) -> tuple[float, float]:
        return 0.0, self._time_x_limit()

    def _on_time_x_range_changed(self, _view_box, x_range) -> None:
        self._sync_x_range(self.tf_plot, x_range)

    def _on_tf_x_range_changed(self, _view_box, x_range) -> None:
        self._sync_x_range(self.time_plot, x_range)

    def _sync_x_range(self, target_plot: pg.PlotWidget, x_range) -> None:
        if self._syncing_x_ranges or self._metadata is None or self._trace is None:
            return

        x_min, x_max = self._clamp_x_range(float(x_range[0]), float(x_range[1]))
        target_min, target_max = target_plot.getViewBox().viewRange()[0]
        if np.isclose(x_min, target_min) and np.isclose(x_max, target_max):
            return

        self._syncing_x_ranges = True
        try:
            target_plot.setXRange(x_min, x_max, padding=0.0)
        finally:
            self._syncing_x_ranges = False

    def _set_linked_x_range(self, x_min: float, x_max: float) -> None:
        x_min, x_max = self._clamp_x_range(x_min, x_max)
        self._syncing_x_ranges = True
        try:
            self.time_plot.setXRange(x_min, x_max, padding=0.0)
            self.tf_plot.setXRange(x_min, x_max, padding=0.0)
        finally:
            self._syncing_x_ranges = False

    def _clamp_x_range(self, x_min: float, x_max: float) -> tuple[float, float]:
        low, high = self._common_x_limit()
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            return low, high

        span = min(x_max - x_min, high - low)
        new_min = x_min
        new_max = x_min + span
        if new_min < low:
            new_min = low
            new_max = low + span
        if new_max > high:
            new_max = high
            new_min = high - span
        return new_min, new_max

    def _mark_time_active(self, _ev) -> None:
        self._active_plot = self.time_plot

    def _mark_tf_active(self, _ev) -> None:
        self._active_plot = self.tf_plot

    def _apply_frequency_range(self, frequency_range: tuple[float | None, float | None]) -> None:
        if self._tf_log_bounds is None:
            self.tf_plot.getAxis("left").setTicks(None)
            return
        y_min, y_max = frequency_range
        lower_bound, upper_bound = self._tf_log_bounds
        if y_min is None or y_max is None:
            self.tf_plot.setYRange(lower_bound, upper_bound, padding=0.0)
            return
        log_min = max(np.log10(max(float(y_min), 1e-9)), lower_bound)
        log_max = min(np.log10(max(float(y_max), 1e-9)), upper_bound)
        if log_min >= log_max:
            raise ValueError("时频图频率范围超出可显示频段。")
        self.tf_plot.setYRange(log_min, log_max, padding=0.0)

    def _update_frequency_axis_ticks(self, *_args) -> None:
        axis = self.tf_plot.getAxis("left")
        if self._tf_log_bounds is None:
            axis.setTicks(None)
            return

        y_min, y_max = sorted(float(value) for value in self.tf_plot.getViewBox().viewRange()[1])
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            return

        lo_decade = int(np.floor(y_min))
        hi_decade = int(np.ceil(y_max))
        major_ticks: list[tuple[float, str]] = []
        minor_ticks: list[tuple[float, str]] = []
        for decade in range(lo_decade, hi_decade + 1):
            tick = float(decade)
            if y_min <= tick <= y_max:
                major_ticks.append((tick, f"{10.0 ** decade:.0f}"))
            for factor in range(2, 10):
                sub_tick = decade + float(np.log10(factor))
                if y_min <= sub_tick <= y_max:
                    minor_ticks.append((sub_tick, ""))

        axis.setTicks([major_ticks, minor_ticks])

    def _apply_color_levels(
        self,
        values: np.ndarray,
        levels: tuple[float | None, float | None],
        auto_levels: bool,
    ) -> None:
        if auto_levels:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return
            level_min = float(np.nanpercentile(finite, 2.0))
            level_max = float(np.nanpercentile(finite, 99.5))
            if level_min >= level_max:
                level_min = level_max - 1.0
        else:
            level_min, level_max = levels
            if level_min is None or level_max is None or level_min >= level_max:
                raise ValueError("时频图颜色范围需要满足 VMin < VMax。")
        self.tf_image_item.setLevels((level_min, level_max))
        self.histogram_widget.item.setLevels(level_min, level_max)

    def _style_plot(self, plot_widget: pg.PlotWidget, axis_font: QFont, label_font: QFont) -> None:
        plot_widget.setBackground("w")
        plot_widget.showGrid(x=True, y=True, alpha=0.15)
        for axis_name in ("left", "bottom"):
            axis = plot_widget.getAxis(axis_name)
            style_axis(axis, axis_font, label_font)
            # Positive tickLength points outward for both left and bottom axes.
            axis.setStyle(tickLength=8)
            axis.enableAutoSIPrefix(False)
            axis.labelUnitPrefix = ""
        plot_widget.getViewBox().setMouseEnabled(x=True, y=True)

    def _labeled_widget(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(label_text)
        label.setFont(widget.font())
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return container
