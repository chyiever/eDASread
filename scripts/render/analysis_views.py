"""Extra analysis tabs for channel traces, PSD and f-k spectra."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from scipy import signal

from scripts.core.models import FileMetadata
from scripts.processing.view_data import robust_abs_percentile, robust_median
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
        axis_font = QFont("Times New Roman", 16)
        label_font = QFont("Times New Roman", 16)
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
            style_axis(plot_widget.getAxis(axis_name), axis_font, label_font)
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
        axis_font = QFont("Times New Roman", 16)
        label_font = QFont("Times New Roman", 16)
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
