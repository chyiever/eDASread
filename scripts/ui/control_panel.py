"""Parameter control widgets used by the main window."""

from __future__ import annotations

import re

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFont, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _format_number(value) -> str:
    return f"{value:g}"


def parse_point_range_text(text: str):
    """Parse ``A~B`` into ``(start, end)`` ints, or ``None`` when invalid."""
    raw = text.strip().replace("～", "~")
    if not raw:
        return None
    parts = raw.split("~", maxsplit=1)
    try:
        if len(parts) == 2:
            if not parts[0] and not parts[1]:
                return None
            start = int(parts[0]) if parts[0].strip() else 1
            end = int(parts[1]) if parts[1].strip() else start
        else:
            start = end = int(parts[0])
    except ValueError:
        return None
    if start > end:
        start, end = end, start
    return start, end


def parse_color_range_text(text: str):
    """Parse ``A~B`` into two floats (``None`` means auto)."""
    raw = text.strip().replace("～", "~")
    parts = raw.split("~", maxsplit=1)
    try:
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            return None
        low = float(parts[0].strip())
        high = float(parts[1].strip())
    except ValueError:
        return None
    if low > high:
        low, high = high, low
    return low, high


def parse_filter_text(text: str):
    """Parse a merged filter string into ``(filter_type, low_hz, high_hz)``.

    Forms: ``1~`` high-pass 1 Hz; ``~300`` low-pass 300 Hz; ``5k~20k``
    band-pass 5 kHz - 20 kHz; empty/``none`` = no filter.
    """
    raw = text.strip().replace("～", "~")
    if not raw or raw.lower() in ("none", "auto"):
        return "none", None, None
    parts = re.split(r"~", raw, maxsplit=1)
    if len(parts) < 2:
        return "none", None, None

    def _to_hz(side: str):
        side = side.strip().lower()
        if not side:
            return None
        multiplier = 1.0
        if side.endswith("k"):
            multiplier = 1000.0
            side = side[:-1]
        try:
            value = float(side) * multiplier
        except ValueError:
            return None
        return value

    low = _to_hz(parts[0])
    high = _to_hz(parts[1])
    if low is None and high is None:
        return "none", None, None
    if low is not None and high is None:
        return "highpass", low, None
    if low is None and high is not None:
        return "lowpass", None, high
    if low > high:
        low, high = high, low
    return "bandpass", low, high


def format_filter_text(filter_type: str, low_hz, high_hz) -> str:
    if filter_type in ("none", "") or (low_hz is None and high_hz is None):
        return ""
    if filter_type == "highpass":
        return f"{_format_number(float(low_hz))}~"
    if filter_type == "lowpass":
        return f"~{_format_number(float(high_hz))}"
    return f"{_format_number(float(low_hz))}~{_format_number(float(high_hz))}"


class WheelIntLineEdit(QLineEdit):
    """Integer line edit with wheel-step support bounded by [min, max]."""

    wheelStepped = pyqtSignal(int)

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._minimum = 1
        self._maximum = 1_000_000
        self.setValidator(QIntValidator(self._minimum, self._maximum, self))

    def set_bounds(self, minimum: int, maximum: int) -> None:
        minimum = int(minimum)
        maximum = int(maximum)
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        self._minimum = minimum
        self._maximum = maximum
        self.setValidator(QIntValidator(self._minimum, self._maximum, self))

        current_text = self.text().strip()
        if not current_text:
            self.setText(str(self._minimum))
            return

        try:
            current_value = int(current_text)
        except ValueError:
            self.setText(str(self._minimum))
            return

        self.setText(str(max(self._minimum, min(current_value, self._maximum))))

    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return

        step = 1 if delta > 0 else -1
        current_text = self.text().strip()
        try:
            current_value = int(current_text)
        except ValueError:
            current_value = self._minimum

        next_value = max(self._minimum, min(current_value + step, self._maximum))
        if next_value != current_value:
            self.setText(str(next_value))
            self.wheelStepped.emit(next_value)
        event.accept()


class ControlPanel(QWidget):
    """Collection of user-editable rendering and filtering controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ui_font = QFont("Times New Roman", 10)
        compact_font = QFont("Times New Roman", 9)

        self.point_start_edit = QLineEdit("1")
        self.point_end_edit = QLineEdit("100")
        self.vmin_edit = QLineEdit("-0.4")
        self.vmax_edit = QLineEdit("0.4")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["jet", "hsv", "seismic"])
        self.colormap_combo.setCurrentText("seismic")
        self.trace_spacing_edit = QLineEdit("5")
        self.first_trace_pos_edit = QLineEdit("0")
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems(["通道号", "距离"])
        self.y_axis_combo.setCurrentText("通道号")
        self.distance_unit_combo = QComboBox()
        self.distance_unit_combo.addItems(["m", "km"])
        self.distance_unit_combo.setCurrentText("m")
        self.distance_unit_combo.setEnabled(False)

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["时空颜色图", "多道波形(Wiggle)"])
        self.display_mode_combo.setCurrentText("时空颜色图")

        self.point_range_edit = QLineEdit("1~100")
        self.color_range_edit = QLineEdit("-0.4~0.4")
        self.filter_edit = QLineEdit("10~")
        self._merging_edit_sync = False
        self.point_range_edit.textChanged.connect(self._sync_merged_to_parts)
        self.color_range_edit.textChanged.connect(self._sync_merged_to_parts)
        self.filter_edit.textChanged.connect(self._sync_merged_to_parts)

        self.wiggle_scale_combo = QComboBox()
        self.wiggle_scale_combo.addItems([
            "全局最大值", "三道最大值", "单道最大值",
            "全局均方根", "三道均方根", "单道均方根", "全局平均值",
        ])
        self.wiggle_scale_combo.setCurrentText("全局最大值")
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(-60, 60)
        self.gain_slider.setValue(0)
        self.gain_slider.setFixedWidth(70)
        self.gain_minus_button = QPushButton("−")
        self.gain_minus_button.setFixedSize(24, 24)
        self.gain_minus_button.setToolTip("增益 -5 dB")
        self.gain_plus_button = QPushButton("+")
        self.gain_plus_button.setFixedSize(24, 24)
        self.gain_plus_button.setToolTip("增益 +5 dB")
        self.gain_minus_button.clicked.connect(lambda: self.gain_slider.setValue(self.gain_slider.value() - 5))
        self.gain_plus_button.clicked.connect(lambda: self.gain_slider.setValue(self.gain_slider.value() + 5))
        self.gain_edit = QLineEdit("0")
        self.gain_edit.setFixedWidth(48)
        self.gain_edit.setAlignment(Qt.AlignCenter)
        self.gain_edit.setValidator(QIntValidator(-60, 60, self))
        self.flip_checkbox = QCheckBox("负向翻转")
        self.clip_edit = QLineEdit("0")
        self.fill_checkbox = QCheckBox("填充")
        self.fill_checkbox.setChecked(True)
        self.fill_negative_checkbox = QCheckBox("负向填充")
        self.spacing_combo = QComboBox()
        self.spacing_combo.addItems(["地震道", "像素点"])
        self.spacing_combo.setCurrentText("地震道")
        self.spacing_edit = QLineEdit("1")

        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")
        self.reset_view_button = QPushButton("Reset View")
        self.back_view_button = QPushButton("Back View")
        self.zoom_out_2x_button = QPushButton("Zoom Out 2x")
        self.action_buttons = [
            self.apply_button,
            self.reset_button,
            self.reset_view_button,
            self.back_view_button,
            self.zoom_out_2x_button,
        ]

        self.waveform_point_edit = WheelIntLineEdit("1")
        self.waveform_point_edit.setFixedWidth(90)
        self.play_waveform_button = QPushButton("播放波形")

        self.spike_button = QPushButton("去尖峰噪声")
        self.common_button = QPushButton("去共模噪声")
        self.random_button = QPushButton("去随机噪声")
        self.fk_filter_button = QPushButton("FK滤波")
        self.denoise_buttons = [
            self.spike_button,
            self.common_button,
            self.random_button,
            self.fk_filter_button,
        ]
        self.fk_spacing_edit = QLineEdit("5")
        self.fk_vmin_edit = QLineEdit("800")
        self.fk_vmax_edit = QLineEdit("1200")

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["none", "lowpass", "highpass", "bandpass"])
        self.filter_type_combo.setCurrentText("highpass")
        self.low_cut_edit = QLineEdit("10")
        self.high_cut_edit = QLineEdit()

        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText("Export folder")
        self.save_filtered_checkbox = QCheckBox("Save Filtered Data")
        self.save_as_txt_checkbox = QCheckBox("Save as txt")

        self.browse_export_button = QPushButton("Browse Export")
        self.export_button = QPushButton("Export Current View")

        for widget in [
            self.point_range_edit,
            self.color_range_edit,
            self.filter_edit,
            self.colormap_combo,
            self.export_path_edit,
            self.waveform_point_edit,
            self.fk_spacing_edit,
            self.fk_vmin_edit,
            self.fk_vmax_edit,
            self.trace_spacing_edit,
            self.first_trace_pos_edit,
            self.y_axis_combo,
            self.distance_unit_combo,
            self.display_mode_combo,
        ]:
            widget.setFont(ui_font)
        for checkbox in [self.save_filtered_checkbox, self.save_as_txt_checkbox]:
            checkbox.setFont(ui_font)

        point_validator = QIntValidator(0, 100000, self)
        cutoff_validator = QDoubleValidator(0.0, 100000.0, 3, self)
        cutoff_validator.setNotation(QDoubleValidator.StandardNotation)
        color_validator = QDoubleValidator(-1000.0, 1000.0, 3, self)
        color_validator.setNotation(QDoubleValidator.StandardNotation)
        positive_validator = QDoubleValidator(0.001, 1000000.0, 3, self)
        positive_validator.setNotation(QDoubleValidator.StandardNotation)

        self.point_start_edit.setValidator(point_validator)
        self.point_end_edit.setValidator(point_validator)
        self.low_cut_edit.setValidator(cutoff_validator)
        self.high_cut_edit.setValidator(cutoff_validator)
        self.vmin_edit.setValidator(color_validator)
        self.vmax_edit.setValidator(color_validator)
        self.fk_spacing_edit.setValidator(positive_validator)
        self.fk_vmin_edit.setValidator(positive_validator)
        self.fk_vmax_edit.setValidator(positive_validator)
        self.trace_spacing_edit.setValidator(positive_validator)
        self.first_trace_pos_edit.setValidator(QDoubleValidator(-1e6, 1e6, 3, self))

        for button in [
            *self.action_buttons,
            self.play_waveform_button,
            *self.denoise_buttons,
            self.browse_export_button,
            self.export_button,
        ]:
            button.setFont(ui_font)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        for button in self.action_buttons:
            button.setMinimumWidth(105)
        self.play_waveform_button.setMinimumWidth(90)

        main_group = QGroupBox("Tab1 Parameters")
        main_group.setFont(ui_font)
        main_layout = QGridLayout(main_group)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setHorizontalSpacing(8)
        main_layout.setVerticalSpacing(4)

        param_area = self._build_parameter_grid(
            [
                ("Point Range", self.point_range_edit, 128),
                ("Color Range", self.color_range_edit, 128),
                ("Colormap", self.colormap_combo, 104),
                ("Y轴显示", self.y_axis_combo, 84),
                ("距离单位", self.distance_unit_combo, 60),
                ("图类型", self.display_mode_combo, 148),
            ],
            [
                ("道间距(m)", self.trace_spacing_edit, 70),
                ("首道位置(m)", self.first_trace_pos_edit, 70),
                ("滤波(Hz)", self.filter_edit, 128),
                ("波形点", self.waveform_point_edit, 70),
            ],
        )

        controls_row_widget = QWidget()
        controls_row_layout = QHBoxLayout(controls_row_widget)
        controls_row_layout.setContentsMargins(0, 0, 0, 0)
        controls_row_layout.setSpacing(6)
        for button in [*self.action_buttons, self.play_waveform_button]:
            controls_row_layout.addWidget(button, 1)

        denoise_buttons_row = QWidget()
        denoise_buttons_layout = QGridLayout(denoise_buttons_row)
        denoise_buttons_layout.setContentsMargins(0, 0, 0, 0)
        denoise_buttons_layout.setHorizontalSpacing(6)
        denoise_buttons_layout.setVerticalSpacing(4)
        denoise_buttons_layout.addWidget(self.spike_button, 0, 0)
        denoise_buttons_layout.addWidget(self.common_button, 0, 1)
        denoise_buttons_layout.addWidget(self.random_button, 1, 0)
        denoise_buttons_layout.addWidget(self.fk_filter_button, 1, 1)
        denoise_param_row = self._build_triple_input_row(
            "DX(m)",
            self.fk_spacing_edit,
            "VMin",
            self.fk_vmin_edit,
            "VMax",
            self.fk_vmax_edit,
        )
        for edit in [self.fk_spacing_edit, self.fk_vmin_edit, self.fk_vmax_edit]:
            edit.setFixedWidth(80)

        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(4)
        export_path_row = QHBoxLayout()
        export_path_row.setSpacing(6)
        export_path_row.addWidget(self.export_path_edit, 1)
        export_path_row.addWidget(self.browse_export_button)
        export_path_row.addWidget(self.export_button)
        export_layout.addLayout(export_path_row)
        export_option_row = QHBoxLayout()
        export_option_row.setContentsMargins(0, 0, 0, 0)
        export_option_row.setSpacing(12)
        export_option_row.addWidget(self.save_filtered_checkbox)
        export_option_row.addWidget(self.save_as_txt_checkbox)
        export_option_row.addStretch(1)
        export_layout.addLayout(export_option_row)

        for button in [*self.action_buttons, *self.denoise_buttons, self.play_waveform_button]:
            button.setMinimumHeight(24)
            button.setMaximumHeight(28)
        for edit in [
            self.point_range_edit,
            self.color_range_edit,
            self.filter_edit,
            self.fk_spacing_edit,
            self.fk_vmin_edit,
            self.fk_vmax_edit,
            self.waveform_point_edit,
            self.trace_spacing_edit,
            self.first_trace_pos_edit,
        ]:
            edit.setMaximumHeight(24)
            edit.setFont(compact_font)

        main_layout.addWidget(param_area, 0, 0, 2, 2)
        main_layout.addWidget(denoise_buttons_row, 0, 2, 2, 1)
        main_layout.addWidget(controls_row_widget, 2, 0, 1, 2)
        main_layout.addWidget(denoise_param_row, 2, 2)
        main_layout.addWidget(export_widget, 3, 0, 1, 3)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        main_layout.setColumnStretch(2, 1)

        for wiggle_widget in [
            self.wiggle_scale_combo,
            self.gain_slider,
            self.gain_edit,
            self.flip_checkbox,
            self.clip_edit,
            self.fill_checkbox,
            self.fill_negative_checkbox,
            self.spacing_combo,
            self.spacing_edit,
        ]:
            wiggle_widget.setFont(ui_font)
        for wiggle_checkbox in [
            self.flip_checkbox,
            self.fill_checkbox,
            self.fill_negative_checkbox,
        ]:
            wiggle_checkbox.setFixedWidth(110)
        for wiggle_edit in [self.clip_edit, self.spacing_edit]:
            wiggle_edit.setMaximumHeight(24)
            wiggle_edit.setFont(compact_font)
        self.gain_edit.setMaximumHeight(24)
        self.gain_edit.setFont(compact_font)
        self.gain_slider.setMaximumHeight(24)

        self.gain_slider.valueChanged.connect(
            lambda value: self.gain_edit.setText(str(value))
        )
        self.gain_edit.editingFinished.connect(
            lambda: self.gain_slider.setValue(
                max(-60, min(60, int(self.gain_edit.text() or 0)))
            )
        )

        self.wiggle_group = QGroupBox("波形变面积参数")
        self.wiggle_group.setFont(ui_font)
        wiggle_grid = QGridLayout(self.wiggle_group)
        wiggle_grid.setContentsMargins(6, 4, 6, 4)
        wiggle_grid.setHorizontalSpacing(8)
        wiggle_grid.setVerticalSpacing(4)

        gain_cell = QWidget()
        gain_cell_layout = QVBoxLayout(gain_cell)
        gain_cell_layout.setContentsMargins(0, 0, 0, 0)
        gain_cell_layout.setSpacing(2)
        gain_label = QLabel("增益(dB)")
        gain_label.setFont(self.gain_slider.font())
        gain_label.setAlignment(Qt.AlignCenter)
        gain_cell_body = QHBoxLayout()
        gain_cell_body.setSpacing(3)
        gain_cell_body.addWidget(self.gain_minus_button)
        gain_cell_body.addWidget(self.gain_slider, 1)
        gain_cell_body.addWidget(self.gain_plus_button)
        gain_cell_body.addWidget(self.gain_edit)
        gain_cell_layout.addWidget(gain_label)
        gain_cell_layout.addLayout(gain_cell_body)

        wiggle_grid.addWidget(self._build_parameter_cell("振幅缩放", self.wiggle_scale_combo), 0, 0)
        wiggle_grid.addWidget(gain_cell, 0, 1)
        wiggle_grid.addWidget(self.flip_checkbox, 0, 2)
        wiggle_grid.addWidget(self._build_parameter_cell("振幅范围", self.clip_edit), 0, 3)
        wiggle_grid.addWidget(self.fill_checkbox, 1, 0)
        wiggle_grid.addWidget(self.fill_negative_checkbox, 1, 1)
        wiggle_grid.addWidget(self._build_parameter_cell("邻道间隔", self.spacing_combo), 1, 2)
        wiggle_grid.addWidget(self._build_parameter_cell("间隔值", self.spacing_edit), 1, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(main_group)
        layout.addWidget(self.wiggle_group)
        # Hidden by default until the user selects the Wiggle plot mode.
        self.wiggle_group.setVisible(False)

        self.browse_export_button.clicked.connect(self._choose_export_directory)

    def set_waveform_point_bounds(self, minimum: int, maximum: int) -> None:
        self.waveform_point_edit.set_bounds(minimum, maximum)

    def set_denoise_buttons_enabled(self, enabled: bool) -> None:
        for button in self.denoise_buttons:
            button.setEnabled(enabled)

    def _build_parameter_grid(self, *rows: list[tuple[str, QWidget, int]]) -> QWidget:
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        columns = max((len(row) for row in rows), default=0)
        for row_index, row_items in enumerate(rows):
            for col_index, (label_text, widget, width) in enumerate(row_items):
                grid.addWidget(
                    self._build_labeled_pair(label_text, widget, width),
                    row_index,
                    col_index,
                )
        if columns > 0:
            grid.setColumnStretch(columns, 1)
            grid.setColumnMinimumWidth(columns, 12)
        return container

    def _build_labeled_pair(self, label_text: str, widget: QWidget, width: int) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        label = QLabel(label_text)
        label.setFont(widget.font())
        widget.setFixedWidth(width)
        widget.setMinimumWidth(width)

        row_layout.addWidget(label)
        row_layout.addWidget(widget)
        return row_widget

    def _build_parameter_cell(self, label_text: str, widget: QWidget) -> QWidget:
        cell = QWidget()
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QLabel(label_text)
        label.setFont(widget.font())
        label.setAlignment(Qt.AlignCenter)

        fixed_width = 96 if isinstance(widget, QComboBox) else 86
        widget.setFixedWidth(fixed_width)
        widget.setMinimumWidth(fixed_width)

        layout.addWidget(label)
        layout.addWidget(widget, 0, Qt.AlignHCenter)
        return cell

    def _build_triple_input_row(
        self,
        first_label_text: str,
        first_widget: QWidget,
        second_label_text: str,
        second_widget: QWidget,
        third_label_text: str,
        third_widget: QWidget,
    ) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        for label_text, widget in [
            (first_label_text, first_widget),
            (second_label_text, second_widget),
            (third_label_text, third_widget),
        ]:
            label = QLabel(label_text)
            label.setFont(widget.font())
            row_layout.addWidget(label)
            row_layout.addWidget(widget, 1)
        return row_widget

    def _sync_merged_to_parts(self, *_args) -> None:
        """Push merged range/filter edits into the internal widgets."""
        if self._merging_edit_sync:
            return
        self._merging_edit_sync = True
        try:
            point_range = parse_point_range_text(self.point_range_edit.text())
            if point_range is not None:
                self.point_start_edit.setText(str(point_range[0]))
                self.point_end_edit.setText(str(point_range[1]))

            color_range = parse_color_range_text(self.color_range_edit.text())
            if color_range is not None:
                if color_range[0] is None:
                    self.vmin_edit.clear()
                    self.vmax_edit.clear()
                else:
                    self.vmin_edit.setText(_format_number(color_range[0]))
                    self.vmax_edit.setText(_format_number(color_range[1]))

            filter_type, low_hz, high_hz = parse_filter_text(self.filter_edit.text())
            self.filter_type_combo.setCurrentText(filter_type)
            self.low_cut_edit.setText(_format_number(low_hz) if low_hz is not None else "")
            self.high_cut_edit.setText(_format_number(high_hz) if high_hz is not None else "")
        finally:
            self._merging_edit_sync = False

    def refresh_merged_edits(self) -> None:
        """Rebuild merged edit texts from the internal widget values."""
        if self._merging_edit_sync:
            return
        self._merging_edit_sync = True
        try:
            start_text = self.point_start_edit.text().strip()
            end_text = self.point_end_edit.text().strip()
            if start_text and end_text:
                self.point_range_edit.setText(f"{start_text}~{end_text}")

            vmin_text = self.vmin_edit.text().strip()
            vmax_text = self.vmax_edit.text().strip()
            if vmin_text and vmax_text:
                self.color_range_edit.setText(f"{vmin_text}~{vmax_text}")
            elif not vmin_text and not vmax_text:
                self.color_range_edit.clear()

            filter_type = self.filter_type_combo.currentText().strip().lower()
            low_text = self.low_cut_edit.text().strip()
            high_text = self.high_cut_edit.text().strip()
            low_hz = float(low_text) if low_text else None
            high_hz = float(high_text) if high_text else None
            self.filter_edit.setText(format_filter_text(filter_type, low_hz, high_hz))
        finally:
            self._merging_edit_sync = False

    def _choose_export_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            self.export_path_edit.text().strip(),
        )
        if directory:
            self.export_path_edit.setText(directory)