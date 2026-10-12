"""Parameter control widgets used by the main window."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
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
    QVBoxLayout,
    QWidget,
)


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

        self.waveform_point_label = QLabel("指定位置点时域绘图")
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
            self.point_start_edit,
            self.point_end_edit,
            self.vmin_edit,
            self.vmax_edit,
            self.colormap_combo,
            self.filter_type_combo,
            self.low_cut_edit,
            self.high_cut_edit,
            self.export_path_edit,
            self.waveform_point_label,
            self.waveform_point_edit,
            self.fk_spacing_edit,
            self.fk_vmin_edit,
            self.fk_vmax_edit,
            self.trace_spacing_edit,
            self.first_trace_pos_edit,
            self.y_axis_combo,
            self.distance_unit_combo,
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

        point_range_row = self._build_labeled_pair_row(
            "Point Start",
            self.point_start_edit,
            "Point End",
            self.point_end_edit,
        )
        color_range_row = self._build_labeled_pair_row(
            "VMin",
            self.vmin_edit,
            "VMax",
            self.vmax_edit,
        )
        display_row = self._build_multi_widget_row(
            ("Point Range", point_range_row, 3),
            ("Color Range", color_range_row, 2),
            ("Colormap", self.colormap_combo, 2),
        )
        distance_axis_row = self._build_multi_widget_row(
            ("道间距(m)", self.trace_spacing_edit, 2),
            ("第一道位置(m)", self.first_trace_pos_edit, 2),
            ("Y轴显示", self.y_axis_combo, 2),
            ("距离单位", self.distance_unit_combo, 2),
        )

        controls_row_widget = QWidget()
        controls_row_layout = QHBoxLayout(controls_row_widget)
        controls_row_layout.setContentsMargins(0, 0, 0, 0)
        controls_row_layout.setSpacing(6)
        for button in self.action_buttons:
            controls_row_layout.addWidget(button, 1)

        waveform_row_widget = QWidget()
        waveform_row_layout = QHBoxLayout(waveform_row_widget)
        waveform_row_layout.setContentsMargins(0, 0, 0, 0)
        waveform_row_layout.setSpacing(6)
        waveform_row_layout.addWidget(self.waveform_point_label)
        waveform_row_layout.addWidget(self.waveform_point_edit)
        waveform_row_layout.addWidget(self.play_waveform_button)
        waveform_row_layout.addStretch(1)

        filter_row = self._build_multi_widget_row(
            ("Type", self.filter_type_combo, 2),
            ("Cutoff (Hz)", self._build_labeled_pair_row(
            "Low Cut",
            self.low_cut_edit,
            "High Cut",
            self.high_cut_edit,
            ), 3),
        )

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
            self.point_start_edit,
            self.point_end_edit,
            self.vmin_edit,
            self.vmax_edit,
            self.low_cut_edit,
            self.high_cut_edit,
            self.fk_spacing_edit,
            self.fk_vmin_edit,
            self.fk_vmax_edit,
            self.waveform_point_edit,
            self.trace_spacing_edit,
            self.first_trace_pos_edit,
        ]:
            edit.setMaximumHeight(24)
            edit.setFont(compact_font)

        main_layout.addWidget(display_row, 0, 0, 1, 2)
        main_layout.addWidget(distance_axis_row, 1, 0, 1, 2)
        main_layout.addWidget(controls_row_widget, 2, 0, 1, 2)
        main_layout.addWidget(filter_row, 3, 0)
        main_layout.addWidget(waveform_row_widget, 3, 1)
        main_layout.addWidget(denoise_buttons_row, 0, 2, 2, 1)
        main_layout.addWidget(denoise_param_row, 3, 2)
        main_layout.addWidget(export_widget, 4, 0, 1, 3)
        main_layout.setColumnStretch(0, 2)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(main_group)

        self.browse_export_button.clicked.connect(self._choose_export_directory)

    def set_waveform_point_bounds(self, minimum: int, maximum: int) -> None:
        self.waveform_point_edit.set_bounds(minimum, maximum)

    def set_denoise_buttons_enabled(self, enabled: bool) -> None:
        for button in self.denoise_buttons:
            button.setEnabled(enabled)

    def _build_labeled_pair_row(
        self,
        left_label_text: str,
        left_widget: QWidget,
        right_label_text: str,
        right_widget: QWidget,
    ) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        left_label = QLabel(left_label_text)
        right_label = QLabel(right_label_text)
        left_label.setFont(left_widget.font())
        right_label.setFont(right_widget.font())

        row_layout.addWidget(left_label)
        row_layout.addWidget(left_widget, 1)
        row_layout.addSpacing(6)
        row_layout.addWidget(right_label)
        row_layout.addWidget(right_widget, 1)
        return row_widget

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

    def _build_multi_widget_row(self, *items: tuple[str, QWidget, int]) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        for label_text, widget, stretch in items:
            label = QLabel(label_text)
            label.setFont(widget.font())
            row_layout.addWidget(label)
            row_layout.addWidget(widget, stretch)
        return row_widget

    def _choose_export_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            self.export_path_edit.text().strip(),
        )
        if directory:
            self.export_path_edit.setText(directory)