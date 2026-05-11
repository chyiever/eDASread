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

        self.point_start_edit = QLineEdit("1")
        self.point_end_edit = QLineEdit("100")
        self.vmin_edit = QLineEdit("-0.4")
        self.vmax_edit = QLineEdit("0.4")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["jet", "hsv", "seismic"])
        self.colormap_combo.setCurrentText("seismic")

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
        ]:
            widget.setFont(ui_font)
        for checkbox in [self.save_filtered_checkbox, self.save_as_txt_checkbox]:
            checkbox.setFont(ui_font)

        point_validator = QIntValidator(0, 100000, self)
        cutoff_validator = QDoubleValidator(0.0, 100000.0, 3, self)
        cutoff_validator.setNotation(QDoubleValidator.StandardNotation)
        color_validator = QDoubleValidator(-1000.0, 1000.0, 3, self)
        color_validator.setNotation(QDoubleValidator.StandardNotation)

        self.point_start_edit.setValidator(point_validator)
        self.point_end_edit.setValidator(point_validator)
        self.low_cut_edit.setValidator(cutoff_validator)
        self.high_cut_edit.setValidator(cutoff_validator)
        self.vmin_edit.setValidator(color_validator)
        self.vmax_edit.setValidator(color_validator)

        for button in [*self.action_buttons, self.browse_export_button, self.export_button]:
            button.setFont(ui_font)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        for button in self.action_buttons:
            button.setMinimumWidth(120)

        point_group = QGroupBox("Display Range")
        point_group.setFont(ui_font)
        point_form = QFormLayout(point_group)
        point_form.setContentsMargins(8, 8, 8, 8)
        point_form.setHorizontalSpacing(10)
        point_form.setVerticalSpacing(8)
        point_form.addRow(QLabel("Point Range"), self._build_labeled_pair_row(
            "Point Start",
            self.point_start_edit,
            "Point End",
            self.point_end_edit,
        ))
        point_form.addRow(QLabel("Color Range"), self._build_labeled_pair_row(
            "VMin",
            self.vmin_edit,
            "VMax",
            self.vmax_edit,
        ))
        point_form.addRow(QLabel("Colormap"), self.colormap_combo)

        controls_row_widget = QWidget()
        controls_row_layout = QHBoxLayout(controls_row_widget)
        controls_row_layout.setContentsMargins(0, 0, 0, 0)
        controls_row_layout.setSpacing(8)
        for button in self.action_buttons:
            controls_row_layout.addWidget(button, 1)
        controls_row_layout.addSpacing(8)
        controls_row_layout.addWidget(self.waveform_point_label)
        controls_row_layout.addWidget(self.waveform_point_edit)
        point_form.addRow(QLabel("Actions"), controls_row_widget)

        filter_group = QGroupBox("Filter")
        filter_group.setFont(ui_font)
        filter_form = QFormLayout(filter_group)
        filter_form.setContentsMargins(8, 8, 8, 8)
        filter_form.setHorizontalSpacing(10)
        filter_form.setVerticalSpacing(8)
        filter_form.addRow(QLabel("Type"), self.filter_type_combo)
        filter_form.addRow(QLabel("Cutoff (Hz)"), self._build_labeled_pair_row(
            "Low Cut",
            self.low_cut_edit,
            "High Cut",
            self.high_cut_edit,
        ))

        export_group = QGroupBox("Export")
        export_group.setFont(ui_font)
        export_layout = QVBoxLayout(export_group)
        export_layout.setContentsMargins(8, 8, 8, 8)
        export_layout.setSpacing(8)
        export_path_row = QHBoxLayout()
        export_path_row.setSpacing(8)
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

        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(8)
        right_column.addWidget(filter_group)
        right_column.addWidget(export_group)

        top_layout = QGridLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setHorizontalSpacing(10)
        top_layout.setVerticalSpacing(8)
        top_layout.addWidget(point_group, 0, 0)
        top_layout.addLayout(right_column, 0, 1)
        top_layout.setColumnStretch(0, 2)
        top_layout.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top_layout)

        self.browse_export_button.clicked.connect(self._choose_export_directory)

    def set_waveform_point_bounds(self, minimum: int, maximum: int) -> None:
        self.waveform_point_edit.set_bounds(minimum, maximum)

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

    def _choose_export_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            self.export_path_edit.text().strip(),
        )
        if directory:
            self.export_path_edit.setText(directory)

