"""Persistent user-parameter storage backed by QSettings.

Parameters typed into Tab1/Tab2/Tab3 controls are saved between sessions so
the UI reopens with the most recently used values and never resets when a
different data file is loaded.
"""

from __future__ import annotations

import re

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit, QWidget


_TRUE_TOKENS = {"1", "true", "yes", "on"}


class ParameterStore:
    """Save/restore widget values by key between application sessions."""

    def __init__(self) -> None:
        # Explicit INI format keeps parameters in a portable file
        # (%APPDATA%\eDASread\eDASread.ini) instead of the Windows registry,
        # so QSettings.setPath can also redirect it in tests.
        self._settings = QSettings(
            QSettings.IniFormat,
            QSettings.UserScope,
            "eDASread",
            "eDASread",
        )

    def _migrate_channel_indices(self) -> None:
        """Convert legacy persisted channel labels once, retaining physical columns."""
        if int(self._settings.value("schema/channel_index_base", 1)) == 0:
            return
        keys = (
            "tab1/point_range", "tab1/point_start", "tab1/point_end",
            "tab1/waveform_point", "tab2/channels", "tab3/channel_range",
            "tab4/channel", "tab5/channel_start", "tab5/channel_end",
        )
        for key in keys:
            if self._settings.contains(key):
                text = str(self._settings.value(key))
                # Only channel fields are converted; counts, spacing and time stay unchanged.
                converted = re.sub(r"\d+", lambda m: str(max(0, int(m.group()) - 1)), text)
                self._settings.setValue(key, converted)
        if not self._settings.contains("tab1/point_range") and all(
            self._settings.contains(key) for key in ("tab1/point_start", "tab1/point_end")
        ):
            start = self._settings.value("tab1/point_start")
            end = self._settings.value("tab1/point_end")
            self._settings.setValue("tab1/point_range", f"{start}~{end}")
        self._settings.setValue("schema/channel_index_base", 0)
        self._settings.sync()

    def interface_font_size(self) -> int:
        try:
            return max(12, min(28, int(self._settings.value("appearance/font_size_px", 18))))
        except (TypeError, ValueError):
            return 18

    def save_interface_font_size(self, size: int) -> None:
        self._settings.setValue("appearance/font_size_px", max(12, min(28, int(size))))
        self._settings.sync()

    def save_widgets(self, widgets: dict[str, QWidget]) -> None:
        self._settings.setValue("schema/channel_index_base", 0)
        for key, widget in widgets.items():
            self._settings.setValue(key, self._widget_value(widget))
        self._settings.sync()

    def restore_widgets(self, widgets: dict[str, QWidget]) -> set[str]:
        """Apply persisted values; return the set of restored keys."""
        self._migrate_channel_indices()
        restored: set[str] = set()
        for key, widget in widgets.items():
            if not self._settings.contains(key):
                continue
            raw_value = self._settings.value(key)
            if raw_value is None:
                continue
            if not self._apply_widget_value(widget, raw_value):
                continue
            restored.add(key)
        return restored

    @staticmethod
    def _widget_value(widget: QWidget):
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return widget.text()

    @staticmethod
    def _apply_widget_value(widget: QWidget, raw_value: object) -> bool:
        text = str(raw_value)
        if isinstance(widget, QCheckBox):
            widget.setChecked(text.strip().lower() in _TRUE_TOKENS)
            return True
        if isinstance(widget, QComboBox):
            # Non-editable combos ignore unknown items; skip saving those.
            if widget.findText(text) < 0:
                return False
            widget.setCurrentText(text)
            return True
        if isinstance(widget, QLineEdit):
            widget.setText(text)
            return True
        return False
