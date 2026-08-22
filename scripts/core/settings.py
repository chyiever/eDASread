"""Persistent user-parameter storage backed by QSettings.

Parameters typed into Tab1/Tab2/Tab3 controls are saved between sessions so
the UI reopens with the most recently used values and never resets when a
different data file is loaded.
"""

from __future__ import annotations

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

    def save_widgets(self, widgets: dict[str, QWidget]) -> None:
        for key, widget in widgets.items():
            self._settings.setValue(key, self._widget_value(widget))
        self._settings.sync()

    def restore_widgets(self, widgets: dict[str, QWidget]) -> set[str]:
        """Apply persisted values; return the set of restored keys."""
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
