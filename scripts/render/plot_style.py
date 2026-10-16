"""Shared pyqtgraph styling helpers."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5.QtGui import QFont


AXIS_TEXT_COLOR = "#42566d"

pg.setConfigOption("foreground", AXIS_TEXT_COLOR)


def style_axis(axis: pg.AxisItem, axis_font: QFont, label_font: QFont) -> None:
    """Apply consistent black axis, tick and label styling."""
    axis_font = QFont("Segoe UI", 11)
    label_font = QFont("Segoe UI", 12)
    axis.setPen(pg.mkPen(AXIS_TEXT_COLOR))
    axis.setTextPen(pg.mkPen(AXIS_TEXT_COLOR))
    axis.setTickPen(pg.mkPen(AXIS_TEXT_COLOR))
    axis.setStyle(tickFont=axis_font)
    axis.label.setFont(label_font)
    axis.label.setDefaultTextColor(pg.mkColor(AXIS_TEXT_COLOR))
