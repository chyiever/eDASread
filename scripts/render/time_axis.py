"""Time axis formatting for millisecond-level display."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pyqtgraph as pg
from PyQt5.QtGui import QFont, QFontMetrics


def seconds_to_label(value: float) -> str:
    """Format a second offset as HH:MM:SS.mmm."""
    if value < 0:
        value = 0.0
    total_ms = int(round(value * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


class TimeAxisItem(pg.AxisItem):
    """Bottom axis that formats values as wall-clock style elapsed time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reference_time: datetime | None = None

    def set_reference_time(self, reference_time: datetime | None) -> None:
        """Set the acquisition start time for wall-clock tick labels."""
        self._reference_time = reference_time

    def tickSpacing(self, minVal, maxVal, size):  # noqa: N802
        """Return tick spacing that never lets adjacent labels overlap.

        Time labels have a fixed wide format (HH:MM:SS.mmm) that PyQtGraph's
        default fill-limit does not account for, so zooming in could produce
        many closely packed major ticks whose labels overlap. We scale the
        default spacing so that the smallest major-tick gap is at least as
        wide as one label.
        """
        levels = super().tickSpacing(minVal, maxVal, size)
        if not levels:
            return levels
        dif = abs(maxVal - minVal)
        if dif <= 0 or size <= 0:
            return levels
        font = self.style.get("tickFont")
        metrics = QFontMetrics(font) if font is not None else QFontMetrics(QFont())
        label_width = metrics.horizontalAdvance(seconds_to_label(999999.0)) + 10
        major, offset = levels[0]
        min_major = label_width * dif / size
        if major >= min_major:
            return levels
        factor = min_major / major
        return [(spacing * factor, off) for spacing, off in levels]

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        if self._reference_time is None:
            return [seconds_to_label(value) if math.isfinite(value) else "" for value in values]

        labels = []
        for value in values:
            if not math.isfinite(value):
                labels.append("")
                continue
            labels.append(
                (self._reference_time + timedelta(seconds=max(value, 0.0))).strftime("%H:%M:%S.%f")[:-3]
            )
        return labels
