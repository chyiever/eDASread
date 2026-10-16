"""Core models used across the application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FileMetadata:
    """Metadata and source channel identities for one eDAS phase dataset."""

    file_path: Path
    sample_rate_hz: float
    points: int
    frame_count: int
    duration_s: float
    file_size_bytes: int
    start_time: Optional[datetime] = None
    channel_start: int = 0
    channel_step: int = 1
    arrivals: tuple[tuple[int, float], ...] = ()
    acquisition_start_time: Optional[datetime] = None

    @property
    def channel_end(self) -> int:
        return self.channel_start + (self.points - 1) * self.channel_step

    def channel_index(self, channel: int) -> int:
        index, remainder = divmod(int(channel) - self.channel_start, self.channel_step)
        if remainder or not 0 <= index < self.points:
            raise ValueError(f"通道 {channel} 不在本文件中：{self.channel_start}~{self.channel_end}，间隔 {self.channel_step}。")
        return index

    def channel_slice(self, start: int, end: int) -> slice:
        return slice(self.channel_index(start), self.channel_index(end) + 1)


@dataclass(frozen=True)
class FilterConfig:
    """Pre-processing parameters for one render request."""

    filter_type: str = "none"
    low_cut_hz: Optional[float] = None
    high_cut_hz: Optional[float] = None
    order: int = 4

    def cache_key(self) -> tuple:
        """Return a hashable key for processed data caching."""
        return (
            self.filter_type,
            self.low_cut_hz,
            self.high_cut_hz,
            self.order,
        )

    def is_enabled(self) -> bool:
        """Return whether this configuration requires signal processing."""
        return self.filter_type.lower() != "none"
