"""Core models used across the application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FileMetadata:
    """Metadata inferred from one eDAS phase bin file."""

    file_path: Path
    sample_rate_hz: float
    points: int
    frame_count: int
    duration_s: float
    file_size_bytes: int
    start_time: Optional[datetime] = None


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
