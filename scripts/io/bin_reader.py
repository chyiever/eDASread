"""Readers for eDAS phase bin files."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from scripts.core.models import FileMetadata

POINTS_PATTERN = re.compile(r"-(\d+)pt-")
SCAN_RATE_PATTERN = re.compile(r"-(\d+)Hz-")
TIMESTAMP_PATTERN = re.compile(r"-(\d{8}T\d{6}(?:\.\d+)?)\.bin$", re.IGNORECASE)
PHASE_RAD_SCALE = np.float32(np.pi / 32767.0)
RAW_DTYPE = np.int32
RAW_BYTES_PER_POINT = np.dtype(RAW_DTYPE).itemsize


def list_bin_files(folder_path: str | Path) -> list[Path]:
    """List all bin files in one directory in lexicographic order."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    return sorted(path for path in folder.glob("*.bin") if path.is_file())


def infer_points_from_filename(file_path: str | Path) -> int | None:
    """Infer the point count from file name fragments like '-0130pt-'."""
    match = POINTS_PATTERN.search(Path(file_path).name)
    return int(match.group(1)) if match else None


def infer_sample_rate_from_filename(file_path: str | Path) -> float | None:
    """Infer the sample rate from file name fragments like '-2000Hz-'."""
    match = SCAN_RATE_PATTERN.search(Path(file_path).name)
    return float(match.group(1)) if match else None


def inspect_bin_file(file_path: str | Path) -> FileMetadata:
    """Parse metadata and validate file consistency."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    points = infer_points_from_filename(path)
    sample_rate_hz = infer_sample_rate_from_filename(path)
    if points is None or sample_rate_hz is None:
        raise ValueError(
            f"Unable to infer points/sample rate from file name: {path.name}"
        )

    file_size_bytes = path.stat().st_size
    total_values, remainder = divmod(file_size_bytes, RAW_BYTES_PER_POINT)
    if remainder != 0:
        raise ValueError(
            f"Invalid file size {file_size_bytes} bytes for int32 raw phase data"
        )
    frame_count, frame_remainder = divmod(total_values, points)
    if frame_remainder != 0:
        raise ValueError(
            f"File {path.name} size does not match inferred point count {points}"
        )

    duration_s = frame_count / sample_rate_hz
    return FileMetadata(
        file_path=path,
        sample_rate_hz=sample_rate_hz,
        points=points,
        frame_count=frame_count,
        duration_s=duration_s,
        file_size_bytes=file_size_bytes,
        start_time=infer_start_time_from_filename(path),
    )


def infer_start_time_from_filename(file_path: str | Path) -> datetime | None:
    """Infer the acquisition start time from a file name timestamp fragment."""
    match = TIMESTAMP_PATTERN.search(Path(file_path).name)
    if not match:
        return None

    timestamp_text = match.group(1)
    fmt = "%Y%m%dT%H%M%S.%f" if "." in timestamp_text else "%Y%m%dT%H%M%S"
    return datetime.strptime(timestamp_text, fmt)


def format_timestamp_for_filename(start_time: datetime) -> str:
    """Format timestamps as YYYYMMDDTHHMMSS.mmm for export file names."""
    return start_time.strftime("%Y%m%dT%H%M%S.%f")[:-3]


def build_export_filename(
    metadata: FileMetadata,
    point_count: int,
    time_offset_s: float = 0.0,
    suffix: str = ".bin",
) -> str:
    """Build an export file name using the requested naming convention."""
    start_time = metadata.start_time or infer_start_time_from_filename(metadata.file_path)
    if start_time is not None:
        start_text = format_timestamp_for_filename(start_time + timedelta(seconds=max(time_offset_s, 0.0)))
    else:
        start_text = "unknown"

    sample_rate_text = f"{int(round(metadata.sample_rate_hz))}Hz"
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"eDAS-{sample_rate_text}-{point_count:04d}pt-{start_text}{normalized_suffix}"


def read_bin_memmap(metadata: FileMetadata) -> np.memmap:
    """Map a phase bin file into a 2D int32 array shaped as frames x points."""
    return np.memmap(
        metadata.file_path,
        dtype=RAW_DTYPE,
        mode="r",
        shape=(metadata.frame_count, metadata.points),
    )


def convert_radians_to_raw(phase_data: np.ndarray) -> np.ndarray:
    """Convert phase radians back to int32 raw values for bin export."""
    raw = np.rint(np.asarray(phase_data, dtype=np.float32) / PHASE_RAD_SCALE)
    raw = np.clip(raw, np.iinfo(RAW_DTYPE).min, np.iinfo(RAW_DTYPE).max)
    return np.asarray(raw, dtype=RAW_DTYPE)


def convert_raw_to_radians(raw_data: np.ndarray) -> np.ndarray:
    """Convert raw int32 values to phase radians as float32."""
    return np.asarray(raw_data, dtype=np.float32) * PHASE_RAD_SCALE
