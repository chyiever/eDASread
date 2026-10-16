"""Snapshot-based view export, independent of live Qt widgets and file selection."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import numpy as np

from scripts.core.models import FileMetadata
from scripts.io.bin_reader import build_export_filename, convert_raw_to_radians, convert_radians_to_raw, read_bin_memmap
from scripts.io.npz_codec import write_edas_npz
from scripts.io.text_export import write_phase_txt


@dataclass(frozen=True)
class ViewExportRequest:
    metadata: FileMetadata
    sources: tuple[FileMetadata, ...]
    data: np.ndarray | None
    frame_start: int
    frame_end: int
    column_start: int
    column_end: int  # exclusive
    time_step: int
    space_step: int
    format: str
    processed: bool
    arrivals: tuple[tuple[int, float], ...]


def read_raw_selection(request: ViewExportRequest) -> np.ndarray:
    """Read source int32 samples, keeping stride phase across file boundaries."""
    chunks = []
    offset = 0
    for source in request.sources:
        lo = max(request.frame_start, offset)
        hi = min(request.frame_end, offset + source.frame_count)
        # First global frame in this source on the exported sampling grid.
        lo += (request.frame_start - lo) % request.time_step
        if lo < hi:
            raw = read_bin_memmap(source)
            chunks.append(np.array(raw[lo-offset:hi-offset:request.time_step,
                                       request.column_start:request.column_end:request.space_step],
                                   dtype=np.int32, copy=True, order="C"))
        offset += source.frame_count
        if offset >= request.frame_end:
            break
    if not chunks:
        raise ValueError("导出范围没有原始样本。")
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


def export_view(request: ViewExportRequest, folder: Path) -> dict:
    r = request
    if r.format not in ("bin", "txt", "npz"):
        raise ValueError("不支持的导出格式。")
    if type(r.time_step) is not int or r.time_step < 1 or type(r.space_step) is not int or r.space_step < 1:
        raise ValueError("时间/空间步长必须为正整数。")
    if not (0 <= r.frame_start < r.frame_end <= r.metadata.frame_count and
            0 <= r.column_start < r.column_end <= r.metadata.points):
        raise ValueError("导出范围无效。")
    processed = r.processed and r.format != "npz"
    if processed:
        if r.data is None:
            raise ValueError("没有可导出的处理后数据。")
        phase = np.ascontiguousarray(r.data[r.frame_start:r.frame_end:r.time_step,
                                           r.column_start:r.column_end:r.space_step], dtype=np.float32)
        data = phase if r.format == "txt" else convert_radians_to_raw(phase)
    else:
        data = read_raw_selection(r)
        if r.format == "txt":
            data = convert_raw_to_radians(data)
    metadata = r.metadata
    rate = metadata.sample_rate_hz / r.time_step
    start_time = metadata.start_time + timedelta(seconds=r.frame_start/metadata.sample_rate_hz) if metadata.start_time else None
    origin = metadata.channel_start + r.column_start * metadata.channel_step
    channel_step = metadata.channel_step * r.space_step
    channel_ids = [origin+i*channel_step for i in range(data.shape[1])]
    time_offset = r.frame_start / metadata.sample_rate_hz
    last_time = (data.shape[0]-1)/rate
    arrivals = [{"channel": int(ch), "time_s": float(np.clip(t-time_offset, 0.0, last_time))} for ch,t in r.arrivals
                if ch in channel_ids and time_offset <= t <= time_offset+last_time]
    filename_metadata = replace(metadata, sample_rate_hz=rate)
    filename = build_export_filename(filename_metadata, data.shape[1], time_offset,
                                     suffix="."+r.format, preserve_leading_token=r.format=="txt")
    path = folder / filename
    if path.resolve() in {item.file_path.resolve() for item in r.sources}:
        raise ValueError("导出路径与源文件相同，请选择其他导出目录。")
    if r.format == "npz":
        acquisition_start = metadata.acquisition_start_time or metadata.start_time
        header = {
            "data_type": "eDAS", "format_version": 1, "dtype": "int32", "raw_unfiltered": True,
            "channel_count": data.shape[1], "frame_count": data.shape[0],
            "sample_rate_hz": rate, "source_sample_rate_hz": metadata.sample_rate_hz,
            "start_time": start_time.isoformat() if start_time else None,
            "acquisition_start_time": acquisition_start.isoformat() if acquisition_start else None,
            "channel_start": origin, "channel_step": channel_step,
            "channel_numbers": channel_ids, "arrivals": arrivals,
            "arrival_time_reference": "seconds_from_first_sample",
            "time_downsample": r.time_step, "space_downsample": r.space_step,
            "downsample_method": "stride_without_filtering",
            "source_files": [str(item.file_path) for item in r.sources],
        }
        write_edas_npz(path, data, header)
    elif r.format == "txt":
        write_phase_txt(path, data)
    else:
        data.tofile(path)
    return {"output_path": path, "frame_count": data.shape[0], "point_count": data.shape[1],
            "filtered": processed, "dtype": str(data.dtype), "format": r.format,
            "sample_rate_hz": rate, "channel_start": origin, "arrival_count":len(arrivals)}
