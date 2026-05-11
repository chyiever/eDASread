"""Convert eDAS phase bin files to time-series SEG-Y files."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYBinaryFileHeader, SEGYTraceHeader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.io.bin_reader import (  # noqa: E402
    convert_raw_to_radians,
    inspect_bin_file,
    list_bin_files,
    read_bin_memmap,
)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%f"
MAX_SAMPLES_PER_TRACE = 32767


def parse_starttime_from_filename(file_path: str | Path) -> UTCDateTime | None:
    """Parse timestamps like 20260323T223434.261 from the file name."""
    path = Path(file_path)
    for part in path.stem.split("-"):
        try:
            parsed = datetime.strptime(part, TIMESTAMP_FORMAT)
        except ValueError:
            continue
        return UTCDateTime(parsed)
    return None


def _build_textual_header(
    metadata,
    source_name: str,
    trace_spacing_m: float | None,
    segment_index: int,
    segment_count: int,
) -> str:
    spacing_text = (
        f"TRACE SPACING(M): {trace_spacing_m:.3f}" if trace_spacing_m is not None else "TRACE SPACING(M): UNKNOWN"
    )
    lines = [
        "C01 CLIENT: EDASREAD BIN TO SEGY CONVERTER",
        f"C02 SOURCE FILE: {source_name[:58]}",
        "C03 LAYOUT: ONE TRACE PER SPATIAL POINT",
        "C04 TRACE SAMPLES ARE ORDERED IN TIME",
        f"C05 SAMPLE RATE(HZ): {metadata.sample_rate_hz:.6f}",
        f"C06 POINT COUNT: {metadata.points}",
        f"C07 FRAME COUNT: {metadata.frame_count}",
        f"C08 DURATION(S): {metadata.duration_s:.6f}",
        f"C09 {spacing_text[:72]}",
        "C10 DATA UNIT: PHASE (RAD)",
        f"C11 SEGMENT: {segment_index}/{segment_count}",
    ]
    while len(lines) < 40:
        lines.append(f"C{len(lines) + 1:02d} ")
    return "".join(f"{line[:80]:<80}" for line in lines)


def _build_stream(
    frame_data: np.ndarray,
    sample_rate_hz: float,
    starttime: UTCDateTime,
    metadata,
    trace_spacing_m: float | None,
    data_encoding: int,
    segment_index: int,
    segment_count: int,
) -> Stream:
    if frame_data.ndim != 2:
        raise ValueError("frame_data must have shape (frames, points)")

    frame_count, point_count = frame_data.shape
    if frame_count <= 0 or point_count <= 0:
        raise ValueError("frame_data must be non-empty")

    delta = 1.0 / sample_rate_hz
    stream = Stream()

    for point_index in range(point_count):
        samples = np.require(frame_data[:, point_index], dtype=np.float32)
        trace = Trace(data=samples)
        trace.stats.delta = delta
        trace.stats.starttime = starttime
        trace.stats.channel = f"P{point_index + 1:04d}"
        trace.stats.station = f"PT{point_index + 1:04d}"
        trace.stats.segy = {}
        trace.stats.segy.trace_header = SEGYTraceHeader()
        header = trace.stats.segy.trace_header
        header.trace_sequence_number_within_line = point_index + 1
        header.trace_sequence_number_within_segy_file = point_index + 1
        header.original_field_record_number = 1
        header.trace_number_within_the_original_field_record = point_index + 1
        header.energy_source_point_number = 1
        header.ensemble_number = 1
        header.trace_number_within_the_ensemble = point_index + 1
        header.number_of_samples_in_this_trace = frame_count
        header.sample_interval_in_ms_for_this_trace = int(round(delta * 1_000_000))
        if trace_spacing_m is not None:
            position_mm = int(round(point_index * trace_spacing_m * 1000.0))
            header.group_coordinate_x = position_mm
            header.coordinate_units = 1
        stream.append(trace)

    stream.stats = AttribDict()
    stream.stats.textual_file_header = _build_textual_header(
        metadata,
        metadata.file_path.name,
        trace_spacing_m,
        segment_index,
        segment_count,
    )
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 1
    stream.stats.binary_file_header.number_of_data_traces_per_ensemble = point_count
    stream.stats.binary_file_header.sample_interval_in_microseconds = int(round(delta * 1_000_000))
    stream.stats.binary_file_header.number_of_samples_per_data_trace = frame_count
    stream.stats.binary_file_header.data_sample_format_code = data_encoding
    return stream


def convert_bin_to_time_series_segy(
    bin_file: str | Path,
    output_file: str | Path | None = None,
    trace_spacing_m: float | None = None,
    convert_to_radians: bool = True,
    data_encoding: int = 1,
    byteorder: str | None = ">",
    max_samples_per_trace: int = MAX_SAMPLES_PER_TRACE,
) -> list[Path]:
    """Convert one eDAS bin file into a time-series SEG-Y file."""
    metadata = inspect_bin_file(bin_file)
    raw_data = read_bin_memmap(metadata)
    frame_data = convert_raw_to_radians(raw_data) if convert_to_radians else np.asarray(raw_data, dtype=np.float32)
    starttime = parse_starttime_from_filename(metadata.file_path) or UTCDateTime(1970, 1, 1)

    if max_samples_per_trace <= 0:
        raise ValueError("max_samples_per_trace must be a positive integer")

    output_base = Path(output_file) if output_file is not None else metadata.file_path.with_suffix(".sgy")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    segment_count = (metadata.frame_count + max_samples_per_trace - 1) // max_samples_per_trace
    resolved_byteorder = byteorder if byteorder is not None else sys.byteorder
    outputs: list[Path] = []

    for segment_offset, start_frame in enumerate(range(0, metadata.frame_count, max_samples_per_trace), start=1):
        end_frame = min(start_frame + max_samples_per_trace, metadata.frame_count)
        segment_data = frame_data[start_frame:end_frame, :]
        segment_starttime = starttime + (start_frame / metadata.sample_rate_hz)

        if segment_count == 1:
            segment_output = output_base
        else:
            segment_output = output_base.with_name(
                f"{output_base.stem}.part{segment_offset:04d}{output_base.suffix}"
            )

        stream = _build_stream(
            frame_data=segment_data,
            sample_rate_hz=metadata.sample_rate_hz,
            starttime=segment_starttime,
            metadata=metadata,
            trace_spacing_m=trace_spacing_m,
            data_encoding=data_encoding,
            segment_index=segment_offset,
            segment_count=segment_count,
        )
        stream.write(
            str(segment_output),
            format="SEGY",
            data_encoding=data_encoding,
            byteorder=resolved_byteorder,
        )
        outputs.append(segment_output)

    return outputs


def convert_bin_folder_to_time_series_segy(
    input_folder: str | Path,
    output_folder: str | Path | None = None,
    pattern: str = "*.bin",
    trace_spacing_m: float | None = None,
    convert_to_radians: bool = True,
    data_encoding: int = 1,
    byteorder: str | None = ">",
    max_samples_per_trace: int = MAX_SAMPLES_PER_TRACE,
) -> list[Path]:
    """Convert all matching bin files in one folder to time-series SEG-Y files."""
    files = list_bin_files(input_folder)
    if pattern != "*.bin":
        files = [path for path in files if path.match(pattern)]
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in folder: {input_folder}")

    destination_root = Path(output_folder) if output_folder is not None else Path(input_folder)
    destination_root.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for file_path in files:
        output_path = destination_root / f"{file_path.stem}.sgy"
        outputs.extend(
            convert_bin_to_time_series_segy(
                file_path,
                output_file=output_path,
                trace_spacing_m=trace_spacing_m,
                convert_to_radians=convert_to_radians,
                data_encoding=data_encoding,
                byteorder=byteorder,
                max_samples_per_trace=max_samples_per_trace,
            )
        )
    return outputs


__all__ = [
    "convert_bin_to_time_series_segy",
    "convert_bin_folder_to_time_series_segy",
    "parse_starttime_from_filename",
]
