"""Readers for eDAS phase bin files."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from scripts.core.models import FileMetadata
from scripts.io.bz_codec import (
    DEFAULT_BLOCK_VALUES,
    DEFAULT_ZSTD_LEVEL,
    BZ_SUFFIXES,
    BitshuffleZstdCompressor,
    RawPacket,
    is_bz_path,
    pack_bz_file_header,
    read_bz_file,
    scan_bz_file,
)

POINTS_PATTERN = re.compile(r"-(\d+)pt-")
SCAN_RATE_PATTERN = re.compile(r"-(\d+(?:\.\d+)?)Hz-")
TIMESTAMP_PATTERN = re.compile(r"-(\d{8}T\d{6}(?:\.\d+)?)\.(?:bin|bz|bzs)$", re.IGNORECASE)
LEADING_TOKEN_PATTERN = re.compile(r"^([^-]+)-")
PHASE_RAD_SCALE = np.float32(np.pi / 32767.0)
RAW_DTYPE = np.int32
RAW_BYTES_PER_POINT = np.dtype(RAW_DTYPE).itemsize
DATA_SUFFIXES = {".bin", ".npz", *BZ_SUFFIXES}


def list_bin_files(folder_path: str | Path) -> list[Path]:
    """List all supported data files in one directory in lexicographic order."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in DATA_SUFFIXES
    )


def list_raw_bin_files(folder_path: str | Path) -> list[Path]:
    """List only uncompressed bin files in one directory."""
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

    if path.suffix.lower() == ".npz":
        from scripts.io.npz_codec import inspect_edas_npz
        return inspect_edas_npz(path)
    if is_bz_path(path):
        return _inspect_bz_file(path)

    return _inspect_raw_bin_file(path)


def _inspect_raw_bin_file(path: Path) -> FileMetadata:
    """Parse metadata and validate one uncompressed bin file."""
    points = infer_points_from_filename(path)
    sample_rate_hz = infer_sample_rate_from_filename(path)
    if points is None or sample_rate_hz is None:
        raise ValueError(
            f"Unable to infer points/sample rate from file name: {path.name}"
        )

    file_size_bytes = path.stat().st_size
    if file_size_bytes <= 0:
        raise ValueError(f"File {path.name} is empty and cannot be loaded.")

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
    if frame_count <= 0:
        raise ValueError(f"File {path.name} contains no complete data frames.")

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


def _inspect_bz_file(path: Path) -> FileMetadata:
    """Parse metadata and validate one compressed bz file."""
    info = scan_bz_file(path)

    filename_points = infer_points_from_filename(path)
    filename_sample_rate = infer_sample_rate_from_filename(path)
    if filename_points is not None and filename_points != info.points:
        raise ValueError(
            f"File name point count {filename_points} does not match bz header {info.points}"
        )
    if filename_sample_rate is not None and filename_sample_rate != info.sample_rate_hz:
        raise ValueError(
            f"File name sample rate {filename_sample_rate:g} does not match bz header {info.sample_rate_hz:g}"
        )

    return FileMetadata(
        file_path=path,
        sample_rate_hz=info.sample_rate_hz,
        points=info.points,
        frame_count=info.frame_count,
        duration_s=info.frame_count / info.sample_rate_hz,
        file_size_bytes=info.compressed_size_bytes,
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


def infer_leading_token_from_filename(file_path: str | Path) -> str | None:
    """Infer the leading token before the first dash in a file name."""
    match = LEADING_TOKEN_PATTERN.match(Path(file_path).name)
    return match.group(1) if match else None


def build_export_filename(
    metadata: FileMetadata,
    point_count: int,
    time_offset_s: float = 0.0,
    suffix: str = ".bin",
    preserve_leading_token: bool = False,
) -> str:
    """Build an export file name using the requested naming convention."""
    start_time = metadata.start_time or infer_start_time_from_filename(metadata.file_path)
    if start_time is not None:
        start_text = format_timestamp_for_filename(start_time + timedelta(seconds=max(time_offset_s, 0.0)))
    else:
        start_text = "unknown"

    sample_rate_text = f"{metadata.sample_rate_hz:.12f}".rstrip("0").rstrip(".") + "Hz"
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    prefix = ""
    if preserve_leading_token:
        leading_token = infer_leading_token_from_filename(metadata.file_path)
        if leading_token:
            prefix = f"{leading_token}-"
    return f"{prefix}eDAS-{sample_rate_text}-{point_count:04d}pt-{start_text}{normalized_suffix}"


def read_bin_memmap(metadata: FileMetadata) -> np.ndarray:
    """Read a phase data file as a 2D int32 array shaped as frames x points."""
    if metadata.file_path.suffix.lower() == ".npz":
        from scripts.io.npz_codec import read_edas_npz
        return read_edas_npz(metadata.file_path)[0]
    if is_bz_path(metadata.file_path):
        return read_bz_file(metadata.file_path)

    if metadata.file_size_bytes <= 0 or metadata.frame_count <= 0:
        raise ValueError(f"File {metadata.file_path.name} contains no readable data frames.")

    return np.memmap(
        metadata.file_path,
        dtype=RAW_DTYPE,
        mode="r",
        shape=(metadata.frame_count, metadata.points),
    )


def read_phase_radians(metadata: FileMetadata) -> np.ndarray:
    """Read phase data already converted to float32 radians.

    For bz files the int32 full-size array is never materialized: each packet is
    converted to radians before being written into the destination, halving peak
    memory and saving one full pass over the data. Raw bin files stay lazy via
    memmap and are converted to radians in a single vectorized pass.
    """
    if is_bz_path(metadata.file_path):
        return read_bz_file(
            metadata.file_path,
            converter=lambda packet: np.asarray(packet, dtype=np.float32) * PHASE_RAD_SCALE,
        )

    raw = read_bin_memmap(metadata)
    return convert_raw_to_radians(raw)


def convert_radians_to_raw(phase_data: np.ndarray) -> np.ndarray:
    """Convert phase radians back to int32 raw values for bin export."""
    raw = np.rint(np.asarray(phase_data, dtype=np.float32) / PHASE_RAD_SCALE)
    raw = np.clip(raw, np.iinfo(RAW_DTYPE).min, np.iinfo(RAW_DTYPE).max)
    return np.asarray(raw, dtype=RAW_DTYPE)


def convert_raw_to_radians(raw_data: np.ndarray) -> np.ndarray:
    """Convert raw int32 values to phase radians as float32."""
    return np.asarray(raw_data, dtype=np.float32) * PHASE_RAD_SCALE


def compress_bin_to_bz_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    packet_seconds: float = 1.0,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    block_values: int = DEFAULT_BLOCK_VALUES,
    overwrite: bool = False,
    progress_callback=None,
    progress_base_bytes: int = 0,
    progress_total_bytes: int | None = None,
    file_index: int = 1,
    file_count: int = 1,
) -> dict:
    """Compress one raw bin file into the bz packet format."""
    started_at = time.perf_counter()
    source = Path(input_path)
    if source.suffix.lower() != ".bin":
        raise ValueError(f"Only .bin files can be compressed to bz: {source}")

    metadata = _inspect_raw_bin_file(source)
    progress_total = progress_total_bytes or metadata.file_size_bytes
    target = Path(output_path) if output_path is not None else source.with_suffix(".bz")
    if target.exists() and not overwrite:
        _emit_compression_progress(
            progress_callback,
            current_bytes=progress_base_bytes + metadata.file_size_bytes,
            total_bytes=progress_total,
            file_index=file_index,
            file_count=file_count,
            file_name=source.name,
            status="skipped",
        )
        return {
            "input_path": source,
            "output_path": target,
            "skipped": True,
            "reason": "output_exists",
            "packets": 0,
            "raw_bytes": metadata.file_size_bytes,
            "compressed_bytes": target.stat().st_size,
            "elapsed_s": time.perf_counter() - started_at,
        }

    target.parent.mkdir(parents=True, exist_ok=True)

    scan_rate_hz = int(round(metadata.sample_rate_hz))
    if not np.isclose(metadata.sample_rate_hz, scan_rate_hz):
        raise ValueError(f"Sample rate must be an integer for bz header: {metadata.sample_rate_hz}")

    frames_per_packet = max(1, int(round(scan_rate_hz * float(packet_seconds))))
    timestamp_base_ns = _metadata_start_time_ns(metadata)
    raw = read_bin_memmap(metadata)
    compressor = BitshuffleZstdCompressor(
        zstd_level=zstd_level,
        block_values=block_values,
    )

    total_packets = 0
    file_header = pack_bz_file_header(
        scan_rate_hz=scan_rate_hz,
        points_per_frame=metadata.points,
        channel_num=1,
        data_source=0,
        storage_downsample_factor=1,
        packet_frames=frames_per_packet,
        file_duration_s=max(1, int(round(metadata.duration_s))),
        zstd_level=zstd_level,
        bitshuffle_block_values=block_values,
        created_timestamp_ns=timestamp_base_ns or None,
    )
    total_compressed_bytes = len(file_header)
    processed_file_bytes = 0
    _emit_compression_progress(
        progress_callback,
        current_bytes=progress_base_bytes,
        total_bytes=progress_total,
        file_index=file_index,
        file_count=file_count,
        file_name=source.name,
        status="compressing",
    )
    with target.open("wb", buffering=1024 * 1024) as file:
        file.write(file_header)
        for frame_start in range(0, metadata.frame_count, frames_per_packet):
            frame_end = min(metadata.frame_count, frame_start + frames_per_packet)
            samples = np.ascontiguousarray(raw[frame_start:frame_end, :], dtype=RAW_DTYPE)
            packet = RawPacket(
                packet_index=total_packets,
                timestamp_ns=timestamp_base_ns + int(round(frame_start * 1_000_000_000 / scan_rate_hz)),
                scan_rate_hz=scan_rate_hz,
                points_per_frame=metadata.points,
                frames=samples.shape[0],
                samples=samples,
            )
            compressed = compressor.compress_packet(packet)
            file.write(compressed.header)
            file.write(compressed.payload)
            total_packets += 1
            total_compressed_bytes += len(compressed.header) + len(compressed.payload)
            processed_file_bytes += int(samples.nbytes)
            _emit_compression_progress(
                progress_callback,
                current_bytes=progress_base_bytes + processed_file_bytes,
                total_bytes=progress_total,
                file_index=file_index,
                file_count=file_count,
                file_name=source.name,
                status="compressing",
            )

    return {
        "input_path": source,
        "output_path": target,
        "skipped": False,
        "packets": total_packets,
        "raw_bytes": metadata.file_size_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": metadata.file_size_bytes / total_compressed_bytes
        if total_compressed_bytes
        else float("inf"),
        "elapsed_s": time.perf_counter() - started_at,
    }


def batch_compress_bin_folder(
    input_folder: str | Path,
    output_folder: str | Path | None = None,
    *,
    packet_seconds: float = 1.0,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    block_values: int = DEFAULT_BLOCK_VALUES,
    overwrite: bool = False,
    progress_callback=None,
) -> dict:
    """Compress every .bin file in one folder into .bz files."""
    started_at = time.perf_counter()
    source_folder = Path(input_folder)
    target_folder = Path(output_folder) if output_folder is not None else source_folder
    bin_files = list_raw_bin_files(source_folder)
    if not bin_files:
        raise ValueError(f"No .bin files found in {source_folder}")

    target_folder.mkdir(parents=True, exist_ok=True)
    results = []
    progress_total_bytes = sum(path.stat().st_size for path in bin_files)
    progress_base_bytes = 0
    for index, source in enumerate(bin_files, start=1):
        target = target_folder / source.with_suffix(".bz").name
        result = compress_bin_to_bz_file(
            source,
            target,
            packet_seconds=packet_seconds,
            zstd_level=zstd_level,
            block_values=block_values,
            overwrite=overwrite,
            progress_callback=progress_callback,
            progress_base_bytes=progress_base_bytes,
            progress_total_bytes=progress_total_bytes,
            file_index=index,
            file_count=len(bin_files),
        )
        results.append(result)
        progress_base_bytes += int(result["raw_bytes"])

    converted = [item for item in results if not item.get("skipped")]
    skipped = [item for item in results if item.get("skipped")]
    raw_bytes = sum(int(item["raw_bytes"]) for item in results)
    compressed_bytes = sum(int(item["compressed_bytes"]) for item in results)
    converted_raw_bytes = sum(int(item["raw_bytes"]) for item in converted)
    converted_compressed_bytes = sum(int(item["compressed_bytes"]) for item in converted)
    elapsed_s = time.perf_counter() - started_at
    throughput_mib_s = (converted_raw_bytes / 1024 / 1024 / elapsed_s) if elapsed_s > 0 else 0.0
    _emit_compression_progress(
        progress_callback,
        current_bytes=progress_total_bytes,
        total_bytes=progress_total_bytes,
        file_index=len(bin_files),
        file_count=len(bin_files),
        file_name="",
        status="finished",
    )
    return {
        "input_folder": source_folder,
        "output_folder": target_folder,
        "total": len(results),
        "converted": len(converted),
        "skipped": len(skipped),
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "converted_raw_bytes": converted_raw_bytes,
        "converted_compressed_bytes": converted_compressed_bytes,
        "compression_ratio": raw_bytes / compressed_bytes if compressed_bytes else float("inf"),
        "converted_compression_ratio": converted_raw_bytes / converted_compressed_bytes
        if converted_compressed_bytes
        else 0.0,
        "compression_percent": (converted_compressed_bytes / converted_raw_bytes * 100.0)
        if converted_raw_bytes
        else 0.0,
        "elapsed_s": elapsed_s,
        "throughput_mib_s": throughput_mib_s,
        "results": results,
    }


def _metadata_start_time_ns(metadata: FileMetadata) -> int:
    if metadata.start_time is None:
        return 0
    return int(metadata.start_time.timestamp() * 1_000_000_000)


def _emit_compression_progress(
    progress_callback,
    *,
    current_bytes: int,
    total_bytes: int,
    file_index: int,
    file_count: int,
    file_name: str,
    status: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "current_bytes": max(0, int(current_bytes)),
            "total_bytes": max(1, int(total_bytes)),
            "file_index": int(file_index),
            "file_count": int(file_count),
            "file_name": file_name,
            "status": status,
        }
    )
