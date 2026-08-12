"""Bitshuffle + Zstd packet codec for eDAS bz files."""

from __future__ import annotations

import ctypes
import shutil
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

FILE_MAGIC = b"BZF1"
PACKET_MAGIC = b"BZS1"
LEGACY_PACKET_MAGIC = b"BZF1"
SUPPORTED_PACKET_MAGICS = {PACKET_MAGIC, LEGACY_PACKET_MAGIC}
VERSION = 1
DTYPE_INT32 = 1
ENDIAN_LITTLE = 1
BYTES_PER_SAMPLE = np.dtype(np.int32).itemsize
DEFAULT_BLOCK_VALUES = 65_536
DEFAULT_ZSTD_LEVEL = 3
BZ_SUFFIXES = {".bz", ".bzs"}

FILE_HEADER_STRUCT = struct.Struct("<4sHHQIIIIIIIiI")
PACKET_HEADER_STRUCT = struct.Struct("<4sHHQQIIIHHIiQQQQII")


@dataclass(frozen=True)
class BzFileHeader:
    """Self-describing file header written before packet records."""

    magic: bytes
    version: int
    header_size: int
    created_timestamp_ns: int
    scan_rate_hz: int
    points_per_frame: int
    channel_num: int
    data_source: int
    storage_downsample_factor: int
    packet_frames: int
    file_duration_s: int
    zstd_level: int
    bitshuffle_block_values: int


@dataclass(frozen=True)
class PacketHeader:
    """Fixed binary header for one compressed packet."""

    magic: bytes
    version: int
    header_size: int
    packet_index: int
    timestamp_ns: int
    scan_rate_hz: int
    points_per_frame: int
    frames: int
    dtype_code: int
    endian_code: int
    bitshuffle_block_values: int
    zstd_level: int
    item_count: int
    raw_bytes: int
    shuffled_bytes: int
    compressed_bytes: int
    raw_crc32: int
    payload_crc32: int


@dataclass(frozen=True)
class RawPacket:
    """Uncompressed int32 packet before bitshuffle + zstd."""

    packet_index: int
    timestamp_ns: int
    scan_rate_hz: int
    points_per_frame: int
    frames: int
    samples: np.ndarray


@dataclass(frozen=True)
class CompressedPacket:
    """Compressed packet bytes plus metrics."""

    packet_index: int
    header: bytes
    payload: bytes
    metrics: dict


@dataclass(frozen=True)
class BzFileInfo:
    """Summary inferred from bz packet headers."""

    path: Path
    sample_rate_hz: float
    points: int
    frame_count: int
    raw_size_bytes: int
    compressed_size_bytes: int
    packet_count: int
    first_timestamp_ns: int
    file_header: BzFileHeader | None = None


class ZstdUnavailableError(RuntimeError):
    """Raised when neither python-zstandard nor zstd.dll can be loaded."""


def is_bz_path(file_path: str | Path) -> bool:
    """Return whether a path has a supported bz-family suffix."""
    return Path(file_path).suffix.lower() in BZ_SUFFIXES


def iter_block_counts(total_count: int, block_values: int) -> Iterable[int]:
    """Yield bitshuffle block sizes for a total item count."""
    if block_values <= 0:
        raise ValueError("bitshuffle block_values must be positive.")
    for start in range(0, total_count, block_values):
        yield min(block_values, total_count - start)


class NumpyBitshuffleBackend:
    """NumPy implementation matching the compressor project's proxy bitshuffle."""

    def shuffle_int32(self, array: np.ndarray, block_values: int) -> bytes:
        """Bit-plane shuffle an int32 array and return bytes."""
        flat = np.asarray(array, dtype=np.int32).reshape(-1)
        out = bytearray()
        for start in range(0, flat.size, block_values):
            block = np.ascontiguousarray(flat[start : start + block_values])
            as_bytes = block.view(np.uint8).reshape(-1, BYTES_PER_SAMPLE)
            bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
            out.extend(np.packbits(bits.T.reshape(-1), bitorder="little").tobytes())
        return bytes(out)

    def unshuffle_int32(
        self,
        payload: bytes,
        item_count: int,
        block_values: int,
    ) -> np.ndarray:
        """Reverse :meth:`shuffle_int32` and return a one-dimensional int32 array."""
        expected_bytes = int(item_count) * BYTES_PER_SAMPLE
        if len(payload) != expected_bytes:
            raise ValueError(f"bad bitshuffle payload size: {len(payload)} != {expected_bytes}")

        source = np.frombuffer(payload, dtype=np.uint8)
        out = np.empty(int(item_count), dtype=np.int32)
        byte_pos = 0
        item_pos = 0
        for count in iter_block_counts(int(item_count), int(block_values)):
            byte_count = count * BYTES_PER_SAMPLE
            block_bytes = source[byte_pos : byte_pos + byte_count]
            bits = np.unpackbits(block_bytes, bitorder="little")[: count * 32]
            restored_bits = bits.reshape(32, count).T
            restored_bytes = np.packbits(restored_bits, axis=1, bitorder="little")
            out[item_pos : item_pos + count] = restored_bytes.reshape(-1).view(np.int32)
            byte_pos += byte_count
            item_pos += count
        return out


class PythonZstdBackend:
    """Zstd backend using the optional python-zstandard package."""

    def __init__(self, level: int = DEFAULT_ZSTD_LEVEL) -> None:
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - depends on user environment
            raise ZstdUnavailableError("python-zstandard is not installed.") from exc

        self._zstd = zstd
        self._cctx = zstd.ZstdCompressor(
            level=int(level),
            write_content_size=True,
            write_checksum=True,
        )
        self._dctx = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        return self._cctx.compress(data)

    def decompress(self, data: bytes, raw_size: int) -> bytes:
        return self._dctx.decompress(data, max_output_size=int(raw_size))


class DllZstdBackend:
    """Small ctypes wrapper around zstd.dll's one-shot APIs."""

    def __init__(self, level: int = DEFAULT_ZSTD_LEVEL) -> None:
        dll_path = _find_zstd_dll()
        if dll_path is None:
            raise ZstdUnavailableError("zstd.dll was not found.")

        self.level = int(level)
        self.dll_path = str(dll_path)
        self.lib = ctypes.CDLL(self.dll_path)
        self.lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_compressBound.restype = ctypes.c_size_t
        self.lib.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.ZSTD_compress.restype = ctypes.c_size_t
        self.lib.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ZSTD_decompress.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint
        self.lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_getErrorName.restype = ctypes.c_char_p

    def _check(self, code: int) -> int:
        if self.lib.ZSTD_isError(code):
            name = self.lib.ZSTD_getErrorName(code).decode("utf-8", errors="replace")
            raise RuntimeError(f"Zstd DLL call failed: {name}")
        return int(code)

    def compress(self, data: bytes) -> bytes:
        if not data:
            return b""
        bound = int(self.lib.ZSTD_compressBound(len(data)))
        source = ctypes.create_string_buffer(data, len(data))
        target = ctypes.create_string_buffer(bound)
        out_size = self._check(
            self.lib.ZSTD_compress(target, bound, source, len(data), self.level)
        )
        return target.raw[:out_size]

    def decompress(self, data: bytes, raw_size: int) -> bytes:
        raw_size = int(raw_size)
        if raw_size == 0:
            return b""
        source = ctypes.create_string_buffer(data, len(data))
        target = ctypes.create_string_buffer(raw_size)
        out_size = self._check(self.lib.ZSTD_decompress(target, raw_size, source, len(data)))
        if out_size != raw_size:
            raise RuntimeError(f"Zstd output size mismatch: {out_size} != {raw_size}")
        return target.raw


def create_zstd_backend(level: int = DEFAULT_ZSTD_LEVEL):
    """Create the best available Zstd backend for this runtime."""
    try:
        return PythonZstdBackend(level=level)
    except ZstdUnavailableError:
        try:
            return DllZstdBackend(level=level)
        except ZstdUnavailableError as exc:
            raise ZstdUnavailableError(
                "Cannot read or write bz files because no Zstd backend is available. "
                "Install it with `pip install zstandard`, or place zstd.dll next to the "
                "application / in the Conda Library\\bin directory."
            ) from exc


def _find_zstd_dll() -> Path | None:
    """Find zstd.dll in common Conda, PyInstaller, executable, or PATH locations."""
    candidates: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.extend([Path(meipass) / "zstd.dll", Path(meipass) / "Library" / "bin" / "zstd.dll"])

    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            executable_dir / "zstd.dll",
            executable_dir / "Library" / "bin" / "zstd.dll",
            Path(sys.prefix) / "zstd.dll",
            Path(sys.prefix) / "Library" / "bin" / "zstd.dll",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    for name in ("zstd.dll", "zstd.exe", "zstd"):
        found = shutil.which(name)
        if found:
            path = Path(found).resolve()
            if path.name.lower() == "zstd.dll":
                return path
            sibling = path.parent / "zstd.dll"
            if sibling.exists():
                return sibling.resolve()
    return None


def pack_bz_file_header(
    *,
    scan_rate_hz: int,
    points_per_frame: int,
    channel_num: int = 1,
    data_source: int = 0,
    storage_downsample_factor: int = 1,
    packet_frames: int = 0,
    file_duration_s: int = 0,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    bitshuffle_block_values: int = DEFAULT_BLOCK_VALUES,
    created_timestamp_ns: int | None = None,
) -> bytes:
    """Pack the BZF1 file header used by the acquisition software."""
    created = int(time.time_ns() if created_timestamp_ns is None else created_timestamp_ns)
    return FILE_HEADER_STRUCT.pack(
        FILE_MAGIC,
        VERSION,
        FILE_HEADER_STRUCT.size,
        created,
        int(scan_rate_hz),
        int(points_per_frame),
        int(channel_num),
        int(data_source),
        int(storage_downsample_factor),
        int(packet_frames),
        int(file_duration_s),
        int(zstd_level),
        int(bitshuffle_block_values),
    )


def parse_bz_file_header(data: bytes) -> BzFileHeader:
    """Parse and validate the fixed-size BZF1 file header."""
    if len(data) < FILE_HEADER_STRUCT.size:
        raise ValueError("incomplete bz file header")

    header = BzFileHeader(*FILE_HEADER_STRUCT.unpack(data[: FILE_HEADER_STRUCT.size]))
    if header.magic != FILE_MAGIC:
        raise ValueError(f"bad bz file magic: {header.magic!r}")
    if header.version != VERSION:
        raise ValueError(f"unsupported bz file version: {header.version}")
    if header.header_size < FILE_HEADER_STRUCT.size:
        raise ValueError(f"bad bz file header size: {header.header_size}")
    return header


def parse_packet_header(data: bytes) -> PacketHeader:
    """Parse and validate one fixed-size packet header."""
    if len(data) != PACKET_HEADER_STRUCT.size:
        raise ValueError(f"bad packet header size: {len(data)} != {PACKET_HEADER_STRUCT.size}")

    header = PacketHeader(*PACKET_HEADER_STRUCT.unpack(data))

    if header.magic not in SUPPORTED_PACKET_MAGICS:
        raise ValueError(f"bad bz packet magic: {header.magic!r}")
    if header.version != VERSION:
        raise ValueError(f"unsupported bz packet version: {header.version}")
    if header.header_size != PACKET_HEADER_STRUCT.size:
        raise ValueError(f"bad bz header size: {header.header_size}")
    if header.dtype_code != DTYPE_INT32:
        raise ValueError(f"unsupported bz dtype code: {header.dtype_code}")
    if header.endian_code != ENDIAN_LITTLE:
        raise ValueError(f"unsupported bz endian code: {header.endian_code}")

    expected_items = int(header.frames) * int(header.points_per_frame)
    if header.item_count != expected_items:
        raise ValueError(f"bad bz item_count: {header.item_count} != {expected_items}")
    if header.raw_bytes != header.item_count * BYTES_PER_SAMPLE:
        raise ValueError(f"bad bz raw_bytes: {header.raw_bytes}")
    if header.shuffled_bytes != header.raw_bytes:
        raise ValueError(f"bad bz shuffled_bytes: {header.shuffled_bytes}")
    if header.compressed_bytes <= 0:
        raise ValueError(f"bad bz compressed_bytes: {header.compressed_bytes}")
    return header


def pack_packet_header(
    *,
    packet: RawPacket,
    item_count: int,
    raw_bytes: int,
    shuffled_bytes: int,
    compressed_bytes: int,
    block_values: int,
    zstd_level: int,
    raw_crc32: int,
    payload_crc32: int,
) -> bytes:
    """Pack packet metadata using the bz packet structure."""
    return PACKET_HEADER_STRUCT.pack(
        PACKET_MAGIC,
        VERSION,
        PACKET_HEADER_STRUCT.size,
        int(packet.packet_index),
        int(packet.timestamp_ns),
        int(packet.scan_rate_hz),
        int(packet.points_per_frame),
        int(packet.frames),
        DTYPE_INT32,
        ENDIAN_LITTLE,
        int(block_values),
        int(zstd_level),
        int(item_count),
        int(raw_bytes),
        int(shuffled_bytes),
        int(compressed_bytes),
        int(raw_crc32),
        int(payload_crc32),
    )


class BitshuffleZstdCompressor:
    """Compress int32 packets into bz packet bytes."""

    def __init__(
        self,
        bitshuffle_backend: NumpyBitshuffleBackend | None = None,
        *,
        zstd_level: int = DEFAULT_ZSTD_LEVEL,
        block_values: int = DEFAULT_BLOCK_VALUES,
        zstd_backend=None,
    ) -> None:
        self.bitshuffle = bitshuffle_backend or NumpyBitshuffleBackend()
        self.zstd_level = int(zstd_level)
        self.block_values = int(block_values)
        self.zstd = zstd_backend or create_zstd_backend(level=self.zstd_level)

    def compress_packet(self, packet: RawPacket) -> CompressedPacket:
        """Compress one packet and return its header + payload."""
        samples = np.asarray(packet.samples, dtype=np.int32)
        if not samples.flags["C_CONTIGUOUS"]:
            samples = np.ascontiguousarray(samples)

        expected_shape = (int(packet.frames), int(packet.points_per_frame))
        if samples.shape != expected_shape:
            raise ValueError(f"bad packet shape: {samples.shape} != {expected_shape}")

        item_count = int(samples.size)
        raw_bytes = item_count * BYTES_PER_SAMPLE

        t0 = time.perf_counter()
        shuffled = self.bitshuffle.shuffle_int32(samples, self.block_values)
        t1 = time.perf_counter()
        payload = self.zstd.compress(shuffled)
        t2 = time.perf_counter()

        raw_crc32 = zlib.crc32(samples.view(np.uint8)) & 0xFFFFFFFF
        payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF

        header = pack_packet_header(
            packet=packet,
            item_count=item_count,
            raw_bytes=raw_bytes,
            shuffled_bytes=len(shuffled),
            compressed_bytes=len(payload),
            block_values=self.block_values,
            zstd_level=self.zstd_level,
            raw_crc32=raw_crc32,
            payload_crc32=payload_crc32,
        )

        metrics = {
            "packet_index": int(packet.packet_index),
            "raw_bytes": raw_bytes,
            "compressed_bytes": len(payload),
            "compression_ratio": raw_bytes / len(payload) if payload else float("inf"),
            "bitshuffle_ms": (t1 - t0) * 1000.0,
            "zstd_ms": (t2 - t1) * 1000.0,
            "compress_ms": (t2 - t0) * 1000.0,
        }
        return CompressedPacket(int(packet.packet_index), header, payload, metrics)


class BitshuffleZstdDecompressor:
    """Decompress bz packet bytes into int32 arrays."""

    def __init__(
        self,
        bitshuffle_backend: NumpyBitshuffleBackend | None = None,
        *,
        zstd_backend=None,
    ) -> None:
        self.bitshuffle = bitshuffle_backend or NumpyBitshuffleBackend()
        self.zstd = zstd_backend or create_zstd_backend()

    def decompress_packet(self, header: PacketHeader, payload: bytes) -> np.ndarray:
        """Validate and decompress one packet."""
        payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        if payload_crc32 != header.payload_crc32:
            raise ValueError(
                f"payload crc mismatch at packet {header.packet_index}: "
                f"{payload_crc32:#x} != {header.payload_crc32:#x}"
            )

        shuffled = self.zstd.decompress(payload, header.shuffled_bytes)
        if len(shuffled) != header.shuffled_bytes:
            raise ValueError(
                f"bad zstd output size at packet {header.packet_index}: "
                f"{len(shuffled)} != {header.shuffled_bytes}"
            )

        values = self.bitshuffle.unshuffle_int32(
            shuffled,
            header.item_count,
            header.bitshuffle_block_values,
        )
        if values.dtype != np.int32:
            values = values.astype(np.int32, copy=False)
        if values.size != header.item_count:
            raise ValueError(
                f"bad item count at packet {header.packet_index}: "
                f"{values.size} != {header.item_count}"
            )

        array = values.reshape(header.frames, header.points_per_frame)
        raw_crc32 = zlib.crc32(array.view(np.uint8)) & 0xFFFFFFFF
        if raw_crc32 != header.raw_crc32:
            raise ValueError(
                f"raw crc mismatch at packet {header.packet_index}: "
                f"{raw_crc32:#x} != {header.raw_crc32:#x}"
            )
        return array


def iter_compressed_packets(path: str | Path) -> Iterable[tuple[PacketHeader, bytes]]:
    """Iterate packet headers and payloads from one bz file."""
    file_path = Path(path)
    with file_path.open("rb") as file:
        _position_at_first_packet(file)
        while True:
            header_offset = file.tell()
            header_bytes = file.read(PACKET_HEADER_STRUCT.size)
            if not header_bytes:
                break
            if len(header_bytes) != PACKET_HEADER_STRUCT.size:
                raise EOFError(f"truncated bz packet header at offset {header_offset}")

            header = parse_packet_header(header_bytes)
            payload = file.read(header.compressed_bytes)
            if len(payload) != header.compressed_bytes:
                raise EOFError(
                    f"truncated bz payload at packet {header.packet_index}: "
                    f"{len(payload)} != {header.compressed_bytes}"
                )
            yield header, payload


def _position_at_first_packet(file) -> BzFileHeader | None:
    """Read an optional BZF1 file header and seek to the first packet."""
    prefix = file.read(8)
    if not prefix:
        return None
    if len(prefix) != 8:
        raise EOFError("truncated bz prefix")

    magic, _version, header_size = struct.unpack("<4sHH", prefix)
    file.seek(0)
    if magic == FILE_MAGIC and header_size != PACKET_HEADER_STRUCT.size:
        header_bytes = file.read(max(header_size, FILE_HEADER_STRUCT.size))
        file_header = parse_bz_file_header(header_bytes)
        extra = int(file_header.header_size) - FILE_HEADER_STRUCT.size
        if extra > 0 and len(header_bytes) < int(file_header.header_size):
            file.read(extra)
        return file_header

    if magic in SUPPORTED_PACKET_MAGICS:
        return None

    raise ValueError(f"unrecognized bz magic: {magic!r}")


def scan_bz_file(path: str | Path) -> BzFileInfo:
    """Scan packet headers and return whole-file metadata without decompression."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    file_size = file_path.stat().st_size
    points: int | None = None
    sample_rate_hz: float | None = None
    frame_count = 0
    raw_size_bytes = 0
    packet_count = 0
    first_timestamp_ns = 0
    expected_packet_index: int | None = None
    file_header: BzFileHeader | None = None

    with file_path.open("rb") as file:
        file_header = _position_at_first_packet(file)
        while True:
            header_offset = file.tell()
            header_bytes = file.read(PACKET_HEADER_STRUCT.size)
            if not header_bytes:
                break
            if len(header_bytes) != PACKET_HEADER_STRUCT.size:
                raise EOFError(f"truncated bz packet header at offset {header_offset}")

            header = parse_packet_header(header_bytes)
            payload_end = file.tell() + header.compressed_bytes
            if payload_end > file_size:
                raise EOFError(
                    f"truncated bz payload at packet {header.packet_index}: "
                    f"ends at {payload_end}, file size is {file_size}"
                )

            if packet_count == 0:
                points = int(header.points_per_frame)
                sample_rate_hz = float(header.scan_rate_hz)
                first_timestamp_ns = int(header.timestamp_ns)
                expected_packet_index = int(header.packet_index)
            else:
                if header.points_per_frame != points:
                    raise ValueError("bz packets use different point counts.")
                if float(header.scan_rate_hz) != sample_rate_hz:
                    raise ValueError("bz packets use different scan rates.")

            if expected_packet_index is not None and header.packet_index != expected_packet_index:
                raise ValueError(
                    f"bz packet index gap: got {header.packet_index}, expected {expected_packet_index}"
                )

            frame_count += int(header.frames)
            raw_size_bytes += int(header.raw_bytes)
            packet_count += 1
            expected_packet_index = int(header.packet_index) + 1
            file.seek(payload_end)

    if packet_count == 0 or points is None or sample_rate_hz is None:
        raise ValueError(f"No bz packets found in {file_path}")

    return BzFileInfo(
        path=file_path,
        sample_rate_hz=sample_rate_hz,
        points=points,
        frame_count=frame_count,
        raw_size_bytes=raw_size_bytes,
        compressed_size_bytes=file_size,
        packet_count=packet_count,
        first_timestamp_ns=first_timestamp_ns,
        file_header=file_header,
    )


def read_bz_file(path: str | Path) -> np.ndarray:
    """Read a bz file into a contiguous frames x points int32 array."""
    info = scan_bz_file(path)
    data = np.empty((info.frame_count, info.points), dtype=np.int32)
    decompressor = BitshuffleZstdDecompressor()

    frame_pos = 0
    expected_packet_index: int | None = None
    for header, payload in iter_compressed_packets(info.path):
        if expected_packet_index is None:
            expected_packet_index = int(header.packet_index)
        elif header.packet_index != expected_packet_index:
            raise ValueError(
                f"bz packet index gap: got {header.packet_index}, expected {expected_packet_index}"
            )

        packet_data = decompressor.decompress_packet(header, payload)
        frame_end = frame_pos + int(header.frames)
        data[frame_pos:frame_end, :] = packet_data
        frame_pos = frame_end
        expected_packet_index = int(header.packet_index) + 1

    if frame_pos != info.frame_count:
        raise ValueError(f"bz frame count mismatch: {frame_pos} != {info.frame_count}")
    return data
