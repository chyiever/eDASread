"""Versioned eDAS NPZ: int32 frames x channels plus a JSON Unicode header.

Never uses pickle. Times in arrivals are seconds relative to this file's first
sample. channel_start/channel_step retain source channel identities after crop.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.core.models import FileMetadata


def validate_header(header: dict, data: np.ndarray) -> None:
    if not isinstance(header, dict) or header.get("data_type") != "eDAS":
        raise ValueError("仅支持头信息 data_type 为 eDAS 的 NPZ 文件。")
    if type(header.get("format_version")) is not int or header["format_version"] != 1:
        raise ValueError("不支持的 eDAS NPZ 格式版本。")
    if data.ndim != 2 or data.dtype != np.dtype("int32") or min(data.shape) <= 0:
        raise ValueError("eDAS NPZ 数据必须为非空 frames×channels int32 矩阵。")
    if (type(header.get("channel_count")) is not int or type(header.get("frame_count")) is not int
        or header["channel_count"] != data.shape[1] or header["frame_count"] != data.shape[0]):
        raise ValueError("NPZ 头信息中的通道数/帧数与矩阵不一致。")
    rate = header.get("sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("NPZ 采样率必须为正数。")
    for key, minimum in (("channel_start", 0), ("channel_step", 1)):
        value = header.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"NPZ {key} 无效。")
    for key in ("start_time", "acquisition_start_time"):
        value = header.get(key)
        if value is not None:
            datetime.fromisoformat(value)
    if header.get("raw_unfiltered") is not True:
        raise ValueError("eDAS NPZ 必须保存未滤波原始数据。")
    if header.get("dtype") != "int32" or header.get("arrival_time_reference") != "seconds_from_first_sample":
        raise ValueError("NPZ 数据类型或到时参考定义无效。")
    expected_channels = [header["channel_start"] + i*header["channel_step"] for i in range(data.shape[1])]
    channel_numbers = header.get("channel_numbers")
    if (not isinstance(channel_numbers, list) or any(type(ch) is not int for ch in channel_numbers)
        or channel_numbers != expected_channels):
        raise ValueError("NPZ 通道号列表与起点/间隔不一致。")
    arrivals = header.get("arrivals", [])
    if not isinstance(arrivals, list):
        raise ValueError("NPZ 到时标记必须为列表。")
    seen = set()
    for pick in arrivals:
        ch, time_s = pick["channel"], pick["time_s"]
        if type(ch) is not int or ch in seen:
            raise ValueError("NPZ 到时通道必须是唯一整数。")
        index, remainder = divmod(ch - header["channel_start"], header["channel_step"])
        if remainder or not 0 <= index < data.shape[1]:
            raise ValueError("NPZ 到时通道不在导出数据内。")
        if isinstance(time_s, bool) or not isinstance(time_s, (float, int)) or not np.isfinite(time_s) or not 0 <= time_s <= (data.shape[0]-1)/rate:
            raise ValueError("NPZ 到时超出有效采样时间。")
        seen.add(ch)


def read_edas_npz(path: str | Path) -> tuple[np.ndarray, dict]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "header" not in archive or "data" not in archive:
                raise ValueError("NPZ 缺少 data 或 header。")
            raw_header = archive["header"]
            if raw_header.ndim != 0 or raw_header.dtype.kind not in ("U", "S"):
                raise ValueError("NPZ header 必须为 JSON 字符串标量。")
            value = raw_header.item()
            header = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
            if not isinstance(header, dict) or header.get("data_type") != "eDAS":
                raise ValueError("仅支持头信息 data_type 为 eDAS 的 NPZ 文件。")
            data = archive["data"]
            validate_header(header, data)
            return data, header
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("eDAS NPZ 头信息不完整或无效。") from exc


def inspect_edas_npz(path: str | Path) -> FileMetadata:
    path = Path(path)
    data, header = read_edas_npz(path)
    return FileMetadata(
        file_path=path, sample_rate_hz=float(header["sample_rate_hz"]),
        points=data.shape[1], frame_count=data.shape[0],
        duration_s=data.shape[0]/header["sample_rate_hz"], file_size_bytes=path.stat().st_size,
        start_time=datetime.fromisoformat(header["start_time"]) if header.get("start_time") else None,
        channel_start=header["channel_start"], channel_step=header["channel_step"],
        arrivals=tuple((p["channel"],float(p["time_s"])) for p in header.get("arrivals",[])),
        acquisition_start_time=datetime.fromisoformat(header["acquisition_start_time"]) if header.get("acquisition_start_time") else None,
    )


def write_edas_npz(path: str | Path, data: np.ndarray, header: dict) -> None:
    validate_header(header, data)
    np.savez_compressed(path, data=data, header=np.array(json.dumps(header, ensure_ascii=False)))
