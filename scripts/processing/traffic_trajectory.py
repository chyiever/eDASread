"""Signal processing for the traffic-DAS trajectory annotation tab."""

from __future__ import annotations

import numpy as np
from scipy import signal


def build_vehicle_energy_image(
    phase_rad: np.ndarray,
    sample_rate_hz: float,
    *,
    low_hz: float = 0.05,
    high_hz: float = 10.0,
    remove_common_mode: bool = True,
    decimation: int = 10,
    balance: str = "p999",
) -> tuple[np.ndarray, float]:
    """Convert a phase waterfall to the non-negative image used for annotation.

    This follows section 3.2 of ``das_vehicle_trajectory_detection.ipynb``:
    band-pass/common-mode preprocessing, Hilbert envelope, block averaging,
    per-channel median subtraction, clipping and optional channel balancing.
    """
    data = np.asarray(phase_rad, dtype=np.float32)
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("相位数据必须是非空的二维数组（帧 × 通道）。")
    fs = float(sample_rate_hz)
    decimation = int(decimation)
    if decimation < 1:
        raise ValueError("时间降采样倍数必须大于等于 1。")
    if not 0 < low_hz < high_hz < fs / 2.0:
        raise ValueError(f"需要 0 < Low Cut < High Cut < Nyquist ({fs / 2:g} Hz)。")
    if data.shape[0] < 16:
        raise ValueError("数据帧数过少，无法进行零相位滤波。")

    # SOS form is stable for the very low high-pass corner used by traffic DAS.
    hp = signal.butter(2, low_hz, btype="highpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(hp, data, axis=0).astype(np.float32)
    lp = signal.butter(4, high_hz, btype="lowpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(lp, filtered, axis=0).astype(np.float32)
    if remove_common_mode:
        filtered -= np.mean(filtered, axis=1, keepdims=True, dtype=np.float32)

    envelope = np.abs(signal.hilbert(filtered, axis=0)).astype(np.float32)
    usable = envelope.shape[0] // decimation * decimation
    if usable == 0:
        raise ValueError("数据长度小于时间降采样倍数。")
    energy = envelope[:usable].reshape(-1, decimation, envelope.shape[1]).mean(axis=1)
    energy -= np.median(energy, axis=0, keepdims=True)
    np.maximum(energy, 0.0, out=energy)

    method = balance.strip().lower()
    if method != "none":
        global_ref = max(float(np.percentile(energy, 99.9)), 1e-12)
        if method == "p999":
            scale = np.percentile(energy, 99.9, axis=0, keepdims=True)
        elif method == "mad":
            scale = np.median(energy, axis=0, keepdims=True)
            positive = scale[scale > 0]
            floor = 0.1 * float(np.median(positive)) if positive.size else global_ref
            scale = np.maximum(scale, floor)
        else:
            raise ValueError(f"未知通道均衡方式：{balance}")
        scale[scale <= 0] = global_ref
        energy = energy / scale * global_ref
    return np.asarray(energy, dtype=np.float32), fs / decimation
