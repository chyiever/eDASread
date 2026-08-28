"""Time-frequency analysis helpers for selected eDAS channels."""

from __future__ import annotations

import numpy as np
from scipy import signal


def compute_time_frequency_map(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    window_seconds: float,
    overlap_percent: float,
    mode: str = "psd",
) -> dict[str, np.ndarray]:
    """Compute a short-time spectrogram for one channel trace."""
    trace = np.asarray(values, dtype=np.float32).reshape(-1)
    if trace.size < 8:
        raise ValueError("时频图至少需要 8 个采样点。")
    if sample_rate_hz <= 0:
        raise ValueError("采样率必须大于 0。")
    if window_seconds <= 0:
        raise ValueError("时频窗长必须大于 0。")
    if overlap_percent < 0 or overlap_percent >= 100:
        raise ValueError("时频重叠比例必须位于 [0, 100)%。")

    nperseg = max(8, int(round(float(window_seconds) * float(sample_rate_hz))))
    nperseg = min(nperseg, trace.size)
    noverlap = int(round(nperseg * float(overlap_percent) / 100.0))
    noverlap = max(0, min(noverlap, nperseg - 1))

    normalized_mode = mode.strip().lower()
    if normalized_mode == "psd":
        freqs, times, values_map = signal.spectrogram(
            trace,
            fs=float(sample_rate_hz),
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            return_onesided=True,
            mode="psd",
        )
    elif normalized_mode == "amplitude":
        freqs, times, values_map = signal.spectrogram(
            trace,
            fs=float(sample_rate_hz),
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="spectrum",
            return_onesided=True,
            mode="magnitude",
        )
    else:
        raise ValueError("时频模式必须是 PSD 或 Amplitude。")

    return {
        "frequency_hz": np.asarray(freqs, dtype=np.float32),
        "time_s": np.asarray(times, dtype=np.float32),
        "values": np.asarray(values_map, dtype=np.float32),
        "mode": normalized_mode,
    }


def build_log_frequency_display_grid(
    frequency_hz: np.ndarray,
    values: np.ndarray,
    *,
    mode: str,
    value_scale: str = "log",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample spectrogram rows onto an evenly spaced log10-frequency grid."""
    freqs_all = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    base_values_all = np.asarray(values, dtype=np.float64)
    valid = freqs_all > 0.0
    if not np.any(valid):
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.float32)

    freqs = freqs_all[valid]
    base_values = base_values_all[valid, :]
    if freqs.size == 0 or base_values.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.float32)

    if freqs.size == 1:
        log_freq = np.array([float(np.log10(freqs[0]))], dtype=np.float64)
        resampled = base_values.copy()
    else:
        log_freq = np.linspace(float(np.log10(freqs[0])), float(np.log10(freqs[-1])), freqs.size)
        target_freqs = np.power(10.0, log_freq)
        lower_indices = np.searchsorted(freqs, target_freqs, side="right") - 1
        lower_indices = np.clip(lower_indices, 0, freqs.size - 2)
        x0 = freqs[lower_indices]
        x1 = freqs[lower_indices + 1]
        weights = (target_freqs - x0) / np.maximum(x1 - x0, np.finfo(np.float64).eps)
        resampled = (
            base_values[lower_indices, :] * (1.0 - weights[:, None])
            + base_values[lower_indices + 1, :] * weights[:, None]
        )

    if value_scale.strip().lower() == "log":
        floor = np.finfo(np.float64).tiny
        if mode.strip().lower() == "psd":
            display_values = 10.0 * np.log10(np.maximum(resampled, floor))
        else:
            display_values = 20.0 * np.log10(np.maximum(resampled, floor))
    else:
        display_values = resampled

    return np.asarray(display_values, dtype=np.float32), np.asarray(log_freq, dtype=np.float32)
