"""Filtering utilities for eDAS phase data."""

from __future__ import annotations

import numpy as np
from scipy import signal

from scripts.core.models import FilterConfig


def validate_filter_config(config: FilterConfig, sample_rate_hz: float) -> None:
    """Validate a filter configuration against Nyquist constraints."""
    if not config.is_enabled():
        return

    nyquist = sample_rate_hz / 2.0
    filter_type = config.filter_type.lower()

    if filter_type == "lowpass":
        if config.high_cut_hz is None or not 0 < config.high_cut_hz < nyquist:
            raise ValueError("Low-pass cutoff must be in (0, Nyquist).")
        return

    if filter_type == "highpass":
        if config.low_cut_hz is None or not 0 < config.low_cut_hz < nyquist:
            raise ValueError("High-pass cutoff must be in (0, Nyquist).")
        return

    if filter_type == "bandpass":
        if config.low_cut_hz is None or config.high_cut_hz is None:
            raise ValueError("Band-pass requires low and high cutoffs.")
        if not 0 < config.low_cut_hz < config.high_cut_hz < nyquist:
            raise ValueError("Band-pass must satisfy 0 < low < high < Nyquist.")
        return

    raise ValueError(f"Unsupported filter type: {config.filter_type}")


def apply_filter(data: np.ndarray, sample_rate_hz: float, config: FilterConfig) -> np.ndarray:
    """Apply the configured zero-phase Butterworth filter to frames x points data."""
    if not config.is_enabled():
        return np.asarray(data, dtype=np.float32)

    validate_filter_config(config, sample_rate_hz)

    filter_type = config.filter_type.lower()
    nyquist = sample_rate_hz / 2.0
    if filter_type == "lowpass":
        wn = config.high_cut_hz / nyquist
        btype = "lowpass"
    elif filter_type == "highpass":
        wn = config.low_cut_hz / nyquist
        btype = "highpass"
    else:
        wn = [config.low_cut_hz / nyquist, config.high_cut_hz / nyquist]
        btype = "bandpass"

    # The filter operates along the time axis for each spatial point.
    sos = signal.butter(config.order, wn, btype=btype, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(data, dtype=np.float32), axis=0)
    return np.asarray(filtered, dtype=np.float32)
