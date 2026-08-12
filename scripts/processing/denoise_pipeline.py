"""DAS denoising and f-k helpers used by the desktop UI."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2, rfft2, rfftfreq
from scipy.signal.windows import tukey

from scripts.processing.denoise import (
    common_mode_noise_removal,
    curvelet_denoising,
    spike_removal,
)


DenoiseStep = Literal["spike", "common", "random", "fk"]

DEFAULT_SPIKE_NCH = 10
DEFAULT_SPIKE_NSP = 10
DEFAULT_SPIKE_THRESHOLD = 10.0
DEFAULT_COMMON_METHOD = "median"
DEFAULT_CURVELET_PAD = 0.3
DEFAULT_CURVELET_KNEE = 0.2
DEFAULT_CURVELET_SOFT = True
DEFAULT_CURVELET_ANGLES = 16
DEFAULT_FK_TAPER = (0.0, 0.05)


def apply_denoise_step(
    data: np.ndarray,
    step: DenoiseStep,
    *,
    sample_rate_hz: float,
    channel_spacing_m: float,
    fk_vmin_mps: float = 800.0,
    fk_vmax_mps: float = 1200.0,
) -> np.ndarray:
    """Apply one DAS denoising step to ``frames x channels`` data."""
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("DAS data must be a 2-D matrix shaped frames x channels.")

    if step == "spike":
        return np.asarray(
            spike_removal(
                matrix,
                nch=min(DEFAULT_SPIKE_NCH, max(1, matrix.shape[1])),
                nsp=min(DEFAULT_SPIKE_NSP, max(1, matrix.shape[0])),
                thresh=DEFAULT_SPIKE_THRESHOLD,
            ),
            dtype=np.float32,
        )

    if step == "common":
        return np.asarray(
            common_mode_noise_removal(matrix, method=DEFAULT_COMMON_METHOD),
            dtype=np.float32,
        )

    if step == "random":
        if min(matrix.shape) < 8:
            raise ValueError("随机噪声 curvelet 去噪至少需要 8 个采样点和 8 个通道。")
        return np.asarray(
            curvelet_denoising(
                matrix,
                pad=DEFAULT_CURVELET_PAD,
                knee_fac=DEFAULT_CURVELET_KNEE,
                soft_thresh=DEFAULT_CURVELET_SOFT,
                nbangles=DEFAULT_CURVELET_ANGLES,
            ),
            dtype=np.float32,
        )

    if step == "fk":
        return fk_filter_high_velocity(
            matrix,
            sample_rate_hz=sample_rate_hz,
            channel_spacing_m=channel_spacing_m,
            vmin_mps=fk_vmin_mps,
            vmax_mps=fk_vmax_mps,
        )

    raise ValueError(f"Unknown denoising step: {step}")


def fk_filter_high_velocity(
    data: np.ndarray,
    *,
    sample_rate_hz: float,
    channel_spacing_m: float,
    vmin_mps: float = 800.0,
    vmax_mps: float = 1200.0,
    taper_fraction: tuple[float, float] = DEFAULT_FK_TAPER,
) -> np.ndarray:
    """Retain high apparent-velocity energy with a DASPy-style f-k taper."""
    _validate_fk_inputs(data, sample_rate_hz, channel_spacing_m)
    vmin, vmax = _ordered_velocity_bounds(vmin_mps, vmax_mps)

    gather = np.asarray(data, dtype=np.float32).T.copy()
    nch, nt = gather.shape
    spatial_taper, temporal_taper = taper_fraction
    if spatial_taper > 0 and nch > 1:
        gather *= tukey(nch, spatial_taper).astype(np.float32)[:, None]
    if temporal_taper > 0 and nt > 1:
        gather *= tukey(nt, temporal_taper).astype(np.float32)[None, :]

    nfft = (_next_power_of_two(nch), _next_power_of_two(nt))
    spectrum = fft2(gather, s=nfft)
    k = fftfreq(nfft[0], d=channel_spacing_m)[:, None]
    f = fftfreq(nfft[1], d=1.0 / sample_rate_hz)[None, :]
    mask = _high_velocity_mask(f, k, vmin, vmax)

    filtered = ifft2(spectrum * mask).real[:nch, :nt]
    return np.asarray(filtered.T, dtype=np.float32)


def compute_fk_spectrum_db(
    data: np.ndarray,
    *,
    sample_rate_hz: float,
    channel_spacing_m: float,
    taper_fraction: tuple[float, float] = DEFAULT_FK_TAPER,
    remove_background: bool = True,
    nfft_channels: int | None = None,
    nfft_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Return f-k amplitude in dB plus frequency and wavenumber axes."""
    _validate_fk_inputs(data, sample_rate_hz, channel_spacing_m)

    gather = np.asarray(data, dtype=np.float32).T.copy()
    nch, nt = gather.shape
    spatial_taper, temporal_taper = taper_fraction
    if spatial_taper > 0 and nch > 1:
        gather *= tukey(nch, spatial_taper).astype(np.float32)[:, None]
    if temporal_taper > 0 and nt > 1:
        gather *= tukey(nt, temporal_taper).astype(np.float32)[None, :]

    nfft = (
        _resolve_nfft(nfft_channels, nch, "FK spatial NFFT"),
        _resolve_nfft(nfft_samples, nt, "FK temporal NFFT"),
    )
    spectrum = fftshift(rfft2(gather, s=nfft), axes=0)
    frequency_hz = rfftfreq(nfft[1], d=1.0 / sample_rate_hz)
    wavenumber_cpm = fftshift(fftfreq(nfft[0], d=channel_spacing_m))

    amplitude = np.abs(spectrum)
    reference = float(np.percentile(amplitude, 99.0))
    if not math.isfinite(reference) or reference <= 0:
        reference = float(np.max(amplitude))
    if not math.isfinite(reference) or reference <= 0:
        reference = 1.0

    db = 20.0 * np.log10(amplitude / reference + np.finfo(np.float32).eps)
    if remove_background:
        db = db - np.median(db, axis=0, keepdims=True)

    return {
        "db": np.asarray(db, dtype=np.float32),
        "frequency_hz": np.asarray(frequency_hz, dtype=np.float32),
        "wavenumber_cpm": np.asarray(wavenumber_cpm, dtype=np.float32),
    }


def _validate_fk_inputs(data: np.ndarray, sample_rate_hz: float, channel_spacing_m: float) -> None:
    matrix = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError("DAS data must be a 2-D matrix shaped frames x channels.")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("f-k processing requires at least 2 samples and 2 channels.")
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    if channel_spacing_m <= 0:
        raise ValueError("Channel spacing must be positive.")


def _ordered_velocity_bounds(vmin_mps: float, vmax_mps: float) -> tuple[float, float]:
    vmin = float(vmin_mps)
    vmax = float(vmax_mps)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise ValueError("FK velocity bounds must be finite.")
    if vmin <= 0 or vmax <= 0:
        raise ValueError("FK velocity bounds must be positive.")
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if math.isclose(vmin, vmax):
        raise ValueError("FK velocity bounds must not be equal.")
    return vmin, vmax


def _high_velocity_mask(
    frequency_hz: np.ndarray,
    wavenumber_cpm: np.ndarray,
    vmin_mps: float,
    vmax_mps: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        apparent_velocity = np.abs(frequency_hz / wavenumber_cpm)

    mask = np.ones_like(apparent_velocity, dtype=np.float32)
    finite = np.isfinite(apparent_velocity)
    mask[finite & (apparent_velocity <= vmin_mps)] = 0.0

    transition = finite & (apparent_velocity > vmin_mps) & (apparent_velocity < vmax_mps)
    if np.any(transition):
        ratio = (apparent_velocity[transition] - vmin_mps) / (vmax_mps - vmin_mps)
        mask[transition] = 0.5 - 0.5 * np.cos(np.pi * ratio)

    return mask


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def _resolve_nfft(value: int | None, minimum: int, name: str) -> int:
    if value is None:
        return _next_power_of_two(minimum)

    nfft = int(value)
    if nfft < minimum:
        raise ValueError(f"{name} must be >= input length {minimum}.")
    return nfft
