"""View preparation helpers for efficient image rendering."""

from __future__ import annotations

import math

import numpy as np

# Maximum number of columns uploaded to the GPU image item. Rows are bounded by
# the visible point range (typically <= 100), so this caps the texture at about
# 1.6 million pixels. Zooming in re-slices the full-resolution data, so detail
# is never lost - only the on-screen envelope is decimated.
MAX_IMAGE_DISPLAY_COLUMNS = 16384

# Stats helpers estimate robust values from a decimated subset for very long
# traces; the estimate is only used for display scaling, never for the data
# itself.
_MAX_STAT_SAMPLES = 1_000_000


def prepare_image_matrix(
    frame_data: np.ndarray,
    start_point: int,
    end_point: int,
) -> np.ndarray:
    """Extract a 0-based inclusive point range and return an image matrix as points x frames."""
    if start_point < 0 or end_point < start_point or end_point >= frame_data.shape[1]:
        raise ValueError("Invalid point range.")

    # UI labels and matrix indices are both zero-based; the range end is inclusive.
    point_slice = slice(start_point, end_point + 1)
    return np.ascontiguousarray(frame_data[:, point_slice].T)


def downsample_image_for_display(
    image: np.ndarray,
    max_columns: int = MAX_IMAGE_DISPLAY_COLUMNS,
) -> tuple[np.ndarray, float]:
    """Decimate an image along its columns for display only.

    Returns ``(display, samples_per_col)``. Each block of ``stride`` original
    columns becomes a single display column holding the block's mean value.
    Averaging is an unbiased, low-pass decimation: with a fixed VMin/VMax the
    zoomed-out image keeps the same brightness as the full-resolution view
    instead of drifting darker (the previous "largest absolute value" rule
    picked the block extreme and made every decimated column appear saturated).
    The full-resolution array is never modified, and the returned mapping lets
    the caller re-slice the original data at higher resolution when zooming in.

    ``samples_per_col`` maps one display column to original columns
    (``stride`` when decimated, ``1.0`` otherwise).
    """
    image = np.asarray(image)
    nrows, ncols = image.shape
    if ncols <= max_columns:
        return np.ascontiguousarray(image), 1.0

    # One display column per block -> stride covers cols/max.
    stride = int(math.ceil(ncols / max_columns))
    nblocks = int(math.ceil(ncols / stride))
    padded = np.full((nrows, nblocks * stride), np.nan, dtype=image.dtype)
    padded[:, :ncols] = image
    blocks = padded.reshape(nrows, nblocks, stride)
    with np.errstate(invalid="ignore"):
        # NaN padding only extends the final block; nanmean ignores it.
        display = np.nanmean(blocks, axis=2)
    display = np.nan_to_num(display, nan=0.0)
    return np.ascontiguousarray(display, dtype=image.dtype), float(stride)


def robust_min_max(
    values: np.ndarray,
    max_samples: int = _MAX_STAT_SAMPLES,
) -> tuple[float, float]:
    """Robust (min, max) estimated from a strided subset for very large arrays."""
    values = np.asarray(values)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size <= max_samples:
        return float(np.nanmin(values)), float(np.nanmax(values))

    nrows = values.shape[0] if values.ndim > 1 else 1
    per_row = max(1, int(max_samples // nrows))
    length = values.shape[-1]
    step = max(1, int(math.ceil(length / per_row)))
    subset = values[:, ::step] if values.ndim > 1 else values[::step]
    return float(np.nanmin(subset)), float(np.nanmax(subset))


def robust_median(values: np.ndarray, max_samples: int = _MAX_STAT_SAMPLES) -> float:
    """Median estimated from a strided subset for very long traces."""
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    if values.size <= max_samples:
        return float(np.nanmedian(values))
    step = max(1, int(math.ceil(values.size / max_samples)))
    return float(np.nanmedian(values[::step]))


def robust_abs_percentile(
    values: np.ndarray,
    percentile: float,
    max_samples: int = _MAX_STAT_SAMPLES,
) -> float:
    """Percentile of |values| estimated from a strided subset."""
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    if values.size <= max_samples:
        return float(np.nanpercentile(np.abs(values), percentile))
    step = max(1, int(math.ceil(values.size / max_samples)))
    return float(np.nanpercentile(np.abs(values[::step]), percentile))