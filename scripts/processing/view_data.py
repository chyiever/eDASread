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
    """Extract a 1-based point range and return an image matrix as points x frames."""
    if start_point < 1 or end_point < start_point or end_point > frame_data.shape[1]:
        raise ValueError("Invalid point range.")

    # UI uses 1-based labels while the internal matrix uses zero-based indices.
    point_slice = slice(start_point - 1, end_point)
    return np.ascontiguousarray(frame_data[:, point_slice].T)


def downsample_image_for_display(
    image: np.ndarray,
    max_columns: int = MAX_IMAGE_DISPLAY_COLUMNS,
) -> tuple[np.ndarray, float]:
    """Decimate an image along its columns for display only.

    Returns ``(display, samples_per_col)``. Each block of ``stride`` original
    columns becomes two display columns holding the block's minimum and maximum,
    so extreme values (positive and negative peaks) are always preserved. The
    full-resolution array is never modified, and the returned mapping lets the
    caller re-slice the original data at higher resolution when zooming in.

    ``samples_per_col`` maps one display column to original columns
    (``stride / 2`` when decimated, ``1.0`` otherwise).
    """
    image = np.asarray(image)
    nrows, ncols = image.shape
    if ncols <= max_columns:
        return np.ascontiguousarray(image), 1.0

    # 2 display columns per block (min/max) -> stride covers 2*cols/max.
    stride = int(math.ceil(2.0 * ncols / max_columns))
    nblocks = int(math.ceil(ncols / stride))
    padded = np.full((nrows, nblocks * stride), np.nan, dtype=image.dtype)
    padded[:, :ncols] = image
    blocks = padded.reshape(nrows, nblocks, stride)
    with np.errstate(invalid="ignore"):
        block_min = np.nanmin(blocks, axis=2)
        block_max = np.nanmax(blocks, axis=2)

    display = np.empty((nrows, nblocks * 2), dtype=image.dtype)
    display[:, 0::2] = block_min
    display[:, 1::2] = block_max
    return np.ascontiguousarray(display), stride / 2.0


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