"""View preparation helpers for efficient image rendering."""

from __future__ import annotations

import numpy as np


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
