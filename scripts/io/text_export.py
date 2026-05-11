"""TXT export helpers for eDAS phase data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_phase_txt(output_path: str | Path, phase_data: np.ndarray, fmt: str = "%.9f") -> None:
    """Write frames x points phase data to a text file in radians."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.asarray(phase_data, dtype=np.float32), fmt=fmt)
