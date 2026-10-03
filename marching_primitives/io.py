from __future__ import annotations

import numpy as np

from .types import GridSpec


def load_sdf_csv(path: str) -> tuple[np.ndarray, GridSpec]:
    """Load a TSDF CSV produced by scripts/mesh_to_sdf.py."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64).reshape(-1)
    resolution = int(data[0])
    value_range = data[1:7]
    sdf = data[7:]
    grid = GridSpec.from_size_and_range((resolution, resolution, resolution), value_range)
    sdf = np.clip(sdf, -grid.truncation, grid.truncation)
    return sdf, grid
