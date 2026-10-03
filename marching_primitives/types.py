from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    size: tuple[int, int, int]
    value_range: tuple[float, float, float, float, float, float]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    points: np.ndarray
    interval: float
    truncation: float
    disp_range: tuple[float, float]
    visualize_arclength: float

    @staticmethod
    def from_size_and_range(
        size: Sequence[int],
        value_range: Sequence[float],
        truncation_scale: float = 1.2,
    ) -> "GridSpec":
        sx, sy, sz = (int(size[0]), int(size[1]), int(size[2]))
        xr0, xr1, yr0, yr1, zr0, zr1 = (
            float(value_range[0]),
            float(value_range[1]),
            float(value_range[2]),
            float(value_range[3]),
            float(value_range[4]),
            float(value_range[5]),
        )
        x = np.linspace(xr0, xr1, sx)
        y = np.linspace(yr0, yr1, sy)
        z = np.linspace(zr0, zr1, sz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        points = np.vstack([xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")])
        interval = (xr1 - xr0) / (sx - 1)
        truncation = truncation_scale * interval
        visualize_arclength = 0.01 * np.sqrt(xr1 - xr0)
        return GridSpec(
            size=(sx, sy, sz),
            value_range=(xr0, xr1, yr0, yr1, zr0, zr1),
            x=x,
            y=y,
            z=z,
            points=points,
            interval=interval,
            truncation=truncation,
            disp_range=(-np.inf, truncation),
            visualize_arclength=visualize_arclength,
        )


@dataclass(frozen=True)
class MPSParams:
    verbose: bool = True
    padding_size: int | None = None
    min_area: int | None = None
    max_division: int = 50
    scale_init_ratio: float = 0.1
    nan_range: float | None = None
    w: float = 0.99
    tolerance: float = 1e-6
    relative_tolerance: float = 1e-4
    switch_tolerance: float = 1e-1
    max_switch: int = 2
    iter_min: int = 2
    max_opti_iter: int = 2
    max_iter: int = 15
    active_multiplier: float = 3.0
    region_timeout_sec: float | None = None
    max_region_refits: int | None = None
    heartbeat_sec: float = 30.0

    def resolved(self, grid: GridSpec) -> "ResolvedMPSParams":
        padding_size = (
            int(np.ceil(12.0 * grid.truncation / grid.interval))
            if self.padding_size is None
            else int(self.padding_size)
        )
        min_area = int(np.ceil(grid.size[0] / 20.0)) if self.min_area is None else int(self.min_area)
        nan_range = 0.5 * grid.interval if self.nan_range is None else float(self.nan_range)
        return ResolvedMPSParams(
            verbose=bool(self.verbose),
            padding_size=padding_size,
            min_area=min_area,
            max_division=int(self.max_division),
            scale_init_ratio=float(self.scale_init_ratio),
            nan_range=nan_range,
            w=float(self.w),
            tolerance=float(self.tolerance),
            relative_tolerance=float(self.relative_tolerance),
            switch_tolerance=float(self.switch_tolerance),
            max_switch=int(self.max_switch),
            iter_min=int(self.iter_min),
            max_opti_iter=int(self.max_opti_iter),
            max_iter=int(self.max_iter),
            active_multiplier=float(self.active_multiplier),
            region_timeout_sec=(
                None if self.region_timeout_sec is None else float(self.region_timeout_sec)
            ),
            max_region_refits=(
                None if self.max_region_refits is None else int(self.max_region_refits)
            ),
            heartbeat_sec=float(self.heartbeat_sec),
        )


@dataclass(frozen=True)
class ResolvedMPSParams:
    verbose: bool
    padding_size: int
    min_area: int
    max_division: int
    scale_init_ratio: float
    nan_range: float
    w: float
    tolerance: float
    relative_tolerance: float
    switch_tolerance: float
    max_switch: int
    iter_min: int
    max_opti_iter: int
    max_iter: int
    active_multiplier: float
    region_timeout_sec: float | None
    max_region_refits: int | None
    heartbeat_sec: float
