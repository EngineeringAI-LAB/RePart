from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import time

import numpy as np

from .fit import fit_superquadric_tsdf
from .pipeline import _bbox_points_from_idx, _extract_regions, idx3d_flatten
from .types import GridSpec, MPSParams, ResolvedMPSParams

@dataclass
class FastMPSConfig:
    num_workers: int | None = None
    chunk_size: int = 1
    parallel_min_regions: int = 3


@dataclass
class _RegionTask:
    idx_region: int
    pixel_idx_list: np.ndarray
    bbox_matlab: np.ndarray
    centroid_matlab_xyz: np.ndarray


@dataclass
class _RegionResult:
    idx_region: int
    x_row: np.ndarray
    occ_idx_in: np.ndarray
    num_idx: np.ndarray
    pixel_idx_list: np.ndarray
    debug_msg: str | None = None


_WORK_SDF: np.ndarray | None = None
_WORK_GRID: GridSpec | None = None
_WORK_PARAMS: ResolvedMPSParams | None = None


def _init_worker(
    sdf: np.ndarray,
    grid: GridSpec,
    params: ResolvedMPSParams,
) -> None:
    global _WORK_SDF, _WORK_GRID, _WORK_PARAMS
    _WORK_SDF = sdf
    _WORK_GRID = grid
    _WORK_PARAMS = params


def _nearest_inside_centroid(
    centroid_matlab_xyz: np.ndarray,
    pixel_idx_list: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    centroid = np.maximum(np.floor(centroid_matlab_xyz).astype(np.int64), 1)
    centroid_flat = idx3d_flatten(
        np.array([[centroid[1]], [centroid[0]], [centroid[2]]], dtype=np.int64), grid
    )[0]
    if np.any(pixel_idx_list == centroid_flat):
        return grid.points[:, centroid_flat]
    pix_pts = grid.points[:, pixel_idx_list].T
    cpt = grid.points[:, centroid_flat]
    d = np.linalg.norm(pix_pts - cpt.reshape(1, 3), axis=1)
    return grid.points[:, pixel_idx_list[np.argmin(d)]]


def _fit_single_region(
    task: _RegionTask,
) -> _RegionResult:
    if _WORK_SDF is None or _WORK_GRID is None or _WORK_PARAMS is None:
        raise RuntimeError("fast_mps worker not initialized")
    sdf = _WORK_SDF
    grid = _WORK_GRID
    para = _WORK_PARAMS
    region_start = time.perf_counter()
    last_heartbeat = region_start
    refit_count = 0
    stagnation_count = 0
    idx = np.ceil(task.bbox_matlab).astype(np.int64)
    idx[3:6] = np.minimum(
        idx[0:3] + idx[3:6] + para.padding_size,
        np.array([grid.size[1], grid.size[0], grid.size[2]], dtype=np.int64),
    )
    idx[0:3] = np.maximum(idx[0:3] - para.padding_size, 1)

    centroid_xyz = _nearest_inside_centroid(task.centroid_matlab_xyz, task.pixel_idx_list, grid)
    valid = np.zeros(6, dtype=bool)
    occ_idx = np.array([], dtype=np.int64)
    num_idx = np.zeros(3, dtype=np.int64)
    x_row = np.zeros(11, dtype=np.float64)
    debug_msg = None

    while not valid.all():
        refit_count += 1
        now = time.perf_counter()
        if para.verbose and para.heartbeat_sec > 0 and (now - last_heartbeat) >= para.heartbeat_sec:
            print(
                f"  region {task.idx_region + 1}: refit {refit_count}, "
                f"elapsed {now - region_start:.1f}s"
            )
            last_heartbeat = now
        if para.max_region_refits is not None and refit_count > para.max_region_refits:
            debug_msg = f"region {task.idx_region + 1}: max_region_refits reached, skipped"
            break
        if para.region_timeout_sec is not None and (now - region_start) > para.region_timeout_sec:
            debug_msg = f"region {task.idx_region + 1}: timeout reached, skipped"
            break

        xg, yg, zg = np.meshgrid(
            np.arange(idx[1], idx[4] + 1),
            np.arange(idx[0], idx[3] + 1),
            np.arange(idx[2], idx[5] + 1),
            indexing="ij",
        )
        indices = np.vstack([xg.ravel(order="F"), yg.ravel(order="F"), zg.ravel(order="F")])
        roi_idx = idx3d_flatten(indices, grid)
        bounding_points = _bbox_points_from_idx(idx, grid)

        scale_init = para.scale_init_ratio * (idx[3:6] - idx[0:3]) * grid.interval
        x_init = np.array(
            [1.0, 1.0, scale_init[1], scale_init[0], scale_init[2], 0.0, 0.0, 0.0, *centroid_xyz]
        )
        fit = fit_superquadric_tsdf(
            sdf[roi_idx],
            x_init,
            grid.truncation,
            grid.points[:, roi_idx],
            roi_idx,
            bounding_points,
            para,
        )
        x_row = fit.x
        occ_idx = fit.occ_idx
        valid = fit.valid
        num_idx = fit.num_idx

        if not valid.all():
            ext = (~valid).astype(np.int64)
            ext[[0, 1]] = ext[[1, 0]]
            ext[[3, 4]] = ext[[4, 3]]
            hit_boundary = (
                np.any((idx[0:3] == 1) & (ext[0:3] == 1))
                or (idx[3] == grid.size[1] and ext[3] == 1)
                or (idx[4] == grid.size[0] and ext[4] == 1)
                or (idx[5] == grid.size[2] and ext[5] == 1)
            )
            if hit_boundary:
                break

            idx_extend = ext * para.padding_size
            idx_new = idx.copy()
            idx_new[3:6] = np.minimum(
                idx_new[3:6] + idx_extend[[4, 3, 5]],
                np.array([grid.size[1], grid.size[0], grid.size[2]], dtype=np.int64),
            )
            idx_new[0:3] = np.maximum(idx_new[0:3] - idx_extend[[1, 0, 2]], 1)
            if np.array_equal(idx_new, idx):
                stagnation_count += 1
                if stagnation_count >= 10:
                    debug_msg = f"region {task.idx_region + 1}: bbox stagnation, skipped"
                    break
            else:
                stagnation_count = 0
            idx = idx_new

    occ_idx_in = occ_idx[sdf[occ_idx] <= 0] if occ_idx.size else np.array([], dtype=np.int64)
    return _RegionResult(
        idx_region=task.idx_region,
        x_row=x_row,
        occ_idx_in=occ_idx_in,
        num_idx=num_idx,
        pixel_idx_list=task.pixel_idx_list,
        debug_msg=debug_msg,
    )


def mps_fast(
    sdf: np.ndarray,
    grid: GridSpec,
    params: MPSParams | None = None,
    fast_config: FastMPSConfig | None = None,
) -> np.ndarray:
    """
    Faster region-parallel variant of MPS.
    Keeps marching/division order but parallelizes per-region fitting.
    """
    para = params or MPSParams()
    cfg = fast_config or FastMPSConfig()
    rpara = para.resolved(grid)
    sdf = np.asarray(sdf, dtype=np.float64).copy()
    num_division = 1
    x_all: list[np.ndarray] = []
    dratio = 3.0 / 5.0
    conn_ratio = np.array([dratio**i for i in range(0, 9)], dtype=np.float64)
    conn_pointer = 0
    num_region = 1

    if cfg.num_workers is None:
        cfg = FastMPSConfig(
            num_workers=max(1, (os.cpu_count() or 1) - 1),
            chunk_size=cfg.chunk_size,
            parallel_min_regions=cfg.parallel_min_regions,
        )

    while num_division < para.max_division:
        if conn_pointer != 0 and num_region != 0:
            conn_pointer = 0
        finite_sdf = sdf[np.isfinite(sdf)]
        if finite_sdf.size == 0:
            break
        conn_threshold = conn_ratio[conn_pointer] * np.min(finite_sdf)
        if conn_threshold > -grid.truncation * 3e-1:
            break

        sdf3d_region = np.reshape(sdf, grid.size, order="F")
        roi = _extract_regions(sdf3d_region <= conn_threshold, rpara.min_area)
        num_region = len(roi)
        if para.verbose:
            print(f"[FAST] Number of regions: {num_region}")
        if num_region == 0:
            if conn_pointer < (conn_ratio.shape[0] - 1):
                conn_pointer += 1
                continue
            break

        tasks = [
            _RegionTask(
                idx_region=i,
                pixel_idx_list=r.pixel_idx_list,
                bbox_matlab=r.bbox_matlab,
                centroid_matlab_xyz=r.centroid_matlab_xyz,
            )
            for i, r in enumerate(roi)
        ]
        x_temp = np.zeros((num_region, 11), dtype=np.float64)
        del_idx = np.zeros(num_region, dtype=bool)
        occ_idx_in: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(num_region)]
        num_idx = np.zeros((num_region, 3), dtype=np.int64)

        if num_region < cfg.parallel_min_regions or cfg.num_workers == 1:
            results = []
            _init_worker(sdf, grid, rpara)
            for t in tasks:
                results.append(_fit_single_region(t))
        else:
            with ProcessPoolExecutor(
                max_workers=cfg.num_workers,
                initializer=_init_worker,
                initargs=(sdf, grid, rpara),
            ) as ex:
                results = list(ex.map(_fit_single_region, tasks, chunksize=max(1, cfg.chunk_size)))

        for r in results:
            i = r.idx_region
            x_temp[i, :] = r.x_row
            occ_idx_in[i] = r.occ_idx_in
            num_idx[i, :] = r.num_idx
            if para.verbose and r.debug_msg is not None:
                print(f"  [FAST] {r.debug_msg}")

        for i, t in enumerate(tasks):
            out_ratio = num_idx[i, 1] / max((num_idx[i, 0] + num_idx[i, 1]), 1)
            if out_ratio > 0.3 or num_idx[i, 0] < rpara.min_area or num_idx[i, 2] <= 1:
                del_idx[i] = True
                sdf[t.pixel_idx_list] = np.nan
                if para.verbose:
                    print(f"[FAST] region {i+1}/{num_region} ...REJECTED")
            else:
                sdf[occ_idx_in[i]] = np.nan
                if para.verbose:
                    print(f"[FAST] region {i+1}/{num_region} ...ACCEPTED")

        accepted = x_temp[~del_idx]
        if accepted.size > 0:
            x_all.append(accepted)
        num_division += 1

    if not x_all:
        return np.zeros((0, 11), dtype=np.float64)
    return np.vstack(x_all)
