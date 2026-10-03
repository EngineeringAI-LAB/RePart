from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import find_objects, label

from .fit import fit_superquadric_tsdf
from .types import GridSpec, MPSParams


@dataclass
class Region:
    pixel_idx_list: np.ndarray
    area: int
    # MATLAB regionprops convention:
    # centroid = [x(col), y(row), z]
    centroid_matlab_xyz: np.ndarray
    # BoundingBox = [x, y, z, width, height, depth]
    bbox_matlab: np.ndarray
    idx: np.ndarray | None = None
    bounding_points: np.ndarray | None = None
    centroid_xyz_point: np.ndarray | None = None


def idx3d_flatten(idx3d_xyz_1based: np.ndarray, grid: GridSpec) -> np.ndarray:
    sx, sy, _ = grid.size
    x = idx3d_xyz_1based[0, :] - 1
    y = idx3d_xyz_1based[1, :] - 1
    z = idx3d_xyz_1based[2, :] - 1
    return (x + sx * y + sx * sy * z).astype(np.int64)


def idx2coordinate(idx_xyz_1based: np.ndarray, grid: GridSpec) -> np.ndarray:
    idx_floor = np.floor(idx_xyz_1based).astype(np.int64)
    idx_floor[idx_floor == 0] = 1
    x = grid.x[idx_floor[0] - 1] + (idx_xyz_1based[0] - idx_floor[0]) * grid.interval
    y = grid.y[idx_floor[1] - 1] + (idx_xyz_1based[1] - idx_floor[1]) * grid.interval
    z = grid.z[idx_floor[2] - 1] + (idx_xyz_1based[2] - idx_floor[2]) * grid.interval
    return np.vstack([x, y, z])


def _extract_regions(mask_xyz: np.ndarray, min_area: int) -> list[Region]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lbl, n = label(mask_xyz, structure=structure)
    slices = find_objects(lbl)
    out: list[Region] = []
    for i in range(1, n + 1):
        s = slices[i - 1]
        if s is None:
            continue
        vox = np.argwhere(lbl == i)
        area = int(vox.shape[0])
        if area < min_area:
            continue
        # vox[:,0]=row(y), vox[:,1]=col(x), vox[:,2]=z in MATLAB terms.
        py = vox[:, 0] + 1
        px = vox[:, 1] + 1
        pz = vox[:, 2] + 1
        pixel_idx_list = (
            (py - 1)
            + mask_xyz.shape[0] * (px - 1)
            + mask_xyz.shape[0] * mask_xyz.shape[1] * (pz - 1)
        ).astype(np.int64)

        x0 = int(px.min())
        y0 = int(py.min())
        z0 = int(pz.min())
        x1 = int(px.max())
        y1 = int(py.max())
        z1 = int(pz.max())
        bbox_matlab = np.array(
            [x0, y0, z0, x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=np.float64
        )
        centroid_matlab = np.array([px.mean(), py.mean(), pz.mean()], dtype=np.float64)
        out.append(
            Region(
                pixel_idx_list=pixel_idx_list,
                area=area,
                centroid_matlab_xyz=centroid_matlab,
                bbox_matlab=bbox_matlab,
            )
        )
    return out


def _bbox_points_from_idx(idx_matlab_1based: np.ndarray, grid: GridSpec) -> np.ndarray:
    x0, y0, z0, x1, y1, z1 = idx_matlab_1based.astype(np.float64)
    # Match MATLAB ordering exactly:
    # [idx(2), idx(2), idx(5), idx(5), idx(2), idx(2), idx(5), idx(5);
    #  idx(1), idx(1), idx(1), idx(1), idx(4), idx(4), idx(4), idx(4);
    #  idx(3), idx(6), idx(3), idx(6), idx(3), idx(6), idx(3), idx(6)]
    corners = np.array(
        [
            [y0, y0, y1, y1, y0, y0, y1, y1],
            [x0, x0, x0, x0, x1, x1, x1, x1],
            [z0, z1, z0, z1, z0, z1, z0, z1],
        ],
        dtype=np.float64,
    )
    return idx2coordinate(corners, grid)


def mps(sdf: np.ndarray, grid: GridSpec, params: MPSParams | None = None) -> np.ndarray:
    para = (params or MPSParams()).resolved(grid)
    sdf = np.asarray(sdf, dtype=np.float64).copy()
    num_division = 1
    x_all: list[np.ndarray] = []
    dratio = 3.0 / 5.0
    conn_ratio = np.array([dratio**i for i in range(0, 9)], dtype=np.float64)
    conn_pointer = 0
    num_region = 1

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
        roi = _extract_regions(sdf3d_region <= conn_threshold, para.min_area)
        num_region = len(roi)
        if para.verbose:
            print(f"Number of regions: {num_region}")
        if num_region == 0:
            if conn_pointer < (conn_ratio.shape[0] - 1):
                conn_pointer += 1
                continue
            break

        x_temp = np.zeros((num_region, 11), dtype=np.float64)
        del_idx = np.zeros(num_region, dtype=bool)
        occ_idx_in: list[np.ndarray] = []
        num_idx = np.zeros((num_region, 3), dtype=np.int64)

        for i, r in enumerate(roi):
            region_start = time.perf_counter()
            last_heartbeat = region_start
            refit_count = 0
            stagnation_count = 0
            idx = np.ceil(r.bbox_matlab).astype(np.int64)
            idx[3:6] = np.minimum(
                idx[0:3] + idx[3:6] + para.padding_size,
                np.array([grid.size[1], grid.size[0], grid.size[2]], dtype=np.int64),
            )
            idx[0:3] = np.maximum(idx[0:3] - para.padding_size, 1)
            xg, yg, zg = np.meshgrid(
                np.arange(idx[1], idx[4] + 1),
                np.arange(idx[0], idx[3] + 1),
                np.arange(idx[2], idx[5] + 1),
                indexing="ij",
            )
            indices = np.vstack([xg.ravel(order="F"), yg.ravel(order="F"), zg.ravel(order="F")])
            r.idx = idx3d_flatten(indices, grid)
            r.bounding_points = _bbox_points_from_idx(idx, grid)

            centroid = np.maximum(np.floor(r.centroid_matlab_xyz).astype(np.int64), 1)
            centroid_flat = idx3d_flatten(
                np.array([[centroid[1]], [centroid[0]], [centroid[2]]], dtype=np.int64), grid
            )[0]
            if np.any(r.pixel_idx_list == centroid_flat):
                r.centroid_xyz_point = grid.points[:, centroid_flat]
            else:
                pix_pts = grid.points[:, r.pixel_idx_list].T
                cpt = grid.points[:, centroid_flat]
                d = np.linalg.norm(pix_pts - cpt.reshape(1, 3), axis=1)
                r.centroid_xyz_point = grid.points[:, r.pixel_idx_list[np.argmin(d)]]

            valid = np.zeros(6, dtype=bool)
            occ_idx = np.array([], dtype=np.int64)
            while not valid.all():
                refit_count += 1
                now = time.perf_counter()
                if para.verbose and para.heartbeat_sec > 0 and (now - last_heartbeat) >= para.heartbeat_sec:
                    print(
                        f"  division {num_division}, region {i+1}/{num_region}, "
                        f"refit {refit_count}, elapsed {now - region_start:.1f}s, "
                        f"roi_voxels {r.idx.size}"
                    )
                    last_heartbeat = now
                if para.max_region_refits is not None and refit_count > para.max_region_refits:
                    if para.verbose:
                        print(
                            f"  division {num_division}, region {i+1}/{num_region} reached "
                            f"max_region_refits={para.max_region_refits}, skipping region."
                        )
                    break
                if para.region_timeout_sec is not None and (now - region_start) > para.region_timeout_sec:
                    if para.verbose:
                        print(
                            f"  division {num_division}, region {i+1}/{num_region} exceeded "
                            f"region_timeout_sec={para.region_timeout_sec:.1f}, skipping region."
                        )
                    break

                scale_init = para.scale_init_ratio * (idx[3:6] - idx[0:3]) * grid.interval
                x_init = np.array(
                    [1.0, 1.0, scale_init[1], scale_init[0], scale_init[2], 0.0, 0.0, 0.0, *r.centroid_xyz_point]
                )
                fit = fit_superquadric_tsdf(
                    sdf[r.idx],
                    x_init,
                    grid.truncation,
                    grid.points[:, r.idx],
                    r.idx,
                    r.bounding_points,
                    para,
                )
                x_temp[i, :] = fit.x
                occ_idx = fit.occ_idx
                valid = fit.valid
                num_idx[i, :] = fit.num_idx

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
                            if para.verbose:
                                print(
                                    f"  division {num_division}, region {i+1}/{num_region} "
                                    "bbox expansion stagnated, skipping region."
                                )
                            break
                    else:
                        stagnation_count = 0
                    idx = idx_new
                    xg, yg, zg = np.meshgrid(
                        np.arange(idx[1], idx[4] + 1),
                        np.arange(idx[0], idx[3] + 1),
                        np.arange(idx[2], idx[5] + 1),
                        indexing="ij",
                    )
                    indices = np.vstack([xg.ravel(order="F"), yg.ravel(order="F"), zg.ravel(order="F")])
                    r.idx = idx3d_flatten(indices, grid)
                    r.bounding_points = _bbox_points_from_idx(idx, grid)

            occ_idx_in.append(occ_idx[sdf[occ_idx] <= 0])

        for i, r in enumerate(roi):
            out_ratio = num_idx[i, 1] / max((num_idx[i, 0] + num_idx[i, 1]), 1)
            if out_ratio > 0.3 or num_idx[i, 0] < para.min_area or num_idx[i, 2] <= 1:
                del_idx[i] = True
                sdf[r.pixel_idx_list] = np.nan
                if para.verbose:
                    print(f"region {i+1}/{num_region} ...REJECTED")
            else:
                sdf[occ_idx_in[i]] = np.nan
                if para.verbose:
                    print(f"region {i+1}/{num_region} ...ACCEPTED")

        accepted = x_temp[~del_idx]
        if accepted.size > 0:
            x_all.append(accepted)
        num_division += 1

    if not x_all:
        return np.zeros((0, 11), dtype=np.float64)
    return np.vstack(x_all)
