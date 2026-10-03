"""Superquadric parameters, assignment, and group geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from marching_primitives import load_sdf_csv
from marching_primitives.types import GridSpec

EPS_MIN = 0.1


EPS_MAX = 2.5


def clamp_eps(e) -> np.ndarray:
    """Clamp physical eps values to the supported range (numpy, float64)."""
    return np.clip(np.asarray(e, dtype=np.float64), EPS_MIN, EPS_MAX)


@dataclass
class SQPrimitive:
    t: np.ndarray
    r: np.ndarray
    log_a: np.ndarray
    raw_e: np.ndarray
    active: bool = True

    def clone(self) -> "SQPrimitive":
        return SQPrimitive(
            self.t.copy(),
            self.r.copy(),
            self.log_a.copy(),
            self.raw_e.copy(),
            self.active,
        )


def euler_to_matrix_np(euler: np.ndarray) -> np.ndarray:
    z, y, x = (float(value) for value in euler)
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return rz @ ry @ rx


def _primitive_params(prim: SQPrimitive) -> np.ndarray:
    eps = clamp_eps(prim.raw_e)
    scales = np.maximum(1e-4, np.exp(prim.log_a.astype(np.float64)))
    return np.array([eps[0], eps[1], *scales, *prim.r, *prim.t], dtype=np.float64)


def _distance(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    e1 = max(0.05, float(params[0]))
    e2 = max(0.05, float(params[1]))
    a1, a2, a3 = [max(1e-4, float(value)) for value in params[2:5]]
    rotation = euler_to_matrix_np(params[5:8])
    translation = params[8:11]
    centered = points @ rotation - translation @ rotation
    radius = np.sqrt(np.sum(centered**2, axis=1)) + 1e-8
    term = (
        ((centered[:, 0] / a1) ** 2) ** (1.0 / e2)
        + ((centered[:, 1] / a2) ** 2) ** (1.0 / e2)
    ) ** (e2 / e1) + ((centered[:, 2] / a3) ** 2) ** (1.0 / e1)
    return radius * np.abs(np.power(np.maximum(term, 1e-8), -e1 / 2.0) - 1.0)


def _inside(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    e1 = max(0.05, float(params[0]))
    e2 = max(0.05, float(params[1]))
    a1, a2, a3 = [max(1e-4, float(value)) for value in params[2:5]]
    rotation = euler_to_matrix_np(params[5:8].astype(np.float32))
    translation = params[8:11]
    local = points @ rotation - translation @ rotation
    xy = (
        np.abs(local[:, 0] / a1) ** (2.0 / e2) + np.abs(local[:, 1] / a2) ** (2.0 / e2)
    ) ** (e2 / e1)
    z = np.abs(local[:, 2] / a3) ** (2.0 / e1)
    return (xy + z).astype(np.float32)


def primitive_assignments(
    primitives: list[SQPrimitive], points: np.ndarray, max_prims: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del max_prims
    count = len(primitives)
    point_count = len(points)
    if count == 0:
        empty = np.zeros((0, point_count), dtype=np.float32)
        return np.zeros(point_count, dtype=np.int64), empty, empty, empty

    params = [_primitive_params(prim) for prim in primitives]
    distances = np.stack(
        [_distance(points, item).astype(np.float32) for item in params]
    )
    inside = np.stack([_inside(points, item) for item in params])
    temperature = float(np.median(distances) + 1e-3)
    logits = -distances / max(temperature, 1e-3)
    logits -= np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    soft = exp_logits / (np.sum(exp_logits, axis=0, keepdims=True) + 1e-8)
    assigned = np.argmax(soft, axis=0).astype(np.int64)
    occupancy = np.exp(-distances / max(temperature, 1e-3)).astype(np.float32)
    return assigned, soft.astype(np.float32), inside, occupancy


def _surface_grid(prim: SQPrimitive, nu: int, nv: int, closed: bool) -> np.ndarray:
    scales = np.exp(prim.log_a).astype(np.float32)
    e1, e2 = (float(value) for value in clamp_eps(prim.raw_e))
    eta = np.linspace(-np.pi / 2, np.pi / 2, nv)
    omega = np.linspace(-np.pi, np.pi, nu, endpoint=not closed)
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")

    def signed_power(value: np.ndarray, power: float) -> np.ndarray:
        return np.sign(value) * (np.abs(value) ** power)

    cos_eta, sin_eta = np.cos(eta_grid), np.sin(eta_grid)
    cos_omega, sin_omega = np.cos(omega_grid), np.sin(omega_grid)
    local = np.stack(
        [
            scales[0] * signed_power(cos_eta, e1) * signed_power(cos_omega, e2),
            scales[1] * signed_power(cos_eta, e1) * signed_power(sin_omega, e2),
            scales[2] * signed_power(sin_eta, e1),
        ],
        axis=-1,
    ).reshape(-1, 3)
    return (local @ euler_to_matrix_np(prim.r).T + prim.t[None, :]).astype(np.float32)


def primitive_to_surface_points(
    prim: SQPrimitive, nu: int = 36, nv: int = 18
) -> np.ndarray:
    return _surface_grid(prim, nu, nv, closed=False)


def primitive_to_surface_mesh(
    prim: SQPrimitive, nu: int = 40, nv: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    vertices = _surface_grid(prim, nu, nv, closed=True)
    faces = []
    for i in range(nv - 1):
        for j in range(nu):
            next_j = (j + 1) % nu
            v00, v01 = i * nu + j, i * nu + next_j
            v10, v11 = (i + 1) * nu + j, (i + 1) * nu + next_j
            faces.extend(([v00, v10, v11], [v00, v11, v01]))
    return vertices, np.asarray(faces, dtype=np.int32)


def load_global_sdf_if_needed(
    sdf_csv_path: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[GridSpec]]:
    if sdf_csv_path is None:
        return None, None
    sdf, grid = load_sdf_csv(sdf_csv_path)
    return np.asarray(sdf, dtype=np.float64), grid


def primitive_to_mps_x(prim: SQPrimitive) -> np.ndarray:
    a = np.clip(np.exp(np.asarray(prim.log_a, dtype=np.float64)), 1e-3, 3.0)
    eps = clamp_eps(prim.raw_e)
    r = np.asarray(prim.r, dtype=np.float64)
    t = np.asarray(prim.t, dtype=np.float64)
    return np.array(
        [eps[0], eps[1], a[0], a[1], a[2], r[0], r[1], r[2], t[0], t[1], t[2]],
        dtype=np.float64,
    )


def mps_x_to_primitive(x: np.ndarray) -> SQPrimitive:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != 11:
        raise ValueError(f"Expected 11 SQ parameters, got {x.shape[0]}")
    eps = clamp_eps(x[0:2]).astype(np.float32)
    scales = np.clip(x[2:5], 1e-3, 3.0)
    return SQPrimitive(
        t=x[8:11].astype(np.float32),
        r=x[5:8].astype(np.float32),
        log_a=np.log(scales).astype(np.float32),
        raw_e=eps,
        active=True,
    )


def rotm_zyx(euler_zyx: np.ndarray) -> np.ndarray:
    z, y, x = [float(v) for v in euler_zyx]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return rz @ ry @ rx


def sq_F(points: np.ndarray, prim: SQPrimitive) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = np.clip(np.exp(np.asarray(prim.log_a, dtype=np.float64)), 1e-4, 10.0)
    e = clamp_eps(prim.raw_e)
    t = np.asarray(prim.t, dtype=np.float64)
    r = np.asarray(prim.r, dtype=np.float64)
    R = rotm_zyx(r)
    local = (pts - t[None, :]) @ R
    x = np.abs(local[:, 0] / a[0])
    y = np.abs(local[:, 1] / a[1])
    z = np.abs(local[:, 2] / a[2])
    ex2 = 2.0 / e[1]
    ex1 = 2.0 / e[0]
    term_xy = np.power(np.power(x, ex2) + np.power(y, ex2) + 1e-12, e[1] / e[0])
    term_z = np.power(z, ex1)
    return term_xy + term_z


def sq_surface_distance(points: np.ndarray, prim: SQPrimitive) -> np.ndarray:
    Fv = sq_F(points, prim)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(pts - np.asarray(prim.t, dtype=np.float64)[None, :], axis=1)
    return r * np.abs(
        np.power(np.maximum(Fv, 1e-12), -0.5 * float(clamp_eps(prim.raw_e)[0])) - 1.0
    )


def assign_points_to_primitives(
    primitives: Sequence[SQPrimitive], points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(primitives) == 0:
        return np.zeros((pts.shape[0],), dtype=np.int64), np.zeros(
            (0, pts.shape[0]), dtype=np.float32
        )
    assign, soft_assign, _, _ = primitive_assignments(
        list(primitives), pts, max_prims=None
    )
    return np.asarray(assign, dtype=np.int64), np.asarray(soft_assign, dtype=np.float32)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = float(np.logical_and(mask1, mask2).sum())
    union = float(np.logical_or(mask1, mask2).sum())
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_eval_segmentation_metrics(
    pred_assign: np.ndarray, gt_labels: np.ndarray
) -> Dict[str, float]:
    if pred_assign.size == 0 or gt_labels.size == 0:
        return {"part_miou": 0.0}

    valid_mask = gt_labels != -1
    if not np.any(valid_mask):
        return {"part_miou": 0.0}

    pred = np.asarray(pred_assign[valid_mask], dtype=np.int64)
    gt = np.asarray(gt_labels[valid_mask], dtype=np.int64)
    gt_ids = np.unique(gt)
    if gt_ids.size == 0:
        return {"part_miou": 0.0}

    pred_ids = np.unique(pred)
    pred_masks = {int(pid): (pred == pid) for pid in pred_ids}

    part_iou_sum = 0.0

    for gid in gt_ids:
        gt_mask = gt == gid
        if float(gt_mask.sum()) <= 0.0:
            continue

        best_iou = 0.0
        relevant_pred_ids = np.unique(pred[gt_mask])
        for pid in relevant_pred_ids:
            iou = compute_mask_iou(gt_mask, pred_masks[int(pid)])
            if iou > best_iou:
                best_iou = iou

        part_iou_sum += best_iou

    part_miou = part_iou_sum / float(len(gt_ids))
    return {"part_miou": float(part_miou)}


def primitive_rotation_matrix(prim: SQPrimitive) -> np.ndarray:
    return rotm_zyx(np.asarray(prim.r, dtype=np.float64))


def primitive_axis_aligned_bounds(prim: SQPrimitive) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(prim.t, dtype=np.float64)
    scales = np.exp(np.asarray(prim.log_a, dtype=np.float64))
    R = primitive_rotation_matrix(prim)
    extent_world = np.abs(R) @ scales
    return center - extent_world, center + extent_world


def aabb_gap_distance(
    bounds_a: Tuple[np.ndarray, np.ndarray],
    bounds_b: Tuple[np.ndarray, np.ndarray],
) -> float:
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b
    sep = np.maximum(np.maximum(min_b - max_a, min_a - max_b), 0.0)
    return float(np.linalg.norm(sep))


def aabb_iou(
    bounds_a: Tuple[np.ndarray, np.ndarray],
    bounds_b: Tuple[np.ndarray, np.ndarray],
) -> float:
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_extent = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_extent))
    vol_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    union = vol_a + vol_b - inter_vol
    if union <= 1e-12:
        return 0.0
    return float(inter_vol / union)


def aabb_volume(bounds: Tuple[np.ndarray, np.ndarray]) -> float:
    bmin, bmax = bounds
    return float(np.prod(np.maximum(bmax - bmin, 0.0)))


def aabb_union(
    bounds_a: Tuple[np.ndarray, np.ndarray], bounds_b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    return np.minimum(bounds_a[0], bounds_b[0]), np.maximum(bounds_a[1], bounds_b[1])


def aabb_contact_ratio(
    bounds_a: Tuple[np.ndarray, np.ndarray],
    bounds_b: Tuple[np.ndarray, np.ndarray],
) -> float:
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b
    extent_a = np.maximum(max_a - min_a, 0.0)
    extent_b = np.maximum(max_b - min_b, 0.0)
    surface_a = 2.0 * float(
        extent_a[0] * extent_a[1]
        + extent_a[0] * extent_a[2]
        + extent_a[1] * extent_a[2]
    )
    surface_b = 2.0 * float(
        extent_b[0] * extent_b[1]
        + extent_b[0] * extent_b[2]
        + extent_b[1] * extent_b[2]
    )
    denom = max(min(surface_a, surface_b), 1e-12)
    best_area = 0.0
    for axis in range(3):
        face_gap = max(
            float(min_b[axis] - max_a[axis]), float(min_a[axis] - max_b[axis]), 0.0
        )
        if face_gap > 1e-6:
            continue
        other_axes = [ax for ax in range(3) if ax != axis]
        overlap = [
            max(0.0, float(min(max_a[ax], max_b[ax]) - max(min_a[ax], min_b[ax])))
            for ax in other_axes
        ]
        best_area = max(best_area, overlap[0] * overlap[1])
    return float(np.clip(best_area / denom, 0.0, 1.0))


def grouped_assignments_from_membership(
    pred_assign: np.ndarray,
    primitive_groups: Sequence[int],
    num_groups: int,
) -> np.ndarray:
    out = np.full_like(pred_assign, fill_value=-1, dtype=np.int64)
    group_map = {idx: int(gid) for idx, gid in enumerate(primitive_groups)}
    for pid, gid in group_map.items():
        out[pred_assign == pid] = gid
    return out
