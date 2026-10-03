from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .math_sq import eul2rotm, rotm2eul_zyx, rotz_deg, sdf_superquadric
from .types import ResolvedMPSParams


@dataclass
class FitResult:
    x: np.ndarray
    occ_idx: np.ndarray
    valid: np.ndarray
    num_idx: np.ndarray


def _project_x0_to_bounds(x0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # SciPy requires strict x0/bounds validity; MATLAB lsqnonlin is less strict.
    # Keep this as a minimal compatibility bridge only when needed.
    eps = 1e-12
    ub_safe = ub.copy()
    bad = ub_safe <= lb
    if np.any(bad):
        ub_safe[bad] = lb[bad] + eps
    x_proj = x0.copy()
    x_proj = np.maximum(x_proj, lb + eps)
    x_proj = np.minimum(x_proj, ub_safe - eps)
    return x_proj, lb, ub_safe


def difference_sq_sdf(
    para: np.ndarray,
    sdf: np.ndarray,
    points: np.ndarray,
    truncation: float,
    weight: np.ndarray,
) -> np.ndarray:
    sdf_para = sdf_superquadric(para, points, truncation)
    dist = (sdf_para - sdf) * np.sqrt(weight)
    # Keep optimizer stable on Inf/NaN residuals due extreme exponents.
    return np.nan_to_num(dist, nan=1e6, posinf=1e6, neginf=-1e6)


def inlier_weight(
    sdf_active: np.ndarray,
    active_idx: np.ndarray,
    sdf_current: np.ndarray,
    sigma2: float,
    w: float,
    truncation: float,
) -> np.ndarray:
    in_idx = sdf_active < 0.0 * truncation
    sdf_cur = sdf_current[active_idx]
    # Keep the Gaussian normalization finite for samples whose world-space
    # grid interval (and therefore truncation) is very large or very small.
    tiny = np.finfo(np.float64).tiny
    sigma2_safe = max(float(sigma2), tiny)
    truncation_safe = max(abs(float(truncation)), tiny)
    normalizer_base = max(2.0 * np.pi * sigma2_safe, tiny)
    normalizer_inv = normalizer_base ** (-0.5)
    denominator = (1.0 - float(w)) * normalizer_inv * truncation_safe
    const = np.inf if denominator == 0.0 else float(w) / denominator
    dist = np.clip(sdf_cur[in_idx], -truncation, truncation) - sdf_active[in_idx]
    weight = np.ones_like(sdf_active)
    p = np.exp(-0.5 / sigma2 * dist * dist)
    p = p / (const + p)
    weight[in_idx] = p
    return weight


def _check_validity(x: np.ndarray, t_lb: np.ndarray, t_ub: np.ndarray, truncation: float) -> np.ndarray:
    r = eul2rotm(x[5:8])
    cp = np.vstack(
        [
            x[8:11] - r[:, 0] * x[2],
            x[8:11] + r[:, 0] * x[2],
            x[8:11] - r[:, 1] * x[3],
            x[8:11] + r[:, 1] * x[3],
            x[8:11] - r[:, 2] * x[4],
            x[8:11] + r[:, 2] * x[4],
        ]
    )
    valid = np.zeros(6, dtype=bool)
    valid[0:3] = cp.min(axis=0) >= (t_lb - truncation)
    valid[3:6] = cp.max(axis=0) <= (t_ub + truncation)
    return valid


def fit_superquadric_tsdf(
    sdf: np.ndarray,
    x_init: np.ndarray,
    truncation: float,
    points: np.ndarray,
    roi_idx: np.ndarray,
    bounding_points: np.ndarray,
    para: ResolvedMPSParams,
) -> FitResult:
    valid = np.zeros(6, dtype=bool)
    t_lb = bounding_points[:, 0]
    t_ub = bounding_points[:, 7]
    lb = np.array([0.0, 0.0, truncation, truncation, truncation, -2 * np.pi, -2 * np.pi, -2 * np.pi, *t_lb])
    ub = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 2 * np.pi, 2 * np.pi, 2 * np.pi, *t_ub])
    x = x_init.copy()
    cost = 0.0
    switched = 0
    nan_idx = ~np.isnan(sdf)
    # This is mathematically exp(2 * truncation), but evaluating it directly
    # overflows for large world-space meshes.  Clipping the log-domain value
    # preserves the original behavior in the normal range and keeps the
    # optimizer numerically defined for pathological scales.
    sigma2 = float(np.exp(np.clip(2.0 * float(truncation), -700.0, 700.0)))

    for it in range(para.max_iter):
        valid = _check_validity(x, t_lb, t_ub, truncation)
        if not valid.all():
            break

        sdf_current = sdf_superquadric(x, points, 0.0)
        active_idx = (
            (sdf_current < para.active_multiplier * truncation)
            & (sdf_current > -para.active_multiplier * truncation)
            & nan_idx
        )
        points_active = points[:, active_idx]
        sdf_active = sdf[active_idx]
        if sdf_active.size == 0:
            break
        weight = inlier_weight(sdf_active, active_idx, sdf_current, sigma2, para.w, truncation)

        r = eul2rotm(x[5:8])
        bp = bounding_points - x[8:11].reshape(3, 1)
        bp_body = r.T @ bp
        scale_limit = np.mean(np.abs(bp_body), axis=1)
        ub[2:5] = scale_limit
        x0, lb_use, ub_use = _project_x0_to_bounds(x, lb, ub)

        res = least_squares(
            lambda q: difference_sq_sdf(q, sdf_active, points_active, truncation, weight),
            x0,
            bounds=(lb_use, ub_use),
            method="trf",
            max_nfev=para.max_opti_iter,
        )
        x_n = res.x
        cost_n_full = float(np.sum(res.fun**2))
        sigma2_n = cost_n_full / max(np.sum(weight), 1e-12)
        cost_n = cost_n_full / float(len(sdf_active))
        rel = np.divide(
            abs(cost - cost_n),
            cost_n,
            out=np.array(np.nan, dtype=np.float64),
            where=(cost_n != 0),
        ).item()

        if (cost_n < para.tolerance and it > 0) or (
            rel < para.relative_tolerance and switched >= para.max_switch and it > para.iter_min
        ):
            x = x_n
            break

        if rel < para.switch_tolerance and it != 0 and switched < para.max_switch:
            switch_success = False

            axis0 = eul2rotm(x[5:8])
            axis1 = np.roll(axis0, 2, axis=1)
            axis2 = np.roll(axis0, 1, axis=1)
            eul1 = rotm2eul_zyx(axis1)
            eul2 = rotm2eul_zyx(axis2)
            x_axis = np.array(
                [
                    [x[1], x[0], x[3], x[4], x[2], *eul1, *x[8:11]],
                    [x[1], x[0], x[4], x[2], x[3], *eul2, *x[8:11]],
                ],
                dtype=np.float64,
            )

            scale_ratio = np.roll(x[2:5], 2) / x[2:5]
            scale_idx = np.where((scale_ratio > 0.8) & (scale_ratio < 1.2))[0] + 1
            x_rot = []
            if 1 in scale_idx:
                eul_rot = rotm2eul_zyx(axis0 @ rotz_deg(45.0))
                if x[1] <= 1:
                    a = ((1 - np.sqrt(2)) * x[1] + np.sqrt(2)) * min(x[2], x[3])
                else:
                    a = ((np.sqrt(2) / 2 - 1) * x[1] + 2 - np.sqrt(2) / 2) * min(x[2], x[3])
                x_rot.append(np.array([x[0], 2 - x[1], a, a, x[4], *eul_rot, *x[8:11]]))
            if 2 in scale_idx:
                eul_rot = rotm2eul_zyx(axis1 @ rotz_deg(45.0))
                if x[0] <= 1:
                    a = ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[3], x[4])
                else:
                    a = ((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[3], x[4])
                x_rot.append(np.array([x[1], 2 - x[0], a, a, x[2], *eul_rot, *x[8:11]]))
            if 3 in scale_idx:
                eul_rot = rotm2eul_zyx(axis2 @ rotz_deg(45.0))
                if x[0] <= 1:
                    a = ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[4], x[2])
                else:
                    a = ((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[4], x[2])
                x_rot.append(np.array([x[1], 2 - x[0], a, a, x[3], *eul_rot, *x[8:11]]))

            x_candidate = np.vstack([x_axis, *x_rot]) if len(x_rot) else x_axis
            costs = []
            valids = []
            for c in x_candidate:
                d = difference_sq_sdf(c, sdf_active, points_active, truncation, weight)
                v = float(np.sum(d**2))
                if np.isfinite(v):
                    costs.append(v)
                    valids.append(c)
            if valids:
                order = np.argsort(np.array(costs))
                for i in order:
                    c = valids[i]
                    r = eul2rotm(c[5:8])
                    bp = bounding_points - c[8:11].reshape(3, 1)
                    bp_body = r.T @ bp
                    ub[2:5] = np.mean(np.abs(bp_body), axis=1)
                    c0, lb_use, ub_use = _project_x0_to_bounds(c, lb, ub)
                    res_sw = least_squares(
                        lambda q: difference_sq_sdf(q, sdf_active, points_active, truncation, weight),
                        c0,
                        bounds=(lb_use, ub_use),
                        method="trf",
                        max_nfev=para.max_opti_iter,
                    )
                    c_sw = float(np.sum(res_sw.fun**2))
                    if c_sw / len(sdf_active) < min(cost_n, cost):
                        x = res_sw.x
                        cost = c_sw / len(sdf_active)
                        sigma2 = c_sw / max(np.sum(weight), 1e-12)
                        switch_success = True
                        break

            if not switch_success:
                x = x_n
                cost = cost_n
                sigma2 = sigma2_n
            switched += 1
        else:
            x = x_n
            cost = cost_n
            sigma2 = sigma2_n

    sdf_occ = sdf_superquadric(x, points, 0.0)
    occ = sdf_occ < para.nan_range
    occ_idx = roi_idx[occ]
    occ_in = sdf_occ <= 0
    num_idx = np.zeros(3, dtype=np.int64)
    num_idx[0] = np.sum((sdf[occ_in] <= 0) | np.isnan(sdf[occ_in]))
    num_idx[1] = np.sum(sdf[occ_in] > 0)
    num_idx[2] = np.sum(sdf[occ_in] <= 0)

    valid = _check_validity(x, t_lb, t_ub, truncation)
    return FitResult(x=x, occ_idx=occ_idx, valid=valid, num_idx=num_idx)
