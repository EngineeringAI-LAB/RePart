from __future__ import annotations

import numpy as np


def eul2rotm(eul: np.ndarray) -> np.ndarray:
    """Match MATLAB eul2rotm implementation in MPS.m."""
    eul = np.asarray(eul, dtype=np.float64).reshape(1, 3)
    ct = np.cos(eul)
    st = np.sin(eul)
    r = np.zeros((3, 3), dtype=np.float64)
    r[0, 0] = ct[0, 1] * ct[0, 0]
    r[0, 1] = st[0, 2] * st[0, 1] * ct[0, 0] - ct[0, 2] * st[0, 0]
    r[0, 2] = ct[0, 2] * st[0, 1] * ct[0, 0] + st[0, 2] * st[0, 0]
    r[1, 0] = ct[0, 1] * st[0, 0]
    r[1, 1] = st[0, 2] * st[0, 1] * st[0, 0] + ct[0, 2] * ct[0, 0]
    r[1, 2] = ct[0, 2] * st[0, 1] * st[0, 0] - st[0, 2] * ct[0, 0]
    r[2, 0] = -st[0, 1]
    r[2, 1] = st[0, 2] * ct[0, 1]
    r[2, 2] = ct[0, 2] * ct[0, 1]
    return r


def rotm2eul_zyx(rotm: np.ndarray) -> np.ndarray:
    """Inverse of eul2rotm convention used by the MATLAB code."""
    r = np.asarray(rotm, dtype=np.float64)
    sy = -r[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    y = np.arcsin(sy)
    cy = np.cos(y)
    if abs(cy) > 1e-8:
        x = np.arctan2(r[1, 0], r[0, 0])
        z = np.arctan2(r[2, 1], r[2, 2])
    else:
        x = np.arctan2(-r[0, 1], r[1, 1])
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def rotz_deg(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def sdf_superquadric(params: np.ndarray, points: np.ndarray, truncation: float) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    r = eul2rotm(params[5:8])
    t = params[8:11]
    x = r.T @ points - (r.T @ t.reshape(3, 1))
    r0 = np.linalg.norm(x, axis=0)

    eps1, eps2 = params[0], params[1]
    a1, a2, a3 = params[2], params[3], params[4]
    scale = (
        ((x[0] / a1) ** 2) ** (1.0 / eps2) + ((x[1] / a2) ** 2) ** (1.0 / eps2)
    ) ** (eps2 / eps1) + ((x[2] / a3) ** 2) ** (1.0 / eps1)
    # Avoid 0 raised to a negative power when a query point lies exactly at
    # the superquadric center (or the intermediate value underflows to zero).
    scale = np.maximum(scale, 1e-12)
    scale = scale ** (-eps1 / 2.0)
    sdf = r0 * (1.0 - scale)
    if truncation != 0:
        sdf = np.clip(sdf, -truncation, truncation)
    return sdf


def sdf_multi_superquadrics(params: np.ndarray, points: np.ndarray, truncation: float) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    out = sdf_superquadric(params[0], points, truncation)
    for i in range(1, params.shape[0]):
        out = np.minimum(out, sdf_superquadric(params[i], points, truncation))
    return out
