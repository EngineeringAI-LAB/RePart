from __future__ import annotations

import numpy as np

from .math_sq import eul2rotm


def _spow(v: np.ndarray, eps: float) -> np.ndarray:
    return np.sign(v) * (np.abs(v) ** eps)


def single_superquadric_mesh(
    x: np.ndarray, n_eta: int = 48, n_omega: int = 96
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a triangle mesh for one superquadric parameter row:
    [eps1, eps2, ax, ay, az, eul_z, eul_y, eul_x, tx, ty, tz]
    """
    p = np.asarray(x, dtype=np.float64).reshape(-1)
    eps1, eps2 = p[0], p[1]
    a1, a2, a3 = p[2], p[3], p[4]
    r = eul2rotm(p[5:8])
    t = p[8:11]

    eta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_eta)
    omega = np.linspace(-np.pi, np.pi, n_omega, endpoint=False)
    ee, oo = np.meshgrid(eta, omega, indexing="ij")

    ce = _spow(np.cos(ee), eps1)
    se = _spow(np.sin(ee), eps1)
    co = _spow(np.cos(oo), eps2)
    so = _spow(np.sin(oo), eps2)

    x_l = a1 * ce * co
    y_l = a2 * ce * so
    z_l = a3 * se
    v_local = np.stack([x_l, y_l, z_l], axis=-1).reshape(-1, 3)
    v_world = (r @ v_local.T).T + t.reshape(1, 3)

    faces: list[list[int]] = []
    for i in range(n_eta - 1):
        for j in range(n_omega):
            jn = (j + 1) % n_omega
            v00 = i * n_omega + j
            v01 = i * n_omega + jn
            v10 = (i + 1) * n_omega + j
            v11 = (i + 1) * n_omega + jn
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    f = np.asarray(faces, dtype=np.int64)
    return v_world, f


def merged_superquadrics_mesh(
    xs: np.ndarray, n_eta: int = 48, n_omega: int = 96
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    v_off = 0
    for i in range(xs.shape[0]):
        v, f = single_superquadric_mesh(xs[i], n_eta=n_eta, n_omega=n_omega)
        all_v.append(v)
        all_f.append(f + v_off)
        v_off += v.shape[0]
    return np.vstack(all_v), np.vstack(all_f)
