"""RePart superquadric sampling, coordinate alignment, and containment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from marching_primitives.math_sq import eul2rotm, sdf_superquadric
from repart.geometry import clamp_eps


def ensure_mesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Scene):
        value = value.dump(concatenate=True)
    if not isinstance(value, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(value).__name__}")
    return value


def normalize_mesh(
    vertices: np.ndarray,
    half: float = 0.5,
    margin: float = 1e-6,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Apply the normalization used by HITOPS mesh mapping."""
    vertices = np.asarray(vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    if extent <= 0.0:
        raise ValueError(f"Invalid mesh extent: {extent}")
    scale = (half - margin) * 2.0 / extent
    return (vertices - center[None, :]) * scale, float(scale), center


def sample_sq_surface(x: np.ndarray, nu: int, nv: int) -> np.ndarray:
    """Sample one RePart/MPS (11,) SQ parameter vector."""
    x = np.asarray(x, dtype=np.float64).reshape(11)
    e1, e2 = [float(v) for v in clamp_eps(x[:2])]
    axes = np.maximum(np.asarray(x[2:5], dtype=np.float64), 1e-8)

    def signed_power(value: np.ndarray, exponent: float) -> np.ndarray:
        return np.sign(value) * np.abs(value) ** exponent

    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(nv), endpoint=True)
    omega = np.linspace(-np.pi, np.pi, int(nu), endpoint=False)
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")
    local = np.stack(
        [
            axes[0]
            * signed_power(np.cos(eta_grid), e1)
            * signed_power(np.cos(omega_grid), e2),
            axes[1]
            * signed_power(np.cos(eta_grid), e1)
            * signed_power(np.sin(omega_grid), e2),
            axes[2] * signed_power(np.sin(eta_grid), e1),
        ],
        axis=-1,
    ).reshape(-1, 3)
    rotation = eul2rotm(np.asarray(x[5:8], dtype=np.float64))
    return local @ rotation.T + np.asarray(x[8:11], dtype=np.float64)[None, :]


def _surface_grid_faces(nu: int, nv: int) -> np.ndarray:
    faces: List[List[int]] = []
    for row in range(int(nv) - 1):
        for col in range(int(nu)):
            next_col = (col + 1) % int(nu)
            v00 = row * int(nu) + col
            v01 = row * int(nu) + next_col
            v10 = (row + 1) * int(nu) + col
            v11 = (row + 1) * int(nu) + next_col
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return np.asarray(faces, dtype=np.int64)


def alignment_to_raw(
    points: np.ndarray,
    alignment: Optional[Dict[str, Any]],
    bootstrap_source: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the inverse transform recorded by RePart supervision."""
    points = np.asarray(points, dtype=np.float64)
    if alignment is None:
        if str(bootstrap_source).lower() == "mps":
            raise ValueError(
                "MPS grouped SQs require bootstrap.sq_label_alignment; "
                "refusing to guess a coordinate transform."
            )
        return points.copy(), {
            "method": "implicit_identity",
            "bootstrap_source": str(bootstrap_source),
        }

    method = str(alignment.get("method", ""))
    if method == "identity":
        return points.copy(), dict(alignment)

    raw_bounds = np.asarray(alignment["raw_bounds"], dtype=np.float64).reshape(2, 3)
    sdf_bounds = np.asarray(
        alignment["sdf_surface_bounds"], dtype=np.float64
    ).reshape(2, 3)
    raw_center = raw_bounds.mean(axis=0)
    sdf_center = sdf_bounds.mean(axis=0)

    if method == "inverse_isotropic_sdf_zero_level_bounds":
        scale = float(alignment["scale_raw_to_sdf"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid alignment scale: {scale}")
        return (
            (points - sdf_center[None, :]) / scale + raw_center[None, :],
            dict(alignment),
        )

    if method == "anisotropic_sdf_zero_level_bounds_to_raw_bounds":
        scales = np.asarray(
            alignment["axis_scales_sdf_to_raw"], dtype=np.float64
        ).reshape(3)
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError(f"Invalid anisotropic scales: {scales.tolist()}")
        return (
            (points - sdf_center[None, :]) * scales[None, :]
            + raw_center[None, :],
            dict(alignment),
        )

    raise ValueError(f"Unsupported sq_label_alignment method: {method!r}")


def raw_to_sq_frame(
    points: np.ndarray,
    alignment: Optional[Dict[str, Any]],
    bootstrap_source: str,
    *,
    sq_frame: str,
    mesh_scale: float,
    mesh_center: np.ndarray,
) -> np.ndarray:
    """Transform raw-mesh points into the SQ parameter frame."""
    points = np.asarray(points, dtype=np.float64)
    if sq_frame == "raw":
        return points.copy()
    if sq_frame == "normalized":
        return (points - mesh_center[None, :]) * float(mesh_scale)
    if sq_frame != "auto":
        raise ValueError(f"Unknown sq_frame: {sq_frame}")

    if alignment is None:
        if str(bootstrap_source).lower() == "mps":
            raise ValueError(
                "MPS grouped SQs require bootstrap.sq_label_alignment; "
                "refusing to guess a coordinate transform."
            )
        return points.copy()

    method = str(alignment.get("method", ""))
    if method == "identity":
        return points.copy()

    raw_bounds = np.asarray(alignment["raw_bounds"], dtype=np.float64).reshape(2, 3)
    sdf_bounds = np.asarray(
        alignment["sdf_surface_bounds"], dtype=np.float64
    ).reshape(2, 3)
    raw_center = raw_bounds.mean(axis=0)
    sdf_center = sdf_bounds.mean(axis=0)
    if method == "inverse_isotropic_sdf_zero_level_bounds":
        scale = float(alignment["scale_raw_to_sdf"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid alignment scale: {scale}")
        return (points - raw_center[None, :]) * scale + sdf_center[None, :]
    if method == "anisotropic_sdf_zero_level_bounds_to_raw_bounds":
        scales = np.asarray(
            alignment["axis_scales_sdf_to_raw"], dtype=np.float64
        ).reshape(3)
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError(f"Invalid anisotropic scales: {scales.tolist()}")
        return (
            (points - raw_center[None, :]) / scales[None, :]
            + sdf_center[None, :]
        )
    raise ValueError(f"Unsupported sq_label_alignment method: {method!r}")


def compute_face_group_containment_sdf(
    mesh: trimesh.Trimesh,
    xs: np.ndarray,
    group_ids_per_primitive: Sequence[int],
    group_ids: Sequence[int],
    grouped_payload: Dict[str, Any],
    *,
    sq_frame: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return minimum analytic SQ SDF for each mesh face and RL group."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    raw_centers = np.asarray(mesh.vertices, dtype=np.float64)[faces].mean(axis=1)
    _, mesh_scale, mesh_center = normalize_mesh(mesh.vertices)
    bootstrap = grouped_payload.get("bootstrap", {})
    bootstrap_source = str(bootstrap.get("bootstrap_source", ""))
    sq_centers = raw_to_sq_frame(
        raw_centers,
        bootstrap.get("sq_label_alignment"),
        bootstrap_source,
        sq_frame=sq_frame,
        mesh_scale=mesh_scale,
        mesh_center=mesh_center,
    )

    group_to_index = {int(gid): index for index, gid in enumerate(group_ids)}
    group_sdf = np.full(
        (len(faces), len(group_ids)), np.inf, dtype=np.float32
    )
    for x, gid in zip(np.asarray(xs, dtype=np.float64), group_ids_per_primitive):
        group_index = group_to_index[int(gid)]
        sdf = sdf_superquadric(x, sq_centers.T, truncation=0)
        group_sdf[:, group_index] = np.minimum(
            group_sdf[:, group_index], sdf.astype(np.float32)
        )

    inside = group_sdf <= 0.0
    return group_sdf, {
        "enabled": True,
        "criterion": "analytic_sq_sdf_le_margin",
        "faces_inside_any_group_at_zero_margin": int(
            np.count_nonzero(np.any(inside, axis=1))
        ),
        "groups_with_contained_faces": int(
            np.count_nonzero(np.any(inside, axis=0))
        ),
    }


def _unique_in_order(values: Sequence[int]) -> List[int]:
    output: List[int] = []
    seen = set()
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def build_group_point_clouds(
    xs: np.ndarray,
    group_ids: Sequence[int],
    grouped_payload: Dict[str, Any],
    mesh_vertices: np.ndarray,
    *,
    sq_frame: str,
    surface_nu: int,
    surface_nv: int,
    max_points_per_group: int,
) -> Tuple[List[int], List[np.ndarray], Dict[str, Any]]:
    """Build one HITOPS-normalized surface cloud per RePart group."""
    xs = np.asarray(xs, dtype=np.float64)
    if xs.ndim != 2 or xs.shape[1] != 11:
        raise ValueError(f"Expected grouped SQ shape (N, 11), got {xs.shape}")
    if len(group_ids) != len(xs):
        raise ValueError(
            f"group_ids length {len(group_ids)} does not match SQ count {len(xs)}"
        )

    group_order = _unique_in_order(group_ids)
    by_group: Dict[int, List[np.ndarray]] = {gid: [] for gid in group_order}
    bootstrap = grouped_payload.get("bootstrap", {})
    bootstrap_source = str(bootstrap.get("bootstrap_source", ""))
    alignment = bootstrap.get("sq_label_alignment")
    _, mesh_scale, mesh_center = normalize_mesh(mesh_vertices)
    resolved_alignment: Dict[str, Any] = {}

    for x, gid in zip(xs, group_ids):
        sampled = sample_sq_surface(x, surface_nu, surface_nv)
        if sq_frame == "normalized":
            normalized = sampled
            resolved_alignment = {
                "method": "input_already_hitops_normalized",
                "bootstrap_source": bootstrap_source,
            }
        else:
            if sq_frame == "raw":
                raw = sampled
                resolved_alignment = {
                    "method": "user_forced_raw",
                    "bootstrap_source": bootstrap_source,
                }
            elif sq_frame == "auto":
                raw, resolved_alignment = alignment_to_raw(
                    sampled, alignment, bootstrap_source
                )
            else:
                raise ValueError(f"Unknown sq_frame: {sq_frame}")
            normalized = (raw - mesh_center[None, :]) * mesh_scale
        by_group[int(gid)].append(normalized.astype(np.float32))

    rng = np.random.RandomState(0)
    clouds: List[np.ndarray] = []
    sizes_before: List[int] = []
    for gid in group_order:
        cloud = np.concatenate(by_group[gid], axis=0)
        sizes_before.append(int(len(cloud)))
        if len(cloud) > int(max_points_per_group):
            choice = rng.choice(
                len(cloud), int(max_points_per_group), replace=False
            )
            cloud = cloud[choice]
        clouds.append(np.asarray(cloud, dtype=np.float32))

    return group_order, clouds, {
        "sq_frame_requested": sq_frame,
        "alignment": resolved_alignment,
        "hitops_mesh_normalization": {
            "center": mesh_center.tolist(),
            "scale": float(mesh_scale),
        },
        "surface_nu": int(surface_nu),
        "surface_nv": int(surface_nv),
        "max_points_per_group": int(max_points_per_group),
        "cloud_sizes_before_subsample": sizes_before,
        "cloud_sizes": [int(len(cloud)) for cloud in clouds],
    }


def export_aligned_grouped_sq_mesh(
    xs: np.ndarray,
    group_ids: Sequence[int],
    grouped_payload: Dict[str, Any],
    mesh_vertices: np.ndarray,
    output_path: Path,
    *,
    sq_frame: str,
    color_fn,
    nu: int = 40,
    nv: int = 24,
) -> str:
    """Export grouped SQ surfaces in the raw mesh frame for debugging."""
    bootstrap = grouped_payload.get("bootstrap", {})
    bootstrap_source = str(bootstrap.get("bootstrap_source", ""))
    alignment = bootstrap.get("sq_label_alignment")
    _, mesh_scale, mesh_center = normalize_mesh(mesh_vertices)
    faces = _surface_grid_faces(nu, nv)
    meshes: List[trimesh.Trimesh] = []
    for x, group_id in zip(np.asarray(xs), group_ids):
        points = sample_sq_surface(x, nu, nv)
        if sq_frame == "normalized":
            raw_points = points / mesh_scale + mesh_center[None, :]
        elif sq_frame == "raw":
            raw_points = points
        elif sq_frame == "auto":
            raw_points, _ = alignment_to_raw(
                points, alignment, bootstrap_source
            )
        else:
            raise ValueError(f"Unknown sq_frame: {sq_frame}")
        color = color_fn(int(group_id))
        surface = trimesh.Trimesh(
            vertices=raw_points, faces=faces, process=False
        )
        surface.visual.face_colors = np.tile(color[None, :], (len(faces), 1))
        surface.visual.vertex_colors = np.tile(
            color[None, :], (len(raw_points), 1)
        )
        meshes.append(surface)

    if not meshes:
        raise ValueError("Cannot export an empty grouped SQ mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    merged.export(output_path)
    return str(output_path)
