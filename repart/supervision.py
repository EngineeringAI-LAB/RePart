"""SQ-to-GT-part labeling by asymmetric AABB volume coverage.

The primary score is:

    volume(SQ_AABB intersection GT_PART_AABB) / volume(SQ_AABB)

This module contains the production-facing implementation used by RL training.
It intentionally excludes experiment export, visualization, and bidirectional
surface-IoU code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

from .geometry import primitive_to_surface_mesh


SQ_LABEL_CACHE_SCHEMA_VERSION = 1
SQ_LABEL_ALGORITHM_VERSION = "bbox_sq_coverage_v2"


class SQLabelCoverageError(RuntimeError):
    """Raised when coverage labels cannot be computed safely."""


@dataclass
class SQCoverageLabelResult:
    labels: np.ndarray
    coverage: np.ndarray
    secondary_labels: np.ndarray
    secondary_coverage: np.ndarray
    coverage_margin: np.ndarray
    coverage_entropy: np.ndarray
    positive_coverage_fraction: np.ndarray
    part_labels: np.ndarray
    coverage_matrix: np.ndarray
    coverage_probabilities: np.ndarray
    orphan_mask: np.ndarray
    near_counts: np.ndarray
    sq_bounds: np.ndarray
    part_bounds: Dict[int, np.ndarray]
    part_bounds_source: str
    part_bounds_error: Optional[str]
    alignment: Dict[str, Any]
    algorithm_config: Dict[str, Any]
    cache_key: str = ""


def aabb_intersection_over_first_volume(
    bounds_a: np.ndarray,
    bounds_b: np.ndarray,
) -> float:
    """Return volume(AABB_a intersection AABB_b) / volume(AABB_a)."""
    bounds_a = np.asarray(bounds_a, dtype=np.float64).reshape(2, 3)
    bounds_b = np.asarray(bounds_b, dtype=np.float64).reshape(2, 3)
    intersection_extent = np.maximum(
        np.minimum(bounds_a[1], bounds_b[1]) - np.maximum(bounds_a[0], bounds_b[0]),
        0.0,
    )
    intersection_volume = float(np.prod(intersection_extent))
    volume_a = float(np.prod(np.maximum(bounds_a[1] - bounds_a[0], 0.0)))
    return float(intersection_volume / volume_a) if volume_a > 1e-12 else 0.0


def aabb_iou(bounds_a: np.ndarray, bounds_b: np.ndarray) -> float:
    """Return volumetric IoU of two AABBs."""
    bounds_a = np.asarray(bounds_a, dtype=np.float64).reshape(2, 3)
    bounds_b = np.asarray(bounds_b, dtype=np.float64).reshape(2, 3)
    intersection_extent = np.maximum(
        np.minimum(bounds_a[1], bounds_b[1]) - np.maximum(bounds_a[0], bounds_b[0]),
        0.0,
    )
    intersection_volume = float(np.prod(intersection_extent))
    volume_a = float(np.prod(np.maximum(bounds_a[1] - bounds_a[0], 0.0)))
    volume_b = float(np.prod(np.maximum(bounds_b[1] - bounds_b[0], 0.0)))
    union_volume = volume_a + volume_b - intersection_volume
    return float(intersection_volume / union_volume) if union_volume > 1e-12 else 0.0


def _mesh_bounds(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            mesh
            for mesh in loaded.geometry.values()
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0
        ]
        if not meshes:
            raise SQLabelCoverageError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise SQLabelCoverageError(f"No mesh geometry in {path}")
    return np.asarray(loaded.bounds, dtype=np.float64)


def _collect_label_objects(
    result_payload: Any,
    valid_labels: Iterable[int],
) -> Dict[int, List[str]]:
    valid = {int(label) for label in valid_labels}
    found: Dict[int, List[str]] = {}

    def visit(node: Mapping[str, Any]) -> None:
        label = int(node.get("id", -999999))
        objects = [str(value) for value in (node.get("objs", []) or [])]
        if label in valid and objects:
            found[label] = objects
        for child in node.get("children", []) or []:
            if isinstance(child, Mapping):
                visit(child)

    roots = result_payload if isinstance(result_payload, list) else [result_payload]
    for root in roots:
        if isinstance(root, Mapping):
            visit(root)
    return found


def part_aabbs_from_labeled_points(
    points: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Build GT-part AABBs from the point-level labels already loaded by RL."""
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(points) != len(labels):
        raise SQLabelCoverageError(
            f"Point/label count mismatch: {len(points)} vs {len(labels)}"
        )
    result: Dict[int, np.ndarray] = {}
    for label in np.unique(labels[labels >= 0]):
        part_points = points[labels == int(label)]
        if len(part_points) == 0:
            continue
        result[int(label)] = np.asarray(
            [part_points.min(axis=0), part_points.max(axis=0)],
            dtype=np.float64,
        )
    if not result:
        raise SQLabelCoverageError("No non-negative GT part labels")
    return result


def resolve_part_aabbs(
    points: np.ndarray,
    labels: np.ndarray,
    sample_dir: Optional[Path],
) -> Tuple[Dict[int, np.ndarray], str, Optional[str]]:
    """Prefer PartNet object-mesh bounds, with labeled-point fallback."""
    point_bounds = part_aabbs_from_labeled_points(points, labels)
    if sample_dir is None:
        return point_bounds, "labeled_points", "sample_dir_not_provided"

    sample_dir = Path(sample_dir)
    result_candidates = (
        sample_dir / "result.json",
        sample_dir / "result_after_merging.json",
    )
    result_path = next((path for path in result_candidates if path.is_file()), None)
    object_dir = sample_dir / "objs"
    if result_path is None or not object_dir.is_dir():
        return (
            point_bounds,
            "labeled_points",
            "missing_result_json_or_objs_directory",
        )

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        objects_by_label = _collect_label_objects(payload, point_bounds)
        mesh_bounds: Dict[int, np.ndarray] = {}
        used_point_fallback = False
        for label in sorted(point_bounds):
            object_ids = objects_by_label.get(label, [])
            bounds_for_label: List[np.ndarray] = []
            for object_id in object_ids:
                path = object_dir / f"{object_id}.obj"
                if path.is_file():
                    bounds_for_label.append(_mesh_bounds(path))
            if bounds_for_label:
                mesh_bounds[label] = np.asarray(
                    [
                        np.min([bounds[0] for bounds in bounds_for_label], axis=0),
                        np.max([bounds[1] for bounds in bounds_for_label], axis=0),
                    ],
                    dtype=np.float64,
                )
            else:
                mesh_bounds[label] = point_bounds[label]
                used_point_fallback = True
        source = (
            "partnet_object_meshes_with_point_fallback"
            if used_point_fallback
            else "partnet_object_meshes"
        )
        error = (
            "one_or_more_gt_labels_missing_object_meshes"
            if used_point_fallback
            else None
        )
        return mesh_bounds, source, error
    except Exception as exc:
        return (
            point_bounds,
            "labeled_points_after_mesh_error",
            f"{type(exc).__name__}: {exc}",
        )


def _sdf_zero_level_bounds(
    sdf_values: np.ndarray,
    sdf_grid: Any,
) -> np.ndarray:
    if sdf_values is None or sdf_grid is None:
        raise SQLabelCoverageError(
            "MPS coverage labeling requires the loaded SDF and grid"
        )
    grid_size = tuple(int(value) for value in sdf_grid.size)
    sdf_array = np.asarray(sdf_values, dtype=np.float64)
    if sdf_array.shape != grid_size:
        sdf_array = np.reshape(sdf_array, grid_size, order="F")
    vertices, _, _, _ = marching_cubes(
        sdf_array,
        level=0.0,
        spacing=(
            float(sdf_grid.x[1] - sdf_grid.x[0]),
            float(sdf_grid.y[1] - sdf_grid.y[0]),
            float(sdf_grid.z[1] - sdf_grid.z[0]),
        ),
        allow_degenerate=False,
    )
    vertices += np.asarray(
        [sdf_grid.x[0], sdf_grid.y[0], sdf_grid.z[0]],
        dtype=np.float64,
    )
    return np.asarray(
        [vertices.min(axis=0), vertices.max(axis=0)],
        dtype=np.float64,
    )


def _vertex_alignment(
    bootstrap_source: str,
    raw_bounds: np.ndarray,
    sdf_values: Optional[np.ndarray],
    sdf_grid: Any,
) -> Tuple[Any, Dict[str, Any]]:
    source = str(bootstrap_source).lower()
    raw_bounds = np.asarray(raw_bounds, dtype=np.float64).reshape(2, 3)
    if source != "mps":
        return (
            lambda vertices: np.asarray(vertices, dtype=np.float64),
            {
                "method": "identity",
                "bootstrap_source": source,
                "raw_bounds": raw_bounds.tolist(),
            },
        )

    sdf_bounds = _sdf_zero_level_bounds(sdf_values, sdf_grid)
    raw_extent = np.maximum(raw_bounds[1] - raw_bounds[0], 1e-12)
    sdf_extent = np.maximum(sdf_bounds[1] - sdf_bounds[0], 1e-12)
    raw_to_sdf_axis_scales = sdf_extent / raw_extent
    isotropic_scale = float(np.median(raw_to_sdf_axis_scales))
    relative_spread = float(
        np.max(np.abs(raw_to_sdf_axis_scales / max(isotropic_scale, 1e-12) - 1.0))
    )
    raw_center = raw_bounds.mean(axis=0)
    sdf_center = sdf_bounds.mean(axis=0)

    if relative_spread <= 0.03:

        def transform(vertices: np.ndarray) -> np.ndarray:
            vertices = np.asarray(vertices, dtype=np.float64)
            return (vertices - sdf_center[None, :]) / isotropic_scale + raw_center[
                None, :
            ]

        alignment = {
            "method": "inverse_isotropic_sdf_zero_level_bounds",
            "bootstrap_source": source,
            "scale_raw_to_sdf": isotropic_scale,
            "axis_scales_raw_to_sdf": raw_to_sdf_axis_scales.tolist(),
            "axis_scale_relative_spread": relative_spread,
            "raw_bounds": raw_bounds.tolist(),
            "sdf_surface_bounds": sdf_bounds.tolist(),
        }
        return transform, alignment

    sdf_to_raw_axis_scales = raw_extent / sdf_extent

    def transform(vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        return (vertices - sdf_center[None, :]) * sdf_to_raw_axis_scales[
            None, :
        ] + raw_center[None, :]

    alignment = {
        "method": "anisotropic_sdf_zero_level_bounds_to_raw_bounds",
        "bootstrap_source": source,
        "axis_scales_sdf_to_raw": sdf_to_raw_axis_scales.tolist(),
        "axis_scale_relative_spread": relative_spread,
        "raw_bounds": raw_bounds.tolist(),
        "sdf_surface_bounds": sdf_bounds.tolist(),
    }
    return transform, alignment


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _area_uniform_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 1e-14)
    if not np.any(valid):
        raise SQLabelCoverageError("SQ mesh has no positive-area triangles")
    valid_indices = np.flatnonzero(valid)
    probabilities = areas[valid] / areas[valid].sum()
    selected = rng.choice(
        valid_indices,
        size=int(count),
        replace=True,
        p=probabilities,
    )
    selected_triangles = triangles[selected]
    r1 = np.sqrt(rng.random(int(count)))
    r2 = rng.random(int(count))
    return (
        selected_triangles[:, 0] * (1.0 - r1)[:, None]
        + selected_triangles[:, 1] * (r1 * (1.0 - r2))[:, None]
        + selected_triangles[:, 2] * (r1 * r2)[:, None]
    )


def _estimate_point_spacing(
    points: np.ndarray,
    tree: cKDTree,
    seed: int,
) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    count = min(len(points), 4096)
    indices = (
        rng.choice(len(points), size=count, replace=False)
        if count < len(points)
        else np.arange(len(points))
    )
    distances, _ = tree.query(points[indices], k=2, workers=-1)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    return float(np.median(nearest)) if len(nearest) else 0.0


def _update_file_signature(digest: Any, path: Path) -> None:
    """Add a cheap, deterministic file identity to a cache fingerprint."""
    path = Path(path)
    digest.update(str(path).encode("utf-8"))
    if not path.is_file():
        digest.update(b"<missing>")
        return
    stat = path.stat()
    digest.update(str(int(stat.st_size)).encode("ascii"))
    digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))


def _update_part_mesh_signature(
    digest: Any,
    sample_dir: Optional[Path],
    valid_labels: Iterable[int],
) -> None:
    if sample_dir is None:
        digest.update(b"<no-sample-dir>")
        return
    sample_dir = Path(sample_dir)
    result_candidates = (
        sample_dir / "result.json",
        sample_dir / "result_after_merging.json",
    )
    result_path = next((path for path in result_candidates if path.is_file()), None)
    if result_path is None:
        digest.update(b"<no-result-json>")
        return
    _update_file_signature(digest, result_path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        objects_by_label = _collect_label_objects(payload, valid_labels)
        for label in sorted(objects_by_label):
            digest.update(str(int(label)).encode("ascii"))
            for object_id in sorted(objects_by_label[label]):
                _update_file_signature(
                    digest,
                    sample_dir / "objs" / f"{object_id}.obj",
                )
    except Exception as exc:
        digest.update(f"<result-parse-error:{type(exc).__name__}>".encode("utf-8"))


def coverage_cache_key(
    primitives: Sequence[Any],
    points: np.ndarray,
    point_labels: np.ndarray,
    sample_id: str,
    bootstrap_source: str,
    *,
    sample_dir: Optional[Path] = None,
    sdf_csv_path: Optional[str] = None,
    surface_samples: int = 200,
    seed: int = 42,
    scene_distance_ratio: float = 0.065,
    sq_min_axis_ratio: float = 0.35,
    spacing_multiplier: float = 4.0,
    min_near_points: int = 3,
) -> str:
    """Fingerprint geometry, GT inputs, and all label-algorithm parameters."""
    digest = hashlib.sha256()
    digest.update(SQ_LABEL_ALGORITHM_VERSION.encode("utf-8"))
    digest.update(str(sample_id).encode("utf-8"))
    digest.update(str(bootstrap_source).encode("utf-8"))
    algorithm_config = {
        "surface_samples": int(surface_samples),
        "seed": int(seed),
        "scene_distance_ratio": float(scene_distance_ratio),
        "sq_min_axis_ratio": float(sq_min_axis_ratio),
        "spacing_multiplier": float(spacing_multiplier),
        "min_near_points": int(min_near_points),
    }
    digest.update(
        json.dumps(
            algorithm_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for primitive in primitives:
        values = np.concatenate(
            [
                np.asarray(primitive.raw_e, dtype=np.float64).reshape(-1),
                np.asarray(primitive.log_a, dtype=np.float64).reshape(-1),
                np.asarray(primitive.r, dtype=np.float64).reshape(-1),
                np.asarray(primitive.t, dtype=np.float64).reshape(-1),
            ]
        )
        digest.update(values.tobytes())
    digest.update(np.asarray(points, dtype=np.float32).tobytes())
    labels = np.asarray(point_labels, dtype=np.int64)
    digest.update(labels.tobytes())
    _update_part_mesh_signature(
        digest,
        sample_dir,
        np.unique(labels[labels >= 0]),
    )
    if sdf_csv_path:
        _update_file_signature(digest, Path(sdf_csv_path))
    return digest.hexdigest()


def assign_sq_labels_by_coverage(
    primitives: Sequence[Any],
    points: np.ndarray,
    point_labels: np.ndarray,
    *,
    sample_id: str,
    bootstrap_source: str,
    sdf_values: Optional[np.ndarray] = None,
    sdf_grid: Any = None,
    sample_dir: Optional[Path] = None,
    surface_samples: int = 200,
    seed: int = 42,
    scene_distance_ratio: float = 0.065,
    sq_min_axis_ratio: float = 0.35,
    spacing_multiplier: float = 4.0,
    min_near_points: int = 3,
) -> SQCoverageLabelResult:
    """Assign one GT part label to each SQ using asymmetric AABB coverage."""
    points = np.asarray(points, dtype=np.float64)
    point_labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    if len(points) != len(point_labels):
        raise SQLabelCoverageError(
            f"Point/label count mismatch: {len(points)} vs {len(point_labels)}"
        )
    valid_mask = point_labels >= 0
    if not np.any(valid_mask):
        raise SQLabelCoverageError("Coverage labeling requires GT point labels")
    valid_points = points[valid_mask]

    algorithm_config = {
        "algorithm_version": SQ_LABEL_ALGORITHM_VERSION,
        "surface_samples": int(surface_samples),
        "seed": int(seed),
        "scene_distance_ratio": float(scene_distance_ratio),
        "sq_min_axis_ratio": float(sq_min_axis_ratio),
        "spacing_multiplier": float(spacing_multiplier),
        "min_near_points": int(min_near_points),
    }
    part_bounds, part_bounds_source, part_bounds_error = resolve_part_aabbs(
        points,
        point_labels,
        sample_dir,
    )
    part_label_ids = np.asarray(sorted(part_bounds), dtype=np.int64)
    part_label_to_column = {
        int(label): int(column) for column, label in enumerate(part_label_ids)
    }
    coverage_matrix = np.zeros(
        (len(primitives), len(part_label_ids)),
        dtype=np.float64,
    )
    raw_bounds = np.asarray(
        [
            np.min([bounds[0] for bounds in part_bounds.values()], axis=0),
            np.max([bounds[1] for bounds in part_bounds.values()], axis=0),
        ],
        dtype=np.float64,
    )
    transform_vertices, alignment = _vertex_alignment(
        bootstrap_source,
        raw_bounds,
        sdf_values,
        sdf_grid,
    )

    gt_tree = cKDTree(valid_points)
    spacing = _estimate_point_spacing(
        valid_points,
        gt_tree,
        _stable_seed(seed, sample_id, "point_spacing"),
    )
    scene_diagonal = float(np.linalg.norm(raw_bounds[1] - raw_bounds[0]))

    labels: List[int] = []
    coverage_values: List[float] = []
    secondary_labels: List[int] = []
    secondary_coverage: List[float] = []
    orphan_mask: List[bool] = []
    near_counts: List[int] = []
    sq_bounds_all: List[np.ndarray] = []

    for primitive_id, primitive in enumerate(primitives):
        vertices, faces = primitive_to_surface_mesh(primitive, nu=40, nv=24)
        aligned_vertices = transform_vertices(vertices)
        sq_bounds = np.asarray(
            [aligned_vertices.min(axis=0), aligned_vertices.max(axis=0)],
            dtype=np.float64,
        )
        sq_bounds_all.append(sq_bounds)

        rng = np.random.default_rng(_stable_seed(seed, sample_id, "sq", primitive_id))
        sampled_surface = _area_uniform_samples(
            aligned_vertices,
            faces,
            surface_samples,
            rng,
        )
        distances, nearest_indices = gt_tree.query(
            sampled_surface,
            k=1,
            workers=-1,
        )
        sq_extent = np.maximum(sq_bounds[1] - sq_bounds[0], 0.0)
        min_semi_axis = 0.5 * float(np.min(sq_extent))
        distance_threshold = max(
            float(scene_distance_ratio) * scene_diagonal,
            float(sq_min_axis_ratio) * min_semi_axis,
            float(spacing_multiplier) * spacing,
        )
        near_mask = distances <= distance_threshold
        near_count = int(np.sum(near_mask))
        is_orphan = near_count < int(min_near_points)

        ranked: List[Tuple[float, float, int, int]] = []
        for candidate, bounds in sorted(part_bounds.items()):
            coverage = aabb_intersection_over_first_volume(sq_bounds, bounds)
            coverage_matrix[
                primitive_id,
                part_label_to_column[int(candidate)],
            ] = float(coverage)
            if coverage <= 0.0:
                continue
            inside = np.all(
                (sampled_surface >= bounds[0][None, :])
                & (sampled_surface <= bounds[1][None, :]),
                axis=1,
            )
            ranked.append(
                (
                    float(coverage),
                    float(aabb_iou(sq_bounds, bounds)),
                    int(np.sum(inside)),
                    int(candidate),
                )
            )
        ranked.sort(
            key=lambda item: (item[0], item[1], item[2], -item[3]),
            reverse=True,
        )

        if is_orphan:
            label = -1
            top_coverage = 0.0
            second_label = -1
            second_score = 0.0
        elif ranked:
            top_coverage, _, _, label = ranked[0]
            if len(ranked) >= 2:
                second_score, _, _, second_label = ranked[1]
            else:
                second_label = -1
                second_score = 0.0
        else:
            near_indices = nearest_indices[near_mask]
            near_distances = distances[near_mask]
            near_labels = point_labels[np.flatnonzero(valid_mask)[near_indices]]
            weights = np.exp(
                -np.square(near_distances / max(distance_threshold, 1e-12))
            )
            fallback_scores = [
                (
                    float(weights[near_labels == candidate].sum()),
                    int(candidate),
                )
                for candidate in np.unique(near_labels)
            ]
            fallback_scores.sort(
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            label = int(fallback_scores[0][1])
            top_coverage = 0.0
            if len(fallback_scores) >= 2:
                second_score = 0.0
                second_label = int(fallback_scores[1][1])
            else:
                second_label = -1
                second_score = 0.0

        labels.append(int(label))
        coverage_values.append(float(top_coverage))
        secondary_labels.append(int(second_label))
        secondary_coverage.append(float(second_score))
        orphan_mask.append(bool(is_orphan))
        near_counts.append(near_count)

    labels_array = np.asarray(labels, dtype=np.int64)
    coverage_array = np.asarray(coverage_values, dtype=np.float64)
    secondary_labels_array = np.asarray(secondary_labels, dtype=np.int64)
    secondary_coverage_array = np.asarray(
        secondary_coverage,
        dtype=np.float64,
    )
    orphan_array = np.asarray(orphan_mask, dtype=bool)
    coverage_sums = coverage_matrix.sum(axis=1, keepdims=True)
    coverage_probabilities = np.divide(
        coverage_matrix,
        coverage_sums,
        out=np.zeros_like(coverage_matrix),
        where=coverage_sums > 1e-12,
    )
    entropy_denominator = max(
        float(np.log(max(len(part_label_ids), 2))),
        1e-12,
    )
    coverage_entropy = (
        -np.sum(
            coverage_probabilities
            * np.log(np.clip(coverage_probabilities, 1e-12, 1.0)),
            axis=1,
        )
        / entropy_denominator
    )
    coverage_margin = np.maximum(
        coverage_array - secondary_coverage_array,
        0.0,
    )
    positive_coverage_fraction = np.mean(
        coverage_matrix > 0.0,
        axis=1,
    )
    coverage_margin[orphan_array] = 0.0
    coverage_entropy[orphan_array] = 0.0
    positive_coverage_fraction[orphan_array] = 0.0

    return SQCoverageLabelResult(
        labels=labels_array,
        coverage=coverage_array,
        secondary_labels=secondary_labels_array,
        secondary_coverage=secondary_coverage_array,
        coverage_margin=coverage_margin,
        coverage_entropy=coverage_entropy,
        positive_coverage_fraction=positive_coverage_fraction,
        part_labels=part_label_ids,
        coverage_matrix=coverage_matrix,
        coverage_probabilities=coverage_probabilities,
        orphan_mask=orphan_array,
        near_counts=np.asarray(near_counts, dtype=np.int64),
        sq_bounds=np.asarray(sq_bounds_all, dtype=np.float64),
        part_bounds=part_bounds,
        part_bounds_source=part_bounds_source,
        part_bounds_error=part_bounds_error,
        alignment=alignment,
        algorithm_config=algorithm_config,
    )


def sq_label_cache_paths(
    cache_dir: Path,
    sample_id: str,
) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return (
        cache_dir / f"{sample_id}.npz",
        cache_dir / f"{sample_id}.json",
    )


def save_sq_coverage_cache(
    result: SQCoverageLabelResult,
    cache_dir: Path,
    *,
    sample_id: str,
    cache_key: str,
    bootstrap_source: str,
) -> Dict[str, str]:
    """Atomically persist the static per-SQ coverage supervision."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = sq_label_cache_paths(cache_dir, sample_id)
    suffix = f".{os.getpid()}.tmp"
    tmp_npz = npz_path.with_name(f".{npz_path.name}{suffix}.npz")
    tmp_json = json_path.with_name(f".{json_path.name}{suffix}")
    part_bounds_array = np.asarray(
        [result.part_bounds[int(label)] for label in result.part_labels],
        dtype=np.float64,
    )

    np.savez_compressed(
        tmp_npz,
        schema_version=np.asarray(SQ_LABEL_CACHE_SCHEMA_VERSION, dtype=np.int64),
        algorithm_version=np.asarray(SQ_LABEL_ALGORITHM_VERSION),
        cache_key=np.asarray(str(cache_key)),
        labels=result.labels,
        coverage=result.coverage,
        secondary_labels=result.secondary_labels,
        secondary_coverage=result.secondary_coverage,
        coverage_margin=result.coverage_margin,
        coverage_entropy=result.coverage_entropy,
        positive_coverage_fraction=result.positive_coverage_fraction,
        part_labels=result.part_labels,
        coverage_matrix=result.coverage_matrix,
        coverage_probabilities=result.coverage_probabilities,
        orphan_mask=result.orphan_mask,
        near_counts=result.near_counts,
        sq_bounds=result.sq_bounds,
        part_bounds=part_bounds_array,
        part_bounds_source=np.asarray(result.part_bounds_source),
        part_bounds_error=np.asarray(result.part_bounds_error or ""),
        alignment_json=np.asarray(json.dumps(result.alignment, sort_keys=True)),
        algorithm_config_json=np.asarray(
            json.dumps(result.algorithm_config, sort_keys=True)
        ),
    )
    metadata = {
        "schema_version": SQ_LABEL_CACHE_SCHEMA_VERSION,
        "algorithm_version": SQ_LABEL_ALGORITHM_VERSION,
        "cache_key": str(cache_key),
        "sample_id": str(sample_id),
        "bootstrap_source": str(bootstrap_source),
        "num_primitives": int(result.labels.shape[0]),
        "num_gt_parts": int(result.part_labels.shape[0]),
        "orphan_count": int(result.orphan_mask.sum()),
        "part_bounds_source": result.part_bounds_source,
        "part_bounds_error": result.part_bounds_error,
        "alignment": result.alignment,
        "algorithm_config": result.algorithm_config,
        "npz_path": str(npz_path),
    }
    tmp_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_npz, npz_path)
    os.replace(tmp_json, json_path)
    result.cache_key = str(cache_key)
    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
    }


def load_sq_coverage_cache(
    cache_dir: Path,
    *,
    sample_id: str,
    expected_cache_key: Optional[str],
) -> Tuple[Optional[SQCoverageLabelResult], str]:
    """Load a cache when schema, algorithm, optional key, and shapes all match."""
    npz_path, _ = sq_label_cache_paths(cache_dir, sample_id)
    if not npz_path.is_file():
        return None, "miss"
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            schema_version = int(np.asarray(data["schema_version"]).item())
            algorithm_version = str(np.asarray(data["algorithm_version"]).item())
            cache_key = str(np.asarray(data["cache_key"]).item())
            if schema_version != SQ_LABEL_CACHE_SCHEMA_VERSION:
                return None, f"schema_mismatch:{schema_version}"
            if algorithm_version != SQ_LABEL_ALGORITHM_VERSION:
                return None, f"algorithm_mismatch:{algorithm_version}"
            if expected_cache_key is not None and cache_key != str(expected_cache_key):
                return None, "key_mismatch"

            part_labels = np.asarray(data["part_labels"], dtype=np.int64)
            part_bounds_array = np.asarray(data["part_bounds"], dtype=np.float64)
            coverage_matrix = np.asarray(
                data["coverage_matrix"],
                dtype=np.float64,
            )
            labels = np.asarray(data["labels"], dtype=np.int64)
            if coverage_matrix.shape != (len(labels), len(part_labels)):
                return None, "coverage_matrix_shape_mismatch"
            if part_bounds_array.shape != (len(part_labels), 2, 3):
                return None, "part_bounds_shape_mismatch"

            result = SQCoverageLabelResult(
                labels=labels,
                coverage=np.asarray(data["coverage"], dtype=np.float64),
                secondary_labels=np.asarray(
                    data["secondary_labels"],
                    dtype=np.int64,
                ),
                secondary_coverage=np.asarray(
                    data["secondary_coverage"],
                    dtype=np.float64,
                ),
                coverage_margin=np.asarray(
                    data["coverage_margin"],
                    dtype=np.float64,
                ),
                coverage_entropy=np.asarray(
                    data["coverage_entropy"],
                    dtype=np.float64,
                ),
                positive_coverage_fraction=np.asarray(
                    data["positive_coverage_fraction"],
                    dtype=np.float64,
                ),
                part_labels=part_labels,
                coverage_matrix=coverage_matrix,
                coverage_probabilities=np.asarray(
                    data["coverage_probabilities"],
                    dtype=np.float64,
                ),
                orphan_mask=np.asarray(data["orphan_mask"], dtype=bool),
                near_counts=np.asarray(data["near_counts"], dtype=np.int64),
                sq_bounds=np.asarray(data["sq_bounds"], dtype=np.float64),
                part_bounds={
                    int(label): part_bounds_array[index]
                    for index, label in enumerate(part_labels)
                },
                part_bounds_source=str(np.asarray(data["part_bounds_source"]).item()),
                part_bounds_error=(
                    str(np.asarray(data["part_bounds_error"]).item()) or None
                ),
                alignment=json.loads(str(np.asarray(data["alignment_json"]).item())),
                algorithm_config=json.loads(
                    str(np.asarray(data["algorithm_config_json"]).item())
                ),
                cache_key=cache_key,
            )
        expected_vector_shape = (len(result.labels),)
        vector_fields = (
            result.coverage,
            result.secondary_labels,
            result.secondary_coverage,
            result.coverage_margin,
            result.coverage_entropy,
            result.positive_coverage_fraction,
            result.orphan_mask,
            result.near_counts,
        )
        if any(value.shape != expected_vector_shape for value in vector_fields):
            return None, "vector_shape_mismatch"
        if result.coverage_probabilities.shape != result.coverage_matrix.shape:
            return None, "coverage_probability_shape_mismatch"
        if result.sq_bounds.shape != (len(result.labels), 2, 3):
            return None, "sq_bounds_shape_mismatch"
        return result, "hit"
    except Exception as exc:
        return None, f"invalid:{type(exc).__name__}:{exc}"
