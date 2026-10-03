"""Curvature-atom projection and public RePart mapping interfaces."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .mapping_output import group_color_rgba, save_mapping
from .sq_geometry import (
    build_group_point_clouds,
    compute_face_group_containment_sdf,
    ensure_mesh,
    export_aligned_grouped_sq_mesh,
    normalize_mesh,
)


def _adjacency_graph(
    num_faces: int,
    face_adjacency: np.ndarray,
    weights: np.ndarray,
) -> sp.csr_matrix:
    if len(face_adjacency) == 0:
        return sp.csr_matrix((num_faces, num_faces), dtype=np.float64)
    rows = np.concatenate([face_adjacency[:, 0], face_adjacency[:, 1]])
    cols = np.concatenate([face_adjacency[:, 1], face_adjacency[:, 0]])
    data = np.concatenate([weights, weights]).astype(np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_faces, num_faces))


def _smooth_atom_labels(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    iterations: int,
    min_agreement: int = 2,
) -> np.ndarray:
    if iterations <= 0:
        return labels
    face_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(face_adjacency) == 0:
        return labels
    num_faces = len(mesh.faces)
    rows = np.concatenate([face_adjacency[:, 0], face_adjacency[:, 1]])
    cols = np.concatenate([face_adjacency[:, 1], face_adjacency[:, 0]])
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = cols[order]
    indptr = np.zeros(num_faces + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)

    current = labels.copy()
    for _ in range(int(iterations)):
        updated = current.copy()
        changed = 0
        for face_idx in range(num_faces):
            start, end = indptr[face_idx], indptr[face_idx + 1]
            degree = int(end - start)
            if degree < 2:
                continue
            values, counts = np.unique(
                current[cols[start:end]], return_counts=True
            )
            winner = int(np.argmax(counts))
            required = max(int(min_agreement), degree // 2 + 1)
            if (
                int(counts[winner]) >= required
                and int(values[winner]) != int(current[face_idx])
            ):
                updated[face_idx] = values[winner]
                changed += 1
        current = updated
        if changed == 0:
            break
    return current


def generate_dihedral_atoms(
    mesh: trimesh.Trimesh,
    *,
    cut_percentile: float,
    cut_abs_min_deg: float,
    cut_abs_max_deg: float,
    min_faces: int,
    smooth_iters: int,
):
    """Generate HITOPS-style atoms without the optional libigl term."""
    num_faces = int(len(mesh.faces))
    face_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    angles = np.degrees(
        np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    )
    if len(face_adjacency) == 0:
        labels = np.arange(num_faces, dtype=np.int32)
        return labels, {
            "source": "generated_dihedral",
            "num_atoms": num_faces,
            "cut_edges": 0,
            "retained_isolated_small_atoms": num_faces,
        }

    percentile_value = float(np.percentile(angles, cut_percentile))
    cut_threshold = float(
        np.clip(percentile_value, cut_abs_min_deg, cut_abs_max_deg)
    )
    keep = angles <= cut_threshold
    keep_graph = _adjacency_graph(
        num_faces,
        face_adjacency[keep],
        np.ones(int(np.count_nonzero(keep)), dtype=np.float64),
    )
    initial_count, initial_labels = connected_components(
        keep_graph, directed=False
    )
    initial_labels = initial_labels.astype(np.int64)
    component_sizes = np.bincount(initial_labels, minlength=initial_count)
    big_components = np.where(component_sizes >= int(min_faces))[0]

    labels = np.full(num_faces, -1, dtype=np.int64)
    for new_label, component in enumerate(big_components.tolist()):
        labels[initial_labels == int(component)] = int(new_label)

    reachable_count = 0
    if len(big_components) > 0:
        graph = _adjacency_graph(num_faces, face_adjacency, angles + 1e-6)
        sources = np.where(labels >= 0)[0]
        distance, _, closest_source = dijkstra(
            graph,
            indices=sources,
            min_only=True,
            return_predecessors=True,
        )
        unresolved = labels < 0
        reachable = (
            unresolved
            & np.isfinite(distance)
            & (np.asarray(closest_source, dtype=np.int64) >= 0)
        )
        labels[reachable] = labels[
            np.asarray(closest_source, dtype=np.int64)[reachable]
        ]
        reachable_count = int(np.count_nonzero(reachable))

    unresolved = labels < 0
    retained_components = np.unique(initial_labels[unresolved])
    next_label = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    for component in retained_components.tolist():
        mask = unresolved & (initial_labels == int(component))
        labels[mask] = next_label
        next_label += 1

    labels = _smooth_atom_labels(mesh, labels, int(smooth_iters))
    unique = np.unique(labels)
    remap = {int(value): idx for idx, value in enumerate(unique.tolist())}
    labels = np.asarray([remap[int(value)] for value in labels], dtype=np.int32)
    _, counts = np.unique(labels, return_counts=True)
    return labels, {
        "source": "generated_dihedral",
        "cut_percentile": float(cut_percentile),
        "cut_percentile_value_deg": percentile_value,
        "cut_abs_min_deg": float(cut_abs_min_deg),
        "cut_abs_max_deg": float(cut_abs_max_deg),
        "cut_threshold_deg": cut_threshold,
        "cut_edges": int(np.count_nonzero(~keep)),
        "initial_components": int(initial_count),
        "min_faces": int(min_faces),
        "dijkstra_reassigned_faces": int(reachable_count),
        "retained_isolated_small_atoms": int(len(retained_components)),
        "smooth_iters": int(smooth_iters),
        "num_atoms": int(len(counts)),
        "atom_size_min": int(counts.min()) if len(counts) else 0,
        "atom_size_median": float(np.median(counts)) if len(counts) else 0.0,
        "atom_size_max": int(counts.max()) if len(counts) else 0,
    }


def map_atoms_to_groups(
    mesh: trimesh.Trimesh,
    atom_labels: np.ndarray,
    group_clouds: Sequence[np.ndarray],
    *,
    vote_mode: str,
    vote_tau: float,
    orphan_face_dist: float,
    orphan_atom_frac: float,
    min_atom_size: int,
    containment_group_sdf: Optional[np.ndarray] = None,
    containment_margin: float = 0.005,
    containment_rescue_frac: float = 0.5,
) -> Dict[str, Any]:
    """Assign each indivisible curvature atom to one RePart group."""
    atom_labels = np.asarray(atom_labels, dtype=np.int64).reshape(-1)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(atom_labels) != len(faces):
        raise ValueError(
            f"Atom label count {len(atom_labels)} != mesh face count {len(faces)}"
        )
    if not group_clouds:
        raise ValueError("No RePart group point clouds")
    if any(len(cloud) == 0 for cloud in group_clouds):
        raise ValueError("Every RePart group point cloud must be non-empty")

    normalized_vertices, scale, center = normalize_mesh(mesh.vertices)
    face_centers = normalized_vertices[faces].mean(axis=1)
    all_points = np.concatenate(group_clouds, axis=0)
    point_group = np.concatenate(
        [
            np.full(len(cloud), index, dtype=np.int32)
            for index, cloud in enumerate(group_clouds)
        ]
    )
    tree = cKDTree(all_points)
    nearest_dist, nearest_index = tree.query(face_centers, k=1)
    nearest_group = point_group[nearest_index].astype(np.int32)
    nearest_dist = nearest_dist.astype(np.float32)
    num_groups = len(group_clouds)

    containment_candidate = np.full(len(faces), -1, dtype=np.int32)
    contained_face = np.zeros(len(faces), dtype=bool)
    if containment_group_sdf is not None:
        containment_group_sdf = np.asarray(
            containment_group_sdf, dtype=np.float32
        )
        expected_shape = (len(faces), num_groups)
        if containment_group_sdf.shape != expected_shape:
            raise ValueError(
                "containment_group_sdf shape "
                f"{containment_group_sdf.shape} != {expected_shape}"
            )
        contained = containment_group_sdf <= float(containment_margin)
        contained_face = np.any(contained, axis=1)
        nearest_is_contained = contained[
            np.arange(len(faces)), nearest_group
        ]
        containment_candidate[nearest_is_contained] = nearest_group[
            nearest_is_contained
        ]
        need_fallback = contained_face & ~nearest_is_contained
        if np.any(need_fallback):
            masked_sdf = np.where(contained, containment_group_sdf, np.inf)
            containment_candidate[need_fallback] = np.argmin(
                masked_sdf[need_fallback], axis=1
            ).astype(np.int32)

    valid_atom_face = atom_labels >= 0
    num_atoms = (
        int(atom_labels[valid_atom_face].max()) + 1
        if np.any(valid_atom_face)
        else 0
    )
    if num_atoms == 0:
        face_group_index = nearest_group.copy()
        accepted_before_fallback = nearest_dist <= float(orphan_face_dist)
        return {
            "face_group_index": face_group_index,
            "nearest_group_index": nearest_group,
            "atom_group_index": np.zeros(0, dtype=np.int32),
            "atom_confidence": np.zeros(0, dtype=np.float64),
            "nearest_distance": nearest_dist,
            "scale": scale,
            "center": center,
            "stats": {
                "num_faces": int(len(faces)),
                "num_atoms": 0,
                "num_groups": int(num_groups),
                "pre_fallback_face_coverage": float(
                    np.mean(accepted_before_fallback)
                ),
                "face_coverage": float(np.mean(face_group_index >= 0)),
                "fallback_faces": int(
                    np.count_nonzero(~accepted_before_fallback)
                ),
                "fallback_atoms": 0,
                "fallback_atom_faces": 0,
                "fallback_preexisting_faces": int(
                    np.count_nonzero(~accepted_before_fallback)
                ),
                "orphan_atoms": 0,
                "assigned_faces": int(len(faces)),
                "orphan_faces": 0,
            },
        }

    face_orphan_by_distance = nearest_dist > float(orphan_face_dist)
    if vote_mode == "count":
        weights = np.ones(len(faces), dtype=np.float64)
    elif vote_mode == "exp":
        weights = np.exp(
            -nearest_dist / max(float(vote_tau), 1e-6)
        ).astype(np.float64)
    else:
        raise ValueError(f"Unknown vote_mode: {vote_mode}")
    weights[~valid_atom_face] = 0.0
    weights[face_orphan_by_distance] = 0.0

    votes = np.zeros((num_atoms, num_groups), dtype=np.float64)
    voting_faces = valid_atom_face & ~face_orphan_by_distance
    np.add.at(
        votes,
        (atom_labels[voting_faces], nearest_group[voting_faces]),
        weights[voting_faces],
    )
    total_votes = votes.sum(axis=1)
    atom_group_index = np.full(num_atoms, -1, dtype=np.int32)
    has_vote = total_votes > 0.0
    atom_group_index[has_vote] = votes[has_vote].argmax(axis=1).astype(np.int32)
    atom_confidence = np.zeros(num_atoms, dtype=np.float64)
    atom_confidence[has_vote] = (
        votes[has_vote].max(axis=1) / total_votes[has_vote]
    )

    atom_size = np.zeros(num_atoms, dtype=np.int64)
    np.add.at(atom_size, atom_labels[valid_atom_face], 1)
    distance_orphans = np.zeros(num_atoms, dtype=np.int64)
    np.add.at(
        distance_orphans,
        atom_labels[valid_atom_face & face_orphan_by_distance],
        1,
    )
    orphan_fraction = distance_orphans / np.maximum(atom_size, 1)
    atom_orphan = (
        (atom_group_index < 0)
        | (orphan_fraction > float(orphan_atom_frac))
        | (atom_size < int(min_atom_size))
    )

    rescue_votes = np.zeros((num_atoms, num_groups), dtype=np.int64)
    rescue_faces = valid_atom_face & (containment_candidate >= 0)
    np.add.at(
        rescue_votes,
        (
            atom_labels[rescue_faces],
            containment_candidate[rescue_faces],
        ),
        1,
    )
    rescue_total = rescue_votes.sum(axis=1)
    rescue_group = np.full(num_atoms, -1, dtype=np.int32)
    has_rescue = rescue_total > 0
    rescue_group[has_rescue] = rescue_votes[has_rescue].argmax(axis=1)
    rescue_share = rescue_votes.max(axis=1) / np.maximum(atom_size, 1)
    rescued_atom = (
        atom_orphan
        & (atom_size >= int(min_atom_size))
        & has_rescue
        & (rescue_share >= float(containment_rescue_frac))
    )
    atom_group_index[rescued_atom] = rescue_group[rescued_atom]
    atom_confidence[rescued_atom] = rescue_share[rescued_atom]
    atom_orphan[rescued_atom] = False

    # Preserve atom consistency while guaranteeing a dense final mapping. If
    # both thresholded proximity voting and volumetric rescue fail, fall back
    # to an unrestricted nearest-surface majority vote over the atom.
    fallback_votes = np.zeros((num_atoms, num_groups), dtype=np.int64)
    np.add.at(
        fallback_votes,
        (atom_labels[valid_atom_face], nearest_group[valid_atom_face]),
        1,
    )
    fallback_total = fallback_votes.sum(axis=1)
    fallback_group = fallback_votes.argmax(axis=1).astype(np.int32)
    fallback_atom = atom_orphan & (atom_size > 0) & (fallback_total > 0)
    atom_group_index[fallback_atom] = fallback_group[fallback_atom]
    atom_confidence[fallback_atom] = (
        fallback_votes[fallback_atom].max(axis=1)
        / fallback_total[fallback_atom]
    )
    atom_orphan[fallback_atom] = False
    atom_group_index[atom_orphan] = -1

    face_group_index = np.full(len(faces), -1, dtype=np.int32)
    face_group_index[valid_atom_face] = atom_group_index[
        atom_labels[valid_atom_face]
    ]
    preexisting_orphan = ~valid_atom_face
    recover_near = preexisting_orphan & ~face_orphan_by_distance
    face_group_index[recover_near] = nearest_group[recover_near]
    recover_contained = (
        preexisting_orphan
        & face_orphan_by_distance
        & (containment_candidate >= 0)
    )
    face_group_index[recover_contained] = containment_candidate[
        recover_contained
    ]
    fallback_preexisting = preexisting_orphan & (face_group_index < 0)
    face_group_index[fallback_preexisting] = nearest_group[fallback_preexisting]

    assigned = face_group_index >= 0
    used = np.unique(face_group_index[assigned])
    nearest_group_counts = np.bincount(
        nearest_group, minlength=num_groups
    ).astype(np.int64)
    valid_confidence = atom_confidence[~atom_orphan]
    stats = {
        "num_faces": int(len(faces)),
        "num_atoms": int(num_atoms),
        "num_groups": int(num_groups),
        "total_group_surface_points": int(len(all_points)),
        "distance_orphan_faces": int(np.count_nonzero(face_orphan_by_distance)),
        "distance_orphan_faces_inside_sq": int(
            np.count_nonzero(face_orphan_by_distance & contained_face)
        ),
        "containment_rescued_atoms": int(np.count_nonzero(rescued_atom)),
        "containment_rescued_atom_faces": int(
            np.count_nonzero(
                valid_atom_face
                & np.isin(atom_labels, np.flatnonzero(rescued_atom))
            )
        ),
        "containment_recovered_preexisting_faces": int(
            np.count_nonzero(recover_contained)
        ),
        "fallback_atoms": int(np.count_nonzero(fallback_atom)),
        "fallback_atom_faces": int(
            np.count_nonzero(
                valid_atom_face
                & np.isin(atom_labels, np.flatnonzero(fallback_atom))
            )
        ),
        "fallback_preexisting_faces": int(
            np.count_nonzero(fallback_preexisting)
        ),
        "fallback_faces": int(
            np.count_nonzero(
                valid_atom_face
                & np.isin(atom_labels, np.flatnonzero(fallback_atom))
            )
            + np.count_nonzero(fallback_preexisting)
        ),
        "orphan_atoms": int(np.count_nonzero(atom_orphan & (atom_size > 0))),
        "assigned_faces": int(np.count_nonzero(assigned)),
        "orphan_faces": int(np.count_nonzero(~assigned)),
        "pre_fallback_face_coverage": float(
            1.0
            - (
                np.count_nonzero(
                    valid_atom_face
                    & np.isin(atom_labels, np.flatnonzero(fallback_atom))
                )
                + np.count_nonzero(fallback_preexisting)
            )
            / max(len(faces), 1)
        ),
        "face_coverage": float(np.mean(assigned)),
        "groups_used": int(len(used)),
        "nearest_groups_used": int(np.count_nonzero(nearest_group_counts)),
        "nearest_group_face_counts": nearest_group_counts.tolist(),
        "atom_confidence_mean": (
            float(valid_confidence.mean()) if len(valid_confidence) else 0.0
        ),
        "atom_confidence_median": (
            float(np.median(valid_confidence)) if len(valid_confidence) else 0.0
        ),
        "nearest_distance_p50": float(np.percentile(nearest_dist, 50)),
        "nearest_distance_p90": float(np.percentile(nearest_dist, 90)),
        "nearest_distance_p99": float(np.percentile(nearest_dist, 99)),
        "vote_mode": vote_mode,
        "vote_tau": float(vote_tau),
        "orphan_face_dist": float(orphan_face_dist),
        "orphan_atom_frac": float(orphan_atom_frac),
        "min_atom_size": int(min_atom_size),
        "containment_margin": float(containment_margin),
        "containment_rescue_frac": float(containment_rescue_frac),
    }
    return {
        "face_group_index": face_group_index,
        "nearest_group_index": nearest_group,
        "atom_group_index": atom_group_index,
        "atom_confidence": atom_confidence,
        "nearest_distance": nearest_dist,
        "scale": float(scale),
        "center": center.astype(np.float64),
        "stats": stats,
    }


MeshInput = Union[str, Path, trimesh.Trimesh]
ArrayInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class MappingConfig:
    """Configuration shared by file and in-memory mapping interfaces."""

    sq_frame: str = "auto"
    surface_nu: int = 72
    surface_nv: int = 36
    max_points_per_group: int = 20000
    cut_percentile: float = 90.0
    cut_abs_min_deg: float = 6.0
    cut_abs_max_deg: float = 12.0
    atom_min_faces: int = 60
    atom_smooth_iters: int = 1
    vote_mode: str = "count"
    vote_tau: float = 0.025
    orphan_face_dist: float = 0.04
    orphan_atom_frac: float = 0.6
    min_atom_size: int = 3
    containment_rescue: bool = True
    containment_margin: float = 0.005
    containment_rescue_frac: float = 0.5
    save_per_group: bool = True
    save_debug_artifacts: bool = False


def _resolve_mesh(mesh: MeshInput):
    if isinstance(mesh, (str, Path)):
        mesh_path = Path(mesh).resolve()
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        return ensure_mesh(loaded), str(mesh_path)
    return ensure_mesh(mesh), "<in_memory>"


def _resolve_atom_labels(
    face_atom_labels: Optional[ArrayInput],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if face_atom_labels is None:
        return None, None
    if isinstance(face_atom_labels, (str, Path)):
        path = Path(face_atom_labels).resolve()
        return np.load(path).astype(np.int32), str(path)
    return np.asarray(face_atom_labels, dtype=np.int32), "<in_memory>"


def map_grouped_sq_data(
    mesh: MeshInput,
    grouped_sqs: np.ndarray,
    grouped_payload: Dict[str, Any],
    output_dir: Union[str, Path],
    *,
    config: Optional[MappingConfig] = None,
    face_atom_labels: Optional[ArrayInput] = None,
    _source_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Map already-loaded grouped SQ data to a mesh.

    ``grouped_sqs`` is the loaded ``*_grouped_multi_sq.npy`` array and
    ``grouped_payload`` is the loaded companion JSON dictionary.
    """
    started = time.time()
    config = config or MappingConfig()
    mesh_obj, mesh_source = _resolve_mesh(mesh)
    xs = np.asarray(grouped_sqs)
    group_ids_per_primitive = [
        int(value) for value in grouped_payload["group_ids"]
    ]
    group_ids, group_clouds, point_cloud_info = build_group_point_clouds(
        xs,
        group_ids_per_primitive,
        grouped_payload,
        np.asarray(mesh_obj.vertices, dtype=np.float64),
        sq_frame=config.sq_frame,
        surface_nu=config.surface_nu,
        surface_nv=config.surface_nv,
        max_points_per_group=config.max_points_per_group,
    )

    atom_labels, atom_source = _resolve_atom_labels(face_atom_labels)
    if atom_labels is None:
        atom_labels, atom_report = generate_dihedral_atoms(
            mesh_obj,
            cut_percentile=config.cut_percentile,
            cut_abs_min_deg=config.cut_abs_min_deg,
            cut_abs_max_deg=config.cut_abs_max_deg,
            min_faces=config.atom_min_faces,
            smooth_iters=config.atom_smooth_iters,
        )
    else:
        atom_report = {
            "source": "precomputed",
            "path": atom_source,
            "num_atoms": (
                int(atom_labels[atom_labels >= 0].max()) + 1
                if np.any(atom_labels >= 0)
                else 0
            ),
        }

    containment_group_sdf = None
    containment_info = {"enabled": False}
    if config.containment_rescue:
        containment_group_sdf, containment_info = (
            compute_face_group_containment_sdf(
                mesh_obj,
                xs,
                group_ids_per_primitive,
                group_ids,
                grouped_payload,
                sq_frame=config.sq_frame,
            )
        )
        containment_info["margin"] = float(config.containment_margin)
        containment_info["rescue_atom_fraction"] = float(
            config.containment_rescue_frac
        )
    point_cloud_info["containment_rescue"] = containment_info

    result = map_atoms_to_groups(
        mesh_obj,
        atom_labels,
        group_clouds,
        vote_mode=config.vote_mode,
        vote_tau=config.vote_tau,
        orphan_face_dist=config.orphan_face_dist,
        orphan_atom_frac=config.orphan_atom_frac,
        min_atom_size=config.min_atom_size,
        containment_group_sdf=containment_group_sdf,
        containment_margin=config.containment_margin,
        containment_rescue_frac=config.containment_rescue_frac,
    )

    output_path = Path(output_dir).resolve()
    paths = {
        "mesh": mesh_source,
        "grouped_sq_npy": "<in_memory>",
        "grouped_sq_json": "<in_memory>",
    }
    if _source_paths:
        paths.update(_source_paths)
    if config.save_debug_artifacts:
        paths["aligned_grouped_sq_mesh"] = export_aligned_grouped_sq_mesh(
            xs,
            group_ids_per_primitive,
            grouped_payload,
            np.asarray(mesh_obj.vertices, dtype=np.float64),
            output_path / "grouped_sq_aligned_to_mesh.ply",
            sq_frame=config.sq_frame,
            color_fn=group_color_rgba,
        )

    report = save_mapping(
        mesh_obj,
        atom_labels,
        atom_report,
        result,
        group_ids,
        point_cloud_info,
        grouped_payload,
        paths,
        output_path,
        save_per_group=config.save_per_group,
        save_debug_artifacts=config.save_debug_artifacts,
        elapsed_seconds=time.time() - started,
    )
    return {
        "report": report,
        "mapping": result,
        "atom_labels": atom_labels,
        "group_ids": group_ids,
    }


def map_grouped_sq_file(
    mesh: MeshInput,
    grouped_sq_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    grouped_sq_json_path: Optional[Union[str, Path]] = None,
    config: Optional[MappingConfig] = None,
    face_atom_labels: Optional[ArrayInput] = None,
) -> Dict[str, Any]:
    """Load grouped SQ NPY/JSON files and delegate to the in-memory interface."""
    npy_path = Path(grouped_sq_path).resolve()
    json_path = (
        Path(grouped_sq_json_path).resolve()
        if grouped_sq_json_path is not None
        else npy_path.with_suffix(".json")
    )
    grouped_sqs = np.load(npy_path)
    grouped_payload = json.loads(json_path.read_text(encoding="utf-8"))
    return map_grouped_sq_data(
        mesh,
        grouped_sqs,
        grouped_payload,
        output_dir,
        config=config,
        face_atom_labels=face_atom_labels,
        _source_paths={
            "grouped_sq_npy": str(npy_path),
            "grouped_sq_json": str(json_path),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project RePart SQ groups to mesh faces with curvature-atom voting"
    )
    parser.add_argument("--mesh", required=True, help="Raw/source mesh path")
    parser.add_argument("--grouped-sq-npy", required=True)
    parser.add_argument("--grouped-sq-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--face-atom-labels", default=None)
    parser.add_argument(
        "--sq-frame",
        choices=["auto", "raw", "normalized"],
        default="auto",
    )
    parser.add_argument("--surface-nu", type=int, default=72)
    parser.add_argument("--surface-nv", type=int, default=36)
    parser.add_argument("--max-points-per-group", type=int, default=20000)
    parser.add_argument("--cut-percentile", type=float, default=90.0)
    parser.add_argument("--cut-abs-min-deg", type=float, default=6.0)
    parser.add_argument("--cut-abs-max-deg", type=float, default=12.0)
    parser.add_argument("--atom-min-faces", type=int, default=60)
    parser.add_argument("--atom-smooth-iters", type=int, default=1)
    parser.add_argument(
        "--vote-mode", choices=["count", "exp"], default="count"
    )
    parser.add_argument("--vote-tau", type=float, default=0.025)
    parser.add_argument("--orphan-face-dist", type=float, default=0.04)
    parser.add_argument("--orphan-atom-frac", type=float, default=0.6)
    parser.add_argument("--min-atom-size", type=int, default=3)
    parser.add_argument("--no-containment-rescue", action="store_true")
    parser.add_argument("--containment-margin", type=float, default=0.005)
    parser.add_argument("--containment-rescue-frac", type=float, default=0.5)
    parser.add_argument("--no-per-group", action="store_true")
    parser.add_argument("--save-debug-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MappingConfig(
        sq_frame=args.sq_frame,
        surface_nu=args.surface_nu,
        surface_nv=args.surface_nv,
        max_points_per_group=args.max_points_per_group,
        cut_percentile=args.cut_percentile,
        cut_abs_min_deg=args.cut_abs_min_deg,
        cut_abs_max_deg=args.cut_abs_max_deg,
        atom_min_faces=args.atom_min_faces,
        atom_smooth_iters=args.atom_smooth_iters,
        vote_mode=args.vote_mode,
        vote_tau=args.vote_tau,
        orphan_face_dist=args.orphan_face_dist,
        orphan_atom_frac=args.orphan_atom_frac,
        min_atom_size=args.min_atom_size,
        containment_rescue=not args.no_containment_rescue,
        containment_margin=args.containment_margin,
        containment_rescue_frac=args.containment_rescue_frac,
        save_per_group=not args.no_per_group,
        save_debug_artifacts=args.save_debug_artifacts,
    )
    run = map_grouped_sq_file(
        args.mesh,
        args.grouped_sq_npy,
        args.output_dir,
        grouped_sq_json_path=args.grouped_sq_json,
        config=config,
        face_atom_labels=args.face_atom_labels,
    )
    print(json.dumps(run["report"], indent=2))
