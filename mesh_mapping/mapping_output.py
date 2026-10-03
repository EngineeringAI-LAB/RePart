"""Minimal mapping outputs and optional debug artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import trimesh

from .sq_geometry import ensure_mesh


def group_color_rgba(group_id: int) -> np.ndarray:
    """Match the RePart group colors used by the SQ exporter."""
    rng = np.random.RandomState(int(group_id) * 9973 + 17)
    rgb = rng.randint(48, 256, size=3, dtype=np.uint8)
    return np.asarray([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)


def _atom_palette(count: int) -> np.ndarray:
    rng = np.random.RandomState(525)
    rgb = rng.randint(60, 240, size=(max(int(count), 1), 3), dtype=np.uint8)
    return np.concatenate(
        [rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=1
    )


def _vertex_colors_from_faces(
    faces: np.ndarray,
    face_group_index: np.ndarray,
    palette: np.ndarray,
    orphan_color: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    num_groups = len(palette)
    bins = np.zeros((int(num_vertices), num_groups + 1), dtype=np.int32)
    effective = np.where(
        face_group_index >= 0, face_group_index, num_groups
    ).astype(np.int64)
    for corner in range(3):
        np.add.at(bins, (faces[:, corner], effective), 1)
    winner = bins.argmax(axis=1)
    colors = np.tile(orphan_color, (num_vertices, 1))
    valid = winner < num_groups
    colors[valid] = palette[winner[valid]]
    return colors


def save_mapping(
    mesh: trimesh.Trimesh,
    atom_labels: np.ndarray,
    atom_report: Dict[str, Any],
    result: Dict[str, Any],
    group_ids: Sequence[int],
    point_cloud_info: Dict[str, Any],
    grouped_payload: Dict[str, Any],
    paths: Dict[str, str],
    output_dir: Path,
    *,
    save_per_group: bool,
    save_debug_artifacts: bool,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Save flat mapping, curvature atoms, and per-group meshes by default."""
    output_dir.mkdir(parents=True, exist_ok=True)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    face_group_index = np.asarray(result["face_group_index"], dtype=np.int32)
    palette = np.stack(
        [group_color_rgba(int(group_id)) for group_id in group_ids], axis=0
    )
    orphan_color = np.asarray([90, 90, 90, 255], dtype=np.uint8)
    face_colors = np.tile(orphan_color, (len(faces), 1))
    valid = face_group_index >= 0
    face_colors[valid] = palette[face_group_index[valid]]

    flat_vertices = vertices[faces].reshape(-1, 3)
    flat_faces = np.arange(len(flat_vertices), dtype=np.int64).reshape(-1, 3)
    flat_mapped = trimesh.Trimesh(
        vertices=flat_vertices, faces=flat_faces, process=False
    )
    flat_mapped.visual.vertex_colors = np.repeat(face_colors, 3, axis=0)
    flat_path = output_dir / "mesh_mapped_grouped_sq_flat.ply"
    flat_mapped.export(flat_path)

    atom_palette = _atom_palette(int(atom_labels.max()) + 1)
    atom_vis = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    )
    atom_vis.visual.face_colors = atom_palette[atom_labels]
    atoms_path = output_dir / "mesh_curvature_atoms.ply"
    atom_vis.export(atoms_path)

    debug_paths: Dict[str, str] = {}
    if save_debug_artifacts:
        actual_group_ids = np.full(len(face_group_index), -1, dtype=np.int32)
        for index, group_id in enumerate(group_ids):
            actual_group_ids[face_group_index == index] = int(group_id)

        mapped = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )
        mapped.visual.face_colors = face_colors
        mapped.visual.vertex_colors = _vertex_colors_from_faces(
            faces,
            face_group_index,
            palette,
            orphan_color,
            len(vertices),
        )
        mapped_path = output_dir / "mesh_mapped_grouped_sq.ply"
        mapped.export(mapped_path)

        nearest_group_index = np.asarray(
            result["nearest_group_index"], dtype=np.int32
        )
        nearest_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )
        nearest_mesh.visual.face_colors = palette[nearest_group_index]
        nearest_mesh.visual.vertex_colors = _vertex_colors_from_faces(
            faces,
            nearest_group_index,
            palette,
            orphan_color,
            len(vertices),
        )
        nearest_path = output_dir / "mesh_nearest_grouped_sq.ply"
        nearest_mesh.export(nearest_path)

        np.save(output_dir / "face_group_indices.npy", face_group_index)
        np.save(output_dir / "face_group_ids.npy", actual_group_ids)
        np.save(
            output_dir / "face_nearest_group_indices.npy",
            nearest_group_index,
        )
        np.save(
            output_dir / "face_atom_labels.npy",
            atom_labels.astype(np.int32),
        )
        np.save(
            output_dir / "atom_confidence.npy",
            np.asarray(result["atom_confidence"], dtype=np.float32),
        )
        np.save(
            output_dir / "face_nearest_distance.npy",
            np.asarray(result["nearest_distance"], dtype=np.float32),
        )
        debug_paths = {
            "mapped_mesh": str(mapped_path),
            "nearest_baseline_mesh": str(nearest_path),
        }

    saved_group_meshes: Dict[str, str] = {}
    if save_per_group:
        for index, group_id in enumerate(group_ids):
            face_indices = np.where(face_group_index == index)[0]
            if len(face_indices) == 0:
                continue
            submesh = ensure_mesh(
                mesh.submesh(
                    [face_indices], only_watertight=False, append=True
                )
            )
            submesh.visual.face_colors = np.tile(
                palette[index][None, :], (len(submesh.faces), 1)
            )
            submesh_path = output_dir / f"per_group_{int(group_id):03d}.ply"
            submesh.export(submesh_path)
            saved_group_meshes[str(int(group_id))] = str(submesh_path)

    report = {
        "algorithm": "curvature_atom_voting_for_sq_groups",
        "paths": {
            **paths,
            "output_dir": str(output_dir),
            "mapped_mesh_flat": str(flat_path),
            "curvature_atoms": str(atoms_path),
            **debug_paths,
        },
        "sample_id": grouped_payload.get("sample_id"),
        "num_primitives": int(grouped_payload.get("num_primitives", 0)),
        "num_groups": int(len(group_ids)),
        "group_index_to_id": {
            str(index): int(group_id)
            for index, group_id in enumerate(group_ids)
        },
        "group_members": grouped_payload.get("group_members", {}),
        "point_cloud": point_cloud_info,
        "atoms": atom_report,
        "mapping": result["stats"],
        "per_group_meshes": saved_group_meshes,
        "debug_artifacts_saved": bool(save_debug_artifacts),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if save_debug_artifacts:
        (output_dir / "mapping_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report
