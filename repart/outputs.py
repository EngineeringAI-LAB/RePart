"""Grouped SQ exports, mesh renders, and training artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from marching_primitives.mesh import merged_superquadrics_mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from .data import PartNetSample
from .geometry import SQPrimitive, primitive_to_mps_x


def group_color_rgba(group_id: int) -> np.ndarray:
    rng = np.random.RandomState(int(group_id) * 9973 + 17)
    rgb = rng.randint(48, 256, size=3, dtype=np.uint8)
    return np.asarray([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)


def export_grouped_sq_ply(
    primitives: Sequence[SQPrimitive], group_ids: Sequence[int], path: str
) -> None:
    if len(primitives) == 0 or len(primitives) != len(group_ids):
        return
    meshes: List[trimesh.Trimesh] = []
    for prim, gid in zip(primitives, group_ids):
        try:
            x = primitive_to_mps_x(prim)[None, :]
            v, f = merged_superquadrics_mesh(x)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            mesh.visual.face_colors = np.tile(
                group_color_rgba(int(gid))[None, :], (len(f), 1)
            )
            meshes.append(mesh)
        except Exception:
            continue
    if not meshes:
        return
    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    merged.export(path)


def load_render_mesh(sample: PartNetSample) -> Optional[trimesh.Trimesh]:
    mesh_candidates: List[str] = []
    if sample.mesh_path:
        mesh_candidates.append(sample.mesh_path)
    if sample.ply_path:
        mesh_candidates.append(sample.ply_path)

    for path in mesh_candidates:
        try:
            loaded = trimesh.load(path, force="scene", process=False)
            if isinstance(loaded, trimesh.Scene):
                meshes = [
                    geom
                    for geom in loaded.geometry.values()
                    if isinstance(geom, trimesh.Trimesh) and geom.faces.size > 0
                ]
                if meshes:
                    return trimesh.util.concatenate(meshes)
                continue
            if isinstance(loaded, trimesh.Trimesh) and loaded.faces.size > 0:
                return loaded
        except Exception:
            continue
    return None


def save_group_highlight_renders(
    out_dir: str,
    sample: PartNetSample,
    sample_id: str,
    step_idx: int,
    points: np.ndarray,
    group_assign: np.ndarray,
    primitive_groups: Sequence[int],
    action_name: str,
    gt_labels: np.ndarray,
) -> None:
    if plt is None or Poly3DCollection is None:
        return
    render_mesh = load_render_mesh(sample)
    if (
        render_mesh is None
        or render_mesh.vertices.size == 0
        or render_mesh.faces.size == 0
    ):
        return

    group_dir = os.path.join(
        out_dir, f"{sample_id}_step_{step_idx:03d}_{action_name}_parts_render"
    )
    os.makedirs(group_dir, exist_ok=True)

    unique_groups: List[int] = []
    seen_groups = set()
    for gid in primitive_groups:
        gid_int = int(gid)
        if gid_int in seen_groups:
            continue
        unique_groups.append(gid_int)
        seen_groups.add(gid_int)

    for gid in np.unique(group_assign):
        gid_int = int(gid)
        if gid_int < 0 or gid_int in seen_groups:
            continue
        unique_groups.append(gid_int)
        seen_groups.add(gid_int)

    if points.shape[0] == 0 or not unique_groups:
        return

    point_tree = cKDTree(points.astype(np.float64))
    _, nearest_idx = point_tree.query(render_mesh.vertices.astype(np.float64), k=1)
    vertex_groups = group_assign[np.asarray(nearest_idx, dtype=np.int64)]
    face_vertex_groups = vertex_groups[render_mesh.faces]
    mesh_triangles = np.asarray(render_mesh.triangles, dtype=np.float64)
    pmin = np.min(render_mesh.vertices, axis=0)
    pmax = np.max(render_mesh.vertices, axis=0)
    center = 0.5 * (pmin + pmax)
    lim = max(1e-3, 0.6 * float(np.max(pmax - pmin)))

    max_base_faces = 40000
    if mesh_triangles.shape[0] > max_base_faces:
        face_keep = np.random.choice(
            mesh_triangles.shape[0], size=max_base_faces, replace=False
        )
        base_triangles = mesh_triangles[face_keep]
        base_face_groups = face_vertex_groups[face_keep]
    else:
        base_triangles = mesh_triangles
        base_face_groups = face_vertex_groups

    for render_idx, gid in enumerate(unique_groups):
        pts_mask_full = group_assign == gid
        member_indices = [
            i for i, pgid in enumerate(primitive_groups) if int(pgid) == gid
        ]
        gt_lines: List[str] = []
        if np.any(pts_mask_full):
            label_vals, label_counts = np.unique(
                gt_labels[pts_mask_full], return_counts=True
            )
            order = np.argsort(-label_counts)
            gt_lines = [
                f"{int(label_vals[i])}:{int(label_counts[i])}" for i in order[:5]
            ]

        face_votes = np.sum(base_face_groups == gid, axis=1)
        highlight_mask = face_votes >= 2
        if not np.any(highlight_mask):
            highlight_mask = face_votes >= 1

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

        base_collection = Poly3DCollection(
            base_triangles,
            facecolors="#d7d7d7",
            edgecolors="none",
            alpha=0.32,
        )
        ax.add_collection3d(base_collection)

        if np.any(highlight_mask):
            highlight_collection = Poly3DCollection(
                base_triangles[highlight_mask],
                facecolors="#d62728",
                edgecolors="none",
                alpha=0.95,
            )
            ax.add_collection3d(highlight_collection)

        ax.set_xlim(center[0] - lim, center[0] + lim)
        ax.set_ylim(center[1] - lim, center[1] + lim)
        ax.set_zlim(center[2] - lim, center[2] + lim)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=22, azim=38)
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.savefig(os.path.join(group_dir, f"{render_idx}.png"), dpi=180)
        plt.close(fig)

        txt_path = os.path.join(group_dir, f"{render_idx}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"group_id: {gid}\n")
            f.write(f"step_idx: {step_idx}\n")
            f.write(f"action_name: {action_name}\n")
            f.write(f"num_primitives: {len(member_indices)}\n")
            f.write(f"primitive_indices: {member_indices}\n")
            f.write(f"num_points: {int(np.sum(pts_mask_full))}\n")
            f.write(
                f"dominant_gt_labels: {', '.join(gt_lines) if gt_lines else 'none'}\n"
            )


def save_training_reward_curve(
    out_dir: str,
    epoch_rewards: Sequence[float],
    epoch_indices: Optional[Sequence[int]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if epoch_indices is None:
        epoch_indices = [int(i + 1) for i in range(len(epoch_rewards))]
    if len(epoch_indices) != len(epoch_rewards):
        raise ValueError(
            f"epoch_indices/rewards length mismatch: "
            f"{len(epoch_indices)} vs {len(epoch_rewards)}"
        )
    epoch_indices = [int(x) for x in epoch_indices]
    epoch_rewards = [float(x) for x in epoch_rewards]
    curve_path = os.path.join(out_dir, "ppo_epoch_rewards.json")
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch_indices": epoch_indices,
                "avg_episode_rewards": epoch_rewards,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if plt is None or len(epoch_rewards) == 0:
        return

    xs = np.asarray(epoch_indices, dtype=np.int32)
    ys = np.asarray(epoch_rewards, dtype=np.float64)
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, color="#1f77b4", linewidth=2)
    ax.scatter(xs, ys, color="#1f77b4", s=18)
    ax.set_title("PPO Epoch Reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Episode Reward")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ppo_epoch_rewards.png"), dpi=180)
    plt.close(fig)


def save_experiment_config(
    out_dir: str,
    args: argparse.Namespace,
    global_cfg: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    device: torch.device,
    samples: Sequence[PartNetSample],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "experiment_config.json")
    payload = {
        "mode": args.mode,
        "device": str(device),
        "num_samples": int(len(samples)),
        "sample_ids": [str(sample.sample_id) for sample in samples],
        "args": dict(sorted(vars(args).items())),
        "global_cfg": dict(sorted(global_cfg.items())),
        "env_kwargs": dict(sorted(env_kwargs.items())),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[CONFIG] saved={config_path} mode={args.mode} "
        f"num_samples={len(samples)} out_dir={args.out_dir}"
    )
    return config_path


class SQOutputMixin:
    @staticmethod
    def _group_member_map(group_ids: Sequence[int]) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for idx, gid in enumerate(group_ids):
            groups.setdefault(int(gid), []).append(int(idx))
        return groups

    def _bootstrap_provenance_summary(self) -> Dict[str, Any]:
        """Compact, JSON-safe provenance of the initial-SQ source for exports."""
        info = self.bootstrap_info or {}
        summary: Dict[str, Any] = {
            "bootstrap_source": info.get("bootstrap_source", "mps"),
            "global_optimizer": info.get("global_optimizer", ""),
            "sdf_csv_path": info.get("sdf_csv_path", ""),
            "fallback_single_bbox": bool(info.get("fallback_single_bbox", False)),
            "bootstrap_num_prims": info.get("bootstrap_num_prims", 0.0),
            "sq_label_method": info.get("sq_label_method", "legacy"),
            "sq_label_part_bounds_source": info.get(
                "sq_label_part_bounds_source",
                "",
            ),
            "sq_label_part_bounds_error": info.get("sq_label_part_bounds_error"),
            "sq_label_alignment": info.get("sq_label_alignment"),
            "sq_label_orphan_count": info.get("sq_label_orphan_count", 0),
            "sq_label_cache_key": info.get("sq_label_cache_key", ""),
            "sq_label_cache_status": info.get("sq_label_cache_status", ""),
            "sq_label_cache_npz": info.get("sq_label_cache_npz", ""),
            "sq_label_cache_json": info.get("sq_label_cache_json", ""),
        }
        if info.get("bootstrap_error"):
            summary["bootstrap_error"] = str(info["bootstrap_error"])
        provenance = info.get("provenance")
        if isinstance(provenance, dict):
            for key in (
                "sq_params_path",
                "sq_params_sha256",
                "initial_sq_path",
                "source_mesh",
                "source_mesh_sha256",
                "normalization_center",
                "normalization_scale",
                "raw_num_prims",
                "kept_num_prims",
                "eps_stats",
            ):
                if key in provenance:
                    summary[key] = provenance[key]
        return summary

    def _write_group_export(
        self,
        primitives: Sequence[SQPrimitive],
        group_ids: Sequence[int],
        stem: str,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = os.path.join(self.ep_dir, f"{self.sample.sample_id}_{stem}")
        npy_path = f"{base}.npy"
        ply_path = f"{base}.ply"
        json_path = f"{base}.json"

        if len(primitives) > 0:
            np.save(
                npy_path, np.stack([primitive_to_mps_x(p) for p in primitives], axis=0)
            )
            export_grouped_sq_ply(primitives, group_ids, ply_path)

        payload: Dict[str, Any] = {
            "sample_id": self.sample.sample_id,
            "export_kind": stem,
            "bootstrap": self._bootstrap_provenance_summary(),
            "num_primitives": int(len(primitives)),
            "num_groups": int(len(set(int(g) for g in group_ids))),
            "group_ids": [int(g) for g in group_ids],
            "group_members": {
                str(gid): members
                for gid, members in self._group_member_map(group_ids).items()
            },
            "npy_path": npy_path if len(primitives) > 0 else "",
            "ply_path": ply_path if len(primitives) > 0 else "",
        }
        if extra_payload:
            payload.update(extra_payload)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {
            f"{stem}_npy": npy_path if len(primitives) > 0 else "",
            f"{stem}_ply": ply_path if len(primitives) > 0 else "",
            f"{stem}_json": json_path,
        }

    def _export_grouped_multi_sq_results(
        self,
        primitives: Sequence[SQPrimitive],
        group_ids: Sequence[int],
    ) -> Dict[str, Any]:
        return self._write_group_export(primitives, group_ids, stem="grouped_multi_sq")

    def _final_group_payload(self, final_npy: str, final_ply: str) -> Dict[str, Any]:
        return {
            "sample_id": self.sample.sample_id,
            "export_kind": "final_sq",
            "bootstrap": self._bootstrap_provenance_summary(),
            "num_primitives": int(len(self.primitives)),
            "num_groups": int(len(set(int(g) for g in self.group_ids))),
            "group_ids": [int(g) for g in self.group_ids],
            "group_members": {
                str(gid): members
                for gid, members in self._group_member_map(self.group_ids).items()
            },
            "npy_path": final_npy if len(self.primitives) > 0 else "",
            "ply_path": final_ply if len(self.primitives) > 0 else "",
        }

    def _export_final_mesh_mapping_ply(
        self,
        final_npy: str,
        final_json: str,
        final_ply: str,
        grouped_export_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        if len(self.primitives) == 0:
            payload = self._final_group_payload(final_npy, final_ply)
            with open(final_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return {
                "final_sq_npy": "",
                "final_sq_json": final_json,
                "final_ply": "",
                "final_ply_kind": "mesh_mapping",
                "final_mesh_mapping_error": "no_primitives",
            }

        mesh_path = self.sample.mesh_path or self.sample.ply_path
        if not mesh_path:
            raise FileNotFoundError(
                f"No mesh/PLY surface path available for sample {self.sample.sample_id}"
            )
        grouped_npy = str(grouped_export_info.get("grouped_multi_sq_npy", ""))
        if not grouped_npy:
            raise FileNotFoundError(
                f"No grouped_multi_sq npy available for sample {self.sample.sample_id}"
            )

        np.save(
            final_npy,
            np.stack([primitive_to_mps_x(p) for p in self.primitives], axis=0),
        )
        payload = self._final_group_payload(final_npy, final_ply)
        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        mapping_dir = Path(self.ep_dir) / f"{self.sample.sample_id}_final_mesh_mapping"
        cmd = [
            sys.executable,
            "-m",
            "mesh_mapping",
            "--mesh",
            str(mesh_path),
            "--grouped-sq-npy",
            grouped_npy,
            "--output-dir",
            str(mapping_dir),
            "--min-atom-size",
            "1",
        ]
        if self.final_mesh_mapping_mode == "nearest":
            cmd.append("--save-debug-artifacts")
        subprocess.run(cmd, check=True)
        mapped_flat = mapping_dir / "mesh_mapped_grouped_sq_flat.ply"
        nearest_baseline = mapping_dir / "mesh_nearest_grouped_sq.ply"
        if self.final_mesh_mapping_mode == "nearest":
            selected_ply = nearest_baseline
        else:
            selected_ply = mapped_flat
        if not selected_ply.exists():
            raise FileNotFoundError(f"mesh_mapping did not write {selected_ply}")
        shutil.copyfile(selected_ply, final_ply)

        return {
            "final_sq_npy": final_npy,
            "final_sq_json": final_json,
            "final_ply": final_ply,
            "final_ply_kind": "mesh_mapping",
            "final_mesh_mapping_mode": self.final_mesh_mapping_mode,
            "final_mesh_mapping_command": " ".join(cmd),
            "final_mesh_mapping_dir": str(mapping_dir),
            "final_mesh_mapping_mapped_flat": str(mapped_flat),
            "final_mesh_mapping_nearest_baseline": str(nearest_baseline),
            "final_mesh_mapping_selected_ply": str(selected_ply),
        }
