from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .geometry import primitive_to_mps_x
from .geometry import (
    sq_surface_distance,
    assign_points_to_primitives,
    primitive_rotation_matrix,
    primitive_axis_aligned_bounds,
    aabb_gap_distance,
    aabb_iou,
    aabb_volume,
    aabb_union,
    aabb_contact_ratio,
)

PRIMITIVE_FEATURE_DIM = 17


class SQStateFeaturesMixin:
    def _sample_point_state(self) -> np.ndarray:
        n = self.points.shape[0]
        m = min(self.point_sample_size, n)
        if n > m:
            idx = np.random.choice(n, size=m, replace=False)
        else:
            idx = np.arange(n)
        pts = self.points[idx].astype(np.float32)
        if m < self.point_sample_size:
            pad_pts = np.zeros((self.point_sample_size - m, 3), dtype=np.float32)
            pts = np.concatenate([pts, pad_pts], axis=0)
        return pts

    def _group_member_indices(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for idx, gid in enumerate(self.group_ids):
            groups.setdefault(int(gid), []).append(int(idx))
        return groups

    def _state_geometry(self) -> Dict[str, Any]:
        cached = self.geometry_cache
        if cached is not None:
            return cached

        n = min(len(self.primitives), self.max_prims)
        scene_center = self.points.mean(axis=0)
        scene_radius = float(
            np.max(np.linalg.norm(self.points - scene_center[None, :], axis=1))
        )
        scene_radius = max(scene_radius, 1e-6)

        prim_bounds = [
            primitive_axis_aligned_bounds(prim) for prim in self.primitives[:n]
        ]
        prim_centers = [
            np.asarray(prim.t, dtype=np.float64) for prim in self.primitives[:n]
        ]
        prim_scales = [
            np.exp(np.asarray(prim.log_a, dtype=np.float64))
            for prim in self.primitives[:n]
        ]
        prim_volumes = [float(np.prod(scales)) for scales in prim_scales]
        prim_rotations = [
            primitive_rotation_matrix(prim) for prim in self.primitives[:n]
        ]

        group_members = self._group_member_indices()
        group_bounds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for gid, member_indices in group_members.items():
            bounds = [
                prim_bounds[int(idx)] for idx in member_indices if 0 <= int(idx) < n
            ]
            if not bounds:
                zero = np.zeros(3, dtype=np.float64)
                group_bounds[int(gid)] = (zero, zero)
                continue
            mins = np.stack([b[0] for b in bounds], axis=0)
            maxs = np.stack([b[1] for b in bounds], axis=0)
            group_bounds[int(gid)] = (np.min(mins, axis=0), np.max(maxs, axis=0))

        gids = sorted(group_members.keys())
        neighbor_pairs: set[Tuple[int, int]] = set()
        if len(gids) > 1:
            neighbor_k = min(self.merge_neighbor_k, len(gids) - 1)
            for gid in gids:
                dists: List[Tuple[float, int]] = []
                for other_gid in gids:
                    if other_gid == gid:
                        continue
                    dist = aabb_gap_distance(group_bounds[gid], group_bounds[other_gid])
                    dists.append((float(dist), int(other_gid)))
                dists.sort(key=lambda x: (x[0], x[1]))
                for _, other_gid in dists[:neighbor_k]:
                    neighbor_pairs.add(tuple(sorted((int(gid), int(other_gid)))))

        candidate_pairs: List[Tuple[int, int]] = []
        candidate_action_ids: List[int] = []
        for gid_i, gid_j in sorted(neighbor_pairs):
            pair = self._canonical_merge_pair_for_groups(
                gid_i, gid_j, group_members=group_members
            )
            if pair is None:
                continue
            i, j = pair
            if i < self.max_prims and j < self.max_prims:
                candidate_pairs.append((int(i), int(j)))
                candidate_action_ids.append(
                    int(self.action_catalog.encode("merge", i, j))
                )

        group_centers = {
            gid: 0.5 * (bounds[0] + bounds[1]) for gid, bounds in group_bounds.items()
        }
        group_sizes = {
            gid: len(member_indices) for gid, member_indices in group_members.items()
        }
        group_diags = {
            gid: float(
                np.linalg.norm(np.maximum(bounds[1] - bounds[0], 0.0)) / scene_radius
            )
            for gid, bounds in group_bounds.items()
        }

        cached = {
            "scene_center": scene_center,
            "scene_radius": scene_radius,
            "prim_bounds": prim_bounds,
            "prim_centers": prim_centers,
            "prim_scales": prim_scales,
            "prim_volumes": prim_volumes,
            "prim_rotations": prim_rotations,
            "group_members": group_members,
            "group_bounds": group_bounds,
            "group_centers": group_centers,
            "group_sizes": group_sizes,
            "group_diags": group_diags,
            "neighbor_pairs": neighbor_pairs,
            "candidate_pairs": candidate_pairs,
            "candidate_action_ids": candidate_action_ids,
        }
        self.geometry_cache = cached
        return cached

    def _neighbor_group_pairs(self) -> set[Tuple[int, int]]:
        return set(self._state_geometry()["neighbor_pairs"])

    def _groups_are_merge_neighbors(self, gid_a: int, gid_b: int) -> bool:
        gid_a = int(gid_a)
        gid_b = int(gid_b)
        if gid_a == gid_b:
            return False
        return tuple(sorted((gid_a, gid_b))) in self._neighbor_group_pairs()

    def _canonical_merge_pair_for_groups(
        self,
        gid_a: int,
        gid_b: int,
        group_members: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        gid_a = int(gid_a)
        gid_b = int(gid_b)
        if gid_a == gid_b:
            return None
        members = (
            group_members if group_members is not None else self._group_member_indices()
        )
        members_a = members.get(gid_a, [])
        members_b = members.get(gid_b, [])
        if not members_a or not members_b:
            return None
        rep_a = int(min(members_a))
        rep_b = int(min(members_b))
        return tuple(sorted((rep_a, rep_b)))

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_catalog.action_dim,), dtype=np.float32)
        mask[self.action_catalog.encode("stop")] = 1.0
        for act_id in self._state_geometry()["candidate_action_ids"]:
            mask[int(act_id)] = 1.0
        return mask

    def _soft_assign_matrix(self, soft_assign: Optional[np.ndarray]) -> np.ndarray:
        n = len(self.primitives)
        num_points = int(self.points.shape[0])
        if soft_assign is None or n == 0:
            return np.zeros((n, num_points), dtype=np.float64)
        soft = np.asarray(soft_assign, dtype=np.float64)
        if soft.ndim != 2:
            return np.zeros((n, num_points), dtype=np.float64)
        if soft.shape[0] == n:
            out = soft
        elif soft.shape[1] == n:
            out = soft.T
        else:
            return np.zeros((n, num_points), dtype=np.float64)
        if out.shape[1] != num_points:
            return np.zeros((n, num_points), dtype=np.float64)
        out = np.clip(out[:n], 0.0, 1.0)
        denom = np.maximum(np.sum(out, axis=0, keepdims=True), 1e-12)
        return out / denom

    @staticmethod
    def _point_pca_shape_features(
        points: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] < 3:
            return 0.0, 0.0, 0.0, np.zeros(3, dtype=np.float64)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        cov = centered.T @ centered / max(int(pts.shape[0]), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        l0 = float(max(eigvals[0], 1e-12))
        linearity = float((eigvals[0] - eigvals[1]) / l0)
        planarity = float((eigvals[1] - eigvals[2]) / l0)
        scatter = float(eigvals[2] / l0)
        return linearity, planarity, scatter, eigvecs[:, 0]

    @staticmethod
    def _stride_sample_indices(num_items: int, max_items: int) -> np.ndarray:
        if num_items <= 0:
            return np.zeros((0,), dtype=np.int64)
        if num_items <= max_items:
            return np.arange(num_items, dtype=np.int64)
        stride = int(math.ceil(float(num_items) / float(max(max_items, 1))))
        return np.arange(0, num_items, stride, dtype=np.int64)[:max_items]

    @staticmethod
    def _mean_unit_vector(vectors: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        if vecs.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1)
        vecs = vecs[norms > 1e-6]
        if vecs.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)
        mean_vec = np.mean(vecs, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm <= 1e-8:
            return np.zeros(3, dtype=np.float64)
        return mean_vec / norm

    def _actor_primitive_features(
        self, pred_assign: np.ndarray, soft_assign: Optional[np.ndarray] = None
    ) -> np.ndarray:
        feats = np.zeros((self.max_prims, PRIMITIVE_FEATURE_DIM), dtype=np.float32)
        if len(self.primitives) == 0:
            return feats

        geom = self._state_geometry()
        scene_center = geom["scene_center"]
        scene_radius = float(geom["scene_radius"])
        total_points = max(int(self.points.shape[0]), 1)
        soft = self._soft_assign_matrix(soft_assign)
        group_members = geom["group_members"]
        group_bounds = geom["group_bounds"]
        neighbor_pairs = geom["neighbor_pairs"]
        group_sizes = geom["group_sizes"]
        group_point_ratios = {
            gid: float(
                sum(int((pred_assign == idx).sum()) for idx in member_indices)
                / total_points
            )
            for gid, member_indices in group_members.items()
        }

        group_neighbor_gaps: Dict[int, float] = {}
        for gid, bounds in group_bounds.items():
            gaps = [
                aabb_gap_distance(bounds, other_bounds) / scene_radius
                for other_gid, other_bounds in group_bounds.items()
                if other_gid != gid
            ]
            group_neighbor_gaps[gid] = float(min(gaps)) if gaps else 1.0

        for k, prim in enumerate(self.primitives[: self.max_prims]):
            count_ratio = float(np.mean(pred_assign == k))
            if len(self.primitives) > 1:
                dists = []
                center_k = np.asarray(prim.t, dtype=np.float64)
                for j, other in enumerate(self.primitives):
                    if j == k:
                        continue
                    d = float(
                        np.linalg.norm(center_k - np.asarray(other.t, dtype=np.float64))
                        / scene_radius
                    )
                    dists.append(d)
                nearest_dist = float(min(dists)) if dists else 1.0
            else:
                nearest_dist = 1.0
            gid = int(self.group_ids[k])
            group_size = float(group_sizes.get(gid, 1) / max(1, self.max_prims))
            gmin, gmax = group_bounds.get(
                gid, (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
            )
            group_diag_norm = float(
                np.linalg.norm(np.maximum(gmax - gmin, 0.0)) / scene_radius
            )
            group_center = 0.5 * (gmin + gmax)
            group_center_norm = float(
                np.linalg.norm(group_center - scene_center) / scene_radius
            )
            group_point_ratio = float(group_point_ratios.get(gid, count_ratio))
            nearest_group_gap = float(group_neighbor_gaps.get(gid, 1.0))

            merge_degree = 0.0
            if len(self.primitives) > 1:
                valid_merge = 0
                for j in range(len(self.primitives)):
                    if j == k:
                        continue
                    pair = tuple(sorted((gid, int(self.group_ids[j]))))
                    if gid != int(self.group_ids[j]) and pair in neighbor_pairs:
                        valid_merge += 1
                merge_degree = float(valid_merge / max(len(self.primitives) - 1, 1))

            pts_k = self.points[pred_assign == k]
            assign_entropy = 0.0
            assign_confidence = 0.0
            assign_boundary_ratio = 0.0
            assign_second_gap = 0.0
            pca_linearity = 0.0
            pca_planarity = 0.0
            pca_scatter = 0.0
            sq_pca_axis_alignment = 0.0
            mean_surface_dist = 0.0
            p90_surface_dist = 0.0
            if pts_k.shape[0] > 0:
                if soft.shape[0] > 0:
                    soft_k = soft[:, pred_assign == k]
                    if soft_k.size > 0:
                        sorted_soft = np.sort(soft_k, axis=0)[::-1]
                        top1 = sorted_soft[0]
                        top2 = (
                            sorted_soft[1]
                            if sorted_soft.shape[0] > 1
                            else np.zeros_like(top1)
                        )
                        assign_confidence = float(np.mean(top1))
                        gaps = top1 - top2
                        assign_second_gap = float(np.mean(gaps))
                        assign_boundary_ratio = float(np.mean(gaps < 0.20))
                        denom = max(math.log(max(soft_k.shape[0], 2)), 1e-6)
                        entropy = (
                            -np.sum(
                                soft_k * np.log(np.clip(soft_k, 1e-12, 1.0)), axis=0
                            )
                            / denom
                        )
                        assign_entropy = float(np.mean(entropy))

                (
                    pca_linearity,
                    pca_planarity,
                    pca_scatter,
                    pca_axis,
                ) = self._point_pca_shape_features(pts_k)
                if np.linalg.norm(pca_axis) > 1e-8:
                    rot_align = np.abs(primitive_rotation_matrix(prim).T @ pca_axis)
                    sq_pca_axis_alignment = float(np.max(rot_align))

                residual_pts = pts_k
                if residual_pts.shape[0] > 512:
                    stride = int(math.ceil(residual_pts.shape[0] / 512.0))
                    residual_pts = residual_pts[::stride]
                try:
                    dists = sq_surface_distance(residual_pts, prim) / scene_radius
                    mean_surface_dist = float(np.mean(dists))
                    p90_surface_dist = float(np.quantile(dists, 0.90))
                except Exception:
                    mean_surface_dist = 0.0
                    p90_surface_dist = 0.0

            feats[k] = np.asarray(
                [
                    nearest_dist,
                    group_size,
                    group_diag_norm,
                    group_center_norm,
                    group_point_ratio,
                    merge_degree,
                    nearest_group_gap,
                    assign_entropy,
                    assign_confidence,
                    assign_boundary_ratio,
                    assign_second_gap,
                    pca_linearity,
                    pca_planarity,
                    pca_scatter,
                    sq_pca_axis_alignment,
                    mean_surface_dist,
                    p90_surface_dist,
                ],
                dtype=np.float32,
            )
        return feats

    def _actor_global_features(
        self, action_mask: np.ndarray, prim_features: np.ndarray
    ) -> np.ndarray:
        n = len(self.primitives)
        num_groups = len(set(self.group_ids))
        step_progress = float(self.steps / max(self.max_steps, 1))
        prim_fill_ratio = float(n / max(self.max_prims, 1))
        group_fill_ratio = float(num_groups / max(self.max_prims, 1))
        group_progress = float(num_groups / max(self.init_num_groups, 1.0))

        group_members = self._group_member_indices()
        singleton_group_frac = float(
            sum(1 for members in group_members.values() if len(members) == 1)
            / max(len(group_members), 1)
        )
        avg_group_size_ratio = (
            float(
                np.mean([len(members) for members in group_members.values()])
                / max(self.max_prims, 1)
            )
            if group_members
            else 0.0
        )

        merge_count = 0.0
        for i in range(min(n, self.max_prims)):
            for j in range(i + 1, min(n, self.max_prims)):
                merge_count += float(
                    action_mask[self.action_catalog.encode("merge", i, j)]
                )
        merge_ratio = float(merge_count / max(n * max(n - 1, 1) / 2.0, 1.0))

        mean_merge_degree = float(np.mean(prim_features[:n, 5])) if n > 0 else 0.0

        return np.asarray(
            [
                step_progress,
                prim_fill_ratio,
                group_fill_ratio,
                group_progress,
                singleton_group_frac,
                avg_group_size_ratio,
                merge_ratio,
                mean_merge_degree,
            ],
            dtype=np.float32,
        )

    def _privileged_global_features(self) -> np.ndarray:
        if not self.has_labels or self.num_gt_parts <= 0:
            return np.zeros((4,), dtype=np.float32)
        num_groups = float(len(set(self.group_ids)))
        target = max(float(self.num_gt_parts), 1.0)
        signed_group_delta = float((num_groups - target) / target)
        abs_group_error = float(abs(num_groups - target) / target)
        return np.asarray(
            [
                float(target / max(self.max_prims, 1)),
                signed_group_delta,
                abs_group_error,
                float(int(num_groups) == int(target)),
            ],
            dtype=np.float32,
        )

    def _target_group_count_ratio_target(self) -> np.ndarray:
        if not self.has_labels or self.num_gt_parts <= 0:
            return np.asarray([0.0], dtype=np.float32)
        return np.asarray(
            [float(max(self.num_gt_parts, 1) / max(self.max_prims, 1))],
            dtype=np.float32,
        )

    def _actor_merge_action_features(
        self,
        action_mask: np.ndarray,
        pred_assign: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_candidates = max(1, self.max_prims * self.merge_neighbor_k)
        feats = np.zeros((max_candidates, 17), dtype=np.float32)
        pair_indices = np.zeros((max_candidates, 2), dtype=np.float32)
        pair_action_ids = np.zeros((max_candidates,), dtype=np.float32)
        pair_mask = np.zeros((max_candidates,), dtype=np.float32)
        n = min(len(self.primitives), self.max_prims)
        if n <= 1:
            return feats, pair_indices, pair_action_ids, pair_mask

        geom = self._state_geometry()
        scene_radius = float(geom["scene_radius"])
        prim_bounds = geom["prim_bounds"]
        prim_centers = geom["prim_centers"]
        prim_scales = geom["prim_scales"]
        prim_volumes = geom["prim_volumes"]
        prim_rotations = geom["prim_rotations"]
        group_members = geom["group_members"]
        group_bounds = geom["group_bounds"]
        group_sizes = geom["group_sizes"]
        group_centers = geom["group_centers"]
        total_points = max(int(self.points.shape[0]), 1)
        group_point_ratios = {
            gid: float(
                sum(int((pred_assign == idx).sum()) for idx in member_indices)
                / total_points
            )
            for gid, member_indices in group_members.items()
        }
        group_diags = geom["group_diags"]
        group_pca_axes: Dict[int, np.ndarray] = {}
        group_mean_normals: Dict[int, np.ndarray] = {}
        for gid, member_indices in group_members.items():
            point_mask = np.zeros((self.points.shape[0],), dtype=bool)
            for idx in member_indices:
                point_mask |= pred_assign == int(idx)
            pts = self.points[point_mask]
            normals = (
                self.point_normals[point_mask]
                if self.has_point_normals
                else np.zeros((0, 3), dtype=np.float32)
            )

            pca_idx = self._stride_sample_indices(int(pts.shape[0]), 512)
            pca_pts = pts[pca_idx] if pca_idx.size > 0 else pts
            _, _, _, pca_axis = self._point_pca_shape_features(pca_pts)
            group_pca_axes[int(gid)] = pca_axis

            if self.has_point_normals and normals.shape[0] > 0:
                normal_idx = self._stride_sample_indices(int(normals.shape[0]), 256)
                group_mean_normals[int(gid)] = self._mean_unit_vector(
                    normals[normal_idx]
                )
            else:
                group_mean_normals[int(gid)] = np.zeros(3, dtype=np.float64)

        candidate_pairs = geom["candidate_pairs"]
        candidate_action_ids = geom["candidate_action_ids"]
        for slot, (i, j) in enumerate(candidate_pairs[:max_candidates]):
            gid_i = int(self.group_ids[i])
            gid_j = int(self.group_ids[j])
            center_dist = float(
                np.linalg.norm(prim_centers[i] - prim_centers[j]) / scene_radius
            )
            gap = float(
                aabb_gap_distance(prim_bounds[i], prim_bounds[j]) / scene_radius
            )
            scale_sim = float(
                np.exp(
                    -np.linalg.norm(
                        np.log(np.maximum(prim_scales[i], 1e-6))
                        - np.log(np.maximum(prim_scales[j], 1e-6))
                    )
                )
            )
            vol_ratio = float(
                min(prim_volumes[i], prim_volumes[j])
                / max(max(prim_volumes[i], prim_volumes[j]), 1e-6)
            )
            rot_align = np.abs(prim_rotations[i].T @ prim_rotations[j])
            orient_sim = float(np.mean(np.max(rot_align, axis=1)))
            combined_group_size = float(
                (group_sizes.get(gid_i, 1) + group_sizes.get(gid_j, 1))
                / max(self.max_prims, 1)
            )
            group_center_gap = float(
                np.linalg.norm(
                    group_centers.get(gid_i, prim_centers[i])
                    - group_centers.get(gid_j, prim_centers[j])
                )
                / scene_radius
            )
            bounds_i = group_bounds.get(gid_i, prim_bounds[i])
            bounds_j = group_bounds.get(gid_j, prim_bounds[j])
            group_bbox_iou = float(aabb_iou(bounds_i, bounds_j))
            ratio_i = float(group_point_ratios.get(gid_i, 0.0))
            ratio_j = float(group_point_ratios.get(gid_j, 0.0))
            group_point_ratio_sim = float(
                min(ratio_i, ratio_j) / max(max(ratio_i, ratio_j), 1e-12)
            )
            diag_i = float(group_diags.get(gid_i, 0.0))
            diag_j = float(group_diags.get(gid_j, 0.0))
            group_diag_ratio = float(
                min(diag_i, diag_j) / max(max(diag_i, diag_j), 1e-12)
            )
            group_aabb_gap = float(aabb_gap_distance(bounds_i, bounds_j) / scene_radius)
            union_bounds = aabb_union(bounds_i, bounds_j)
            vol_i = aabb_volume(bounds_i)
            vol_j = aabb_volume(bounds_j)
            union_vol = aabb_volume(union_bounds)
            bbox_volume_expansion = float(
                (union_vol - vol_i - vol_j) / max(vol_i + vol_j, 1e-12)
            )
            union_diag = float(
                np.linalg.norm(np.maximum(union_bounds[1] - union_bounds[0], 0.0))
                / scene_radius
            )
            bbox_diag_growth = float(union_diag - max(diag_i, diag_j))
            contact_ratio = float(aabb_contact_ratio(bounds_i, bounds_j))
            bbox_volume_expansion = float(np.clip(bbox_volume_expansion, -1.0, 10.0))
            bbox_diag_growth = float(np.clip(bbox_diag_growth, -1.0, 5.0))

            pca_i = group_pca_axes.get(gid_i, np.zeros(3, dtype=np.float64))
            pca_j = group_pca_axes.get(gid_j, np.zeros(3, dtype=np.float64))
            if np.linalg.norm(pca_i) > 1e-8 and np.linalg.norm(pca_j) > 1e-8:
                pca_axis_alignment = float(abs(np.dot(pca_i, pca_j)))
            else:
                pca_axis_alignment = 0.0
            normal_i = group_mean_normals.get(gid_i, np.zeros(3, dtype=np.float64))
            normal_j = group_mean_normals.get(gid_j, np.zeros(3, dtype=np.float64))
            if (
                self.has_point_normals
                and np.linalg.norm(normal_i) > 1e-8
                and np.linalg.norm(normal_j) > 1e-8
            ):
                normal_alignment = float(abs(np.dot(normal_i, normal_j)))
            else:
                normal_alignment = 0.0
            act_id = int(candidate_action_ids[slot])
            valid = float(action_mask[act_id] > 0.0)
            feats[slot] = np.asarray(
                [
                    valid,
                    center_dist,
                    gap,
                    scale_sim,
                    vol_ratio,
                    orient_sim,
                    combined_group_size,
                    group_center_gap,
                    group_bbox_iou,
                    group_point_ratio_sim,
                    group_diag_ratio,
                    group_aabb_gap,
                    bbox_volume_expansion,
                    bbox_diag_growth,
                    contact_ratio,
                    pca_axis_alignment,
                    normal_alignment,
                ],
                dtype=np.float32,
            )
            pair_indices[slot] = np.asarray([i, j], dtype=np.float32)
            pair_action_ids[slot] = float(act_id)
            pair_mask[slot] = valid
        return feats, pair_indices, pair_action_ids, pair_mask

    def _privileged_primitive_features(self, pred_assign: np.ndarray) -> np.ndarray:
        feats = np.zeros((self.max_prims, 6), dtype=np.float32)
        if len(self.primitives) == 0 or not self.has_labels or self.num_gt_parts <= 0:
            return feats

        total_points = max(int(self.points.shape[0]), 1)
        denom_entropy = max(math.log(max(int(self.num_gt_parts), 2)), 1e-6)
        for k in range(min(len(self.primitives), self.max_prims)):
            pts_mask = pred_assign == k
            count = int(pts_mask.sum())
            count_ratio = float(count / total_points)
            if count <= 0:
                feats[k, 0] = count_ratio
                continue
            labels = self.part_labels[pts_mask]
            labels = labels[labels >= 0]
            if labels.size == 0:
                feats[k, 0] = count_ratio
                continue
            _, cnt = np.unique(labels, return_counts=True)
            probs = cnt.astype(np.float64) / max(float(cnt.sum()), 1.0)
            probs_sorted = np.sort(probs)[::-1]
            purity = float(probs_sorted[0])
            second_share = float(probs_sorted[1]) if probs_sorted.size >= 2 else 0.0
            unique_frac = float(len(cnt) / max(int(self.num_gt_parts), 1))
            entropy = float(
                -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / denom_entropy
            )
            fragmented = float(1.0 - purity)
            feats[k] = np.asarray(
                [count_ratio, purity, second_share, unique_frac, entropy, fragmented],
                dtype=np.float32,
            )
        return feats

    def _state(
        self,
        pred_assign: Optional[np.ndarray] = None,
        soft_assign: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        pts_sample = self._sample_point_state()
        sq = np.zeros((self.max_prims, 11), dtype=np.float32)
        sq_mask = np.zeros((self.max_prims,), dtype=np.float32)
        for i, prim in enumerate(self.primitives[: self.max_prims]):
            sq[i] = primitive_to_mps_x(prim).astype(np.float32)
            sq_mask[i] = 1.0

        if pred_assign is None:
            pred_assign, soft_assign = assign_points_to_primitives(
                self.primitives, self.points
            )
        action_mask = self._action_mask()
        prim_features = self._actor_primitive_features(pred_assign, soft_assign)
        global_features = self._actor_global_features(action_mask, prim_features)
        (
            merge_action_features,
            merge_pair_indices,
            merge_pair_action_ids,
            merge_pair_mask,
        ) = self._actor_merge_action_features(action_mask, pred_assign)
        if self.has_labels:
            privileged_prim_stats = self._privileged_primitive_features(pred_assign)
            privileged_global_features = self._privileged_global_features()
            target_group_count_ratio = self._target_group_count_ratio_target()
        else:
            privileged_prim_stats = np.zeros((self.max_prims, 6), dtype=np.float32)
            privileged_global_features = np.zeros((4,), dtype=np.float32)
            target_group_count_ratio = np.zeros((1,), dtype=np.float32)

        state = {
            "sq_params": sq,
            "sq_mask": sq_mask,
            "point_sample": pts_sample,
            "action_mask": action_mask,
            "prim_features": prim_features,
            "global_features": global_features,
            "merge_action_features": merge_action_features,
            "merge_pair_indices": merge_pair_indices,
            "merge_pair_action_ids": merge_pair_action_ids,
            "merge_pair_mask": merge_pair_mask,
            "privileged_prim_stats": privileged_prim_stats,
            "privileged_global_features": privileged_global_features,
            "target_group_count_ratio": target_group_count_ratio,
        }
        return state
