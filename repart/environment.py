from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .geometry import SQPrimitive
from .supervision import (
    SQCoverageLabelResult,
    assign_sq_labels_by_coverage,
    coverage_cache_key,
    load_sq_coverage_cache,
    save_sq_coverage_cache,
)
from .data import PartNetSample, read_pts_file, read_pts_normals_file, read_label_file
from .geometry import (
    assign_points_to_primitives,
    compute_eval_segmentation_metrics,
    grouped_assignments_from_membership,
)
from .data import bootstrap_primitives, load_cached_global_sdf
from .outputs import SQOutputMixin
from .state_features import SQStateFeaturesMixin
from .outputs import save_group_highlight_renders


class ActionCatalog:
    def __init__(self, max_prims: int):
        self.max_prims = int(max_prims)
        self.id_to_action: List[Tuple[str, Tuple[int, ...]]] = []
        self.action_to_id: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self._build()

    def _add(self, name: str, *args: int) -> None:
        idx = len(self.id_to_action)
        key = (name, tuple(args))
        self.id_to_action.append(key)
        self.action_to_id[key] = idx

    def _build(self) -> None:
        self._add("stop")
        for i in range(self.max_prims):
            for j in range(i + 1, self.max_prims):
                self._add("merge", i, j)

    @property
    def action_dim(self) -> int:
        return len(self.id_to_action)

    def decode(self, action_id: int) -> Tuple[str, Tuple[int, ...]]:
        return self.id_to_action[int(action_id)]

    def encode(self, name: str, *args: int) -> int:
        return self.action_to_id[(name, tuple(args))]


class SQEditEnv(SQStateFeaturesMixin, SQOutputMixin):
    def __init__(
        self,
        sample: PartNetSample,
        action_catalog: ActionCatalog,
        out_dir: str,
        max_prims: int = 12,
        max_steps: int = 8,
        point_sample_size: int = 1024,
        global_cfg: Optional[Dict[str, Any]] = None,
        render_group_highlights: bool = False,
        episode_idx: int = 0,
        invalid_action_penalty: float = 0.0,
        stop_reward_bonus: float = 0.01,
        group_excess_penalty_w: float = 0.0,
        stop_group_penalty_w: float = 0.0,
        merge_neighbor_k: int = 2,
        sq_label_method: str = "coverage",
        final_mesh_mapping_mode: str = "atom",
    ):
        self.sample = sample
        self.action_catalog = action_catalog
        self.out_dir = out_dir
        self.summary_only_rollout = (
            os.path.basename(os.path.normpath(out_dir)) == "ppo_rollouts"
        )
        self.max_prims = int(max_prims)
        self.max_steps = int(max_steps)
        self.point_sample_size = int(point_sample_size)
        self.global_cfg = dict(global_cfg or {})
        self.render_group_highlights = bool(render_group_highlights)
        self.episode_idx = int(episode_idx) - 1
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.stop_reward_bonus = float(stop_reward_bonus)
        self.group_excess_penalty_w = float(group_excess_penalty_w)
        self.stop_group_penalty_w = float(stop_group_penalty_w)
        self.merge_neighbor_k = max(1, int(merge_neighbor_k))
        self.sq_label_method = str(sq_label_method).strip().lower()
        if self.sq_label_method not in {"coverage", "legacy"}:
            raise ValueError(
                f"Unsupported sq_label_method={sq_label_method!r}; "
                "expected 'coverage' or 'legacy'."
            )
        self.final_mesh_mapping_mode = str(final_mesh_mapping_mode).strip().lower()
        if self.final_mesh_mapping_mode not in {"atom", "nearest"}:
            raise ValueError(
                f"Unsupported final_mesh_mapping_mode={final_mesh_mapping_mode!r}; "
                "expected 'atom' or 'nearest'."
            )

        self.points = read_pts_file(sample.pts_path)
        self.point_normals = read_pts_normals_file(sample.pts_path)
        self.has_point_normals = (
            self.point_normals is not None
            and self.point_normals.shape[0] == self.points.shape[0]
        )
        if not self.has_point_normals:
            self.point_normals = np.zeros_like(self.points, dtype=np.float32)
        self.has_labels = bool(sample.label_path) and os.path.exists(
            str(sample.label_path)
        )
        if self.has_labels:
            self.part_labels = read_label_file(str(sample.label_path))
            if self.part_labels.shape[0] != self.points.shape[0]:
                raise ValueError(
                    f"Point/label count mismatch for {sample.sample_id}: {self.points.shape[0]} vs {self.part_labels.shape[0]}"
                )
            self.num_gt_parts = int(
                np.unique(self.part_labels[self.part_labels >= 0]).size
            )
        else:
            self.part_labels = np.full((self.points.shape[0],), -1, dtype=np.int64)
            self.num_gt_parts = 0

        self.steps = 0
        self.primitives: List[SQPrimitive] = []
        self.group_ids: List[int] = []
        self.next_group_id: int = 0
        self.init_part_miou = 0.0
        self.init_num_groups = 0.0
        self.prev_part_miou = 0.0
        self.bootstrap_info: Dict[str, Any] = {}
        self.episode_records: List[Dict[str, Any]] = []
        self.geometry_cache: Optional[Dict[str, Any]] = None

        self.sdf_global = None
        self.sdf_grid = None
        self.sq_gt_labels = np.zeros((0,), dtype=np.int64)
        self.sq_gt_label_coverage = np.zeros((0,), dtype=np.float64)
        self.sq_gt_label_result: Optional[SQCoverageLabelResult] = None

    def _invalidate_geometry_cache(self) -> None:
        self.geometry_cache = None

    def reset(self) -> Dict[str, np.ndarray]:
        source_limit = getattr(self.sample, "_joint_source_max_prims", None)
        joint_limit = self.max_prims
        if source_limit is None or int(source_limit) >= joint_limit:
            return self._reset_current_limit()

        # Preserve the source run's SQ truncation and cache labels, then emit
        # state tensors at the joint policy width.
        self.max_prims = int(source_limit)
        try:
            self._reset_current_limit()
        finally:
            self.max_prims = joint_limit
        return self._state()

    def _reset_current_limit(self) -> Dict[str, np.ndarray]:
        self.episode_idx += 1
        self.steps = 0
        self._invalidate_geometry_cache()
        ep_dir = os.path.join(
            self.out_dir, self.sample.sample_id, f"episode_{self.episode_idx:04d}"
        )
        os.makedirs(ep_dir, exist_ok=True)
        self.ep_dir = ep_dir
        self.episode_records = []
        self.primitives, self.bootstrap_info = bootstrap_primitives(
            self.sample,
            ep_dir,
            self.global_cfg,
            write_outputs=not self.summary_only_rollout,
        )
        self.primitives = self.primitives[: self.max_prims]
        self.group_ids = list(range(len(self.primitives)))
        self.next_group_id = len(self.group_ids)
        self.sdf_global, self.sdf_grid = load_cached_global_sdf(
            self.sample, self.bootstrap_info.get("sdf_csv_path") or None
        )
        self.sq_gt_labels = np.zeros((0,), dtype=np.int64)
        self.sq_gt_label_coverage = np.zeros((0,), dtype=np.float64)
        self.sq_gt_label_result = None
        if self.sq_label_method == "coverage" and self.has_labels:
            label_config = {
                "surface_samples": 200,
                "seed": int(self.global_cfg.get("seed", 42)),
                "scene_distance_ratio": 0.065,
                "sq_min_axis_ratio": 0.35,
                "spacing_multiplier": 4.0,
                "min_near_points": 3,
            }
            coverage_source = "mps"
            sample_dir = Path(self.sample.pts_path).parent.parent
            label_cache_key = coverage_cache_key(
                self.primitives,
                self.points,
                self.part_labels,
                sample_id=self.sample.sample_id,
                bootstrap_source=coverage_source,
                sample_dir=sample_dir,
                sdf_csv_path=(self.bootstrap_info.get("sdf_csv_path") or None),
                **label_config,
            )
            cache_dir_value = self.global_cfg.get("sq_label_cache_dir")
            cache_dir = (
                Path(cache_dir_value)
                if cache_dir_value
                else Path(self.out_dir).parent / "sq_label_cache"
            )
            label_result: Optional[SQCoverageLabelResult] = None
            cache_status = "miss"
            cache_paths: Dict[str, str] = {
                "npz_path": str(cache_dir / f"{self.sample.sample_id}.npz"),
                "json_path": str(cache_dir / f"{self.sample.sample_id}.json"),
            }
            if (
                self.sample.cached_sq_label_key == label_cache_key
                and self.sample.cached_sq_label_result is not None
            ):
                label_result = self.sample.cached_sq_label_result
                cache_status = "memory_hit"
            if label_result is None:
                trust_reused_label_cache = bool(
                    self.global_cfg.get("trust_reused_sq_label_cache", False)
                )
                label_result, disk_cache_status = load_sq_coverage_cache(
                    cache_dir,
                    sample_id=self.sample.sample_id,
                    expected_cache_key=(
                        None if trust_reused_label_cache else label_cache_key
                    ),
                )
                cache_status = (
                    f"disk_trusted_{disk_cache_status}"
                    if trust_reused_label_cache and label_result is not None
                    else f"disk_{disk_cache_status}"
                )

            # A trusted cache may have been generated from a different
            # bootstrap primitive set.  In particular, joint fine-tuning can
            # reuse the source cache while regenerating MPS for samples that
            # have no saved initial_sq.npy.  Never apply labels from that old
            # primitive set to the current primitives: recompute the cheap
            # coverage labels below and replace the staged cache entry.
            if label_result is not None and (
                label_result.labels.shape[0] != len(self.primitives)
            ):
                cached_count = int(label_result.labels.shape[0])
                current_count = int(len(self.primitives))
                print(
                    f"[sq-label-cache] WARNING sample={self.sample.sample_id} "
                    f"primitive-count mismatch: cached={cached_count} "
                    f"current={current_count}; recomputing labels for the "
                    "current bootstrap primitives"
                )
                label_result = None
                cache_status = (
                    f"{cache_status}_primitive_count_mismatch:"
                    f"{cached_count}!={current_count}"
                )
            if label_result is None:
                label_result = assign_sq_labels_by_coverage(
                    self.primitives,
                    self.points,
                    self.part_labels,
                    sample_id=self.sample.sample_id,
                    bootstrap_source=coverage_source,
                    sdf_values=self.sdf_global,
                    sdf_grid=self.sdf_grid,
                    sample_dir=sample_dir,
                    **label_config,
                )
                label_result.cache_key = label_cache_key
                try:
                    cache_paths = save_sq_coverage_cache(
                        label_result,
                        cache_dir,
                        sample_id=self.sample.sample_id,
                        cache_key=label_cache_key,
                        bootstrap_source=coverage_source,
                    )
                    cache_status = f"generated_after_{cache_status}"
                except Exception as exc:
                    cache_status = (
                        f"generated_cache_write_error:" f"{type(exc).__name__}:{exc}"
                    )
                    print(
                        f"[sq-label-cache] WARNING sample="
                        f"{self.sample.sample_id} {cache_status}"
                    )
            self.sample.cached_sq_label_key = label_cache_key
            self.sample.cached_sq_label_result = label_result
            if label_result.labels.shape[0] != len(self.primitives):
                raise RuntimeError(
                    f"Coverage SQ-label count mismatch for {self.sample.sample_id}: "
                    f"{label_result.labels.shape[0]} labels for "
                    f"{len(self.primitives)} primitives"
                )
            self.sq_gt_label_result = label_result
            self.sq_gt_labels = label_result.labels.copy()
            self.sq_gt_label_coverage = label_result.coverage.copy()
            self.bootstrap_info["sq_label_method"] = "coverage"
            self.bootstrap_info[
                "sq_label_part_bounds_source"
            ] = label_result.part_bounds_source
            self.bootstrap_info[
                "sq_label_part_bounds_error"
            ] = label_result.part_bounds_error
            self.bootstrap_info["sq_label_alignment"] = dict(label_result.alignment)
            self.bootstrap_info["sq_label_orphan_count"] = int(
                label_result.orphan_mask.sum()
            )
            self.bootstrap_info["sq_label_cache_key"] = label_cache_key
            self.bootstrap_info["sq_label_cache_status"] = cache_status
            if cache_paths:
                self.bootstrap_info["sq_label_cache_npz"] = cache_paths["npz_path"]
                self.bootstrap_info["sq_label_cache_json"] = cache_paths["json_path"]
            if label_result.part_bounds_error:
                print(
                    f"[sq-label] WARNING sample={self.sample.sample_id} "
                    f"part_bounds_source={label_result.part_bounds_source} "
                    f"reason={label_result.part_bounds_error}"
                )
        else:
            self.bootstrap_info["sq_label_method"] = self.sq_label_method
        pred_assign, soft_assign = assign_points_to_primitives(
            self.primitives, self.points
        )
        if self.has_labels:
            grouped_pred = grouped_assignments_from_membership(
                pred_assign, self.group_ids, len(set(self.group_ids))
            )
            init_seg_metrics = compute_eval_segmentation_metrics(
                grouped_pred, self.part_labels
            )
            self.init_part_miou = float(init_seg_metrics["part_miou"])
        else:
            self.init_part_miou = 0.0
        self.init_num_groups = float(len(set(self.group_ids)))
        self.prev_part_miou = self.init_part_miou
        return self._state(pred_assign=pred_assign, soft_assign=soft_assign)

    def _normalized_group_excess(self, num_groups: float) -> float:
        if not self.has_labels or self.num_gt_parts <= 0:
            return 0.0
        # Kept under the old "excess" name for log/checkpoint compatibility.
        # The value is symmetric: both too many and too few groups are penalized.
        target = max(float(self.num_gt_parts), 1.0)
        return abs(float(num_groups) - target) / target

    def _merge(self, i: int, j: int) -> Tuple[bool, List[int], str]:
        if not (
            0 <= i < len(self.primitives) and 0 <= j < len(self.primitives) and i < j
        ):
            return False, [], "merge_invalid"
        if self.group_ids[i] == self.group_ids[j]:
            return False, [], "merge_same_group"
        gid_i = self.group_ids[i]
        gid_j = self.group_ids[j]
        if not self._groups_are_merge_neighbors(int(gid_i), int(gid_j)):
            return False, [], "merge_non_neighbor_group"
        if self._canonical_merge_pair_for_groups(int(gid_i), int(gid_j)) != (
            int(i),
            int(j),
        ):
            return False, [], "merge_non_canonical_group_pair"
        for idx, gid in enumerate(self.group_ids):
            if gid == gid_j:
                self.group_ids[idx] = gid_i
        self._invalidate_geometry_cache()
        return True, [i, j], "merge_group_ok"

    def step(
        self, action_id: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.steps += 1
        name, args = self.action_catalog.decode(action_id)
        pred_before, soft_before = assign_points_to_primitives(
            self.primitives, self.points
        )
        if self.has_labels:
            grouped_before = grouped_assignments_from_membership(
                pred_before, self.group_ids, len(set(self.group_ids))
            )
            before_seg_metrics = compute_eval_segmentation_metrics(
                grouped_before, self.part_labels
            )
            before_part = float(before_seg_metrics["part_miou"])
        else:
            grouped_before = np.full((self.points.shape[0],), -1, dtype=np.int64)
            before_part = 0.0
        before_num_groups = float(len(set(self.group_ids)))

        if name == "stop":
            success, _, msg = True, [], "stop"
            done = True
        elif name == "merge":
            success, _, msg = self._merge(args[0], args[1])
            done = False
        else:
            success, _, msg = False, [], "unknown_action"
            done = False

        if self.steps >= self.max_steps:
            done = True

        pred_after, soft_after = pred_before, soft_before
        grouped_after = (
            grouped_assignments_from_membership(
                pred_after, self.group_ids, len(set(self.group_ids))
            )
            if len(self.primitives) > 0
            else np.full((self.points.shape[0],), -1, dtype=np.int64)
        )
        if self.has_labels:
            after_seg_metrics = compute_eval_segmentation_metrics(
                grouped_after, self.part_labels
            )
            after_part = float(after_seg_metrics["part_miou"])
        else:
            after_part = 0.0
        after_num_groups = float(len(set(self.group_ids)))
        before_group_excess = self._normalized_group_excess(before_num_groups)
        after_group_excess = self._normalized_group_excess(after_num_groups)
        group_count_reward = self.group_excess_penalty_w * (
            before_group_excess - after_group_excess
        )
        stop_group_penalty = 0.0

        if not self.has_labels:
            reward = 0.0
        else:
            reward = after_part - before_part + group_count_reward
            if not success:
                reward += self.invalid_action_penalty
        if self.has_labels and success and name == "stop":
            stop_group_penalty = self.stop_group_penalty_w * after_group_excess
            reward -= stop_group_penalty
            reward += self.stop_reward_bonus

        self.prev_part_miou = after_part

        if (
            self.render_group_highlights
            and self.has_labels
            and name == "stop"
            and not self.summary_only_rollout
        ):
            save_group_highlight_renders(
                self.ep_dir,
                self.sample,
                self.sample.sample_id,
                self.steps,
                self.points,
                grouped_after,
                self.group_ids,
                name,
                self.part_labels,
            )

        info: Dict[str, Any] = {
            "sample_id": self.sample.sample_id,
            "has_labels": float(self.has_labels),
            "step_idx": int(self.steps),
            "action_name": name,
            "action_msg": msg,
            "action_success": float(success),
            "init_part_miou": float(self.init_part_miou),
            "before_part_miou": float(before_part),
            "after_part_miou": float(after_part),
            "num_primitives": float(len(self.primitives)),
            "num_groups": float(after_num_groups),
            "reward": float(reward),
            "done": float(done),
            "group_excess_before": float(before_group_excess),
            "group_excess_after": float(after_group_excess),
            "group_count_error_before": float(before_group_excess),
            "group_count_error_after": float(after_group_excess),
            "group_count_reward": float(group_count_reward),
            "stop_group_penalty": float(stop_group_penalty),
            "final_part_miou": float(after_part),
            "final_part_miou_source": "final_rl_group_point_assignment",
            "final_num_primitives": float(len(self.primitives)),
            "final_num_groups": float(after_num_groups),
        }
        self.episode_records.append(dict(info))

        if done:
            grouped_export_info: Dict[str, Any] = {}
            if not self.summary_only_rollout:
                grouped_multi_prims = [p.clone() for p in self.primitives]
                grouped_multi_group_ids = [int(g) for g in self.group_ids]
                grouped_export_info = self._export_grouped_multi_sq_results(
                    grouped_multi_prims, grouped_multi_group_ids
                )
                info.update(grouped_export_info)
                self.episode_records[-1].update(grouped_export_info)

            final_part_miou = float(after_part)
            if not self.summary_only_rollout:
                final_ply = os.path.join(
                    self.ep_dir, f"{self.sample.sample_id}_final_sq.ply"
                )
                final_npy = os.path.join(
                    self.ep_dir, f"{self.sample.sample_id}_final_sq.npy"
                )
                final_json = os.path.join(
                    self.ep_dir, f"{self.sample.sample_id}_final_sq.json"
                )
                final_mesh_mapping_info = self._export_final_mesh_mapping_ply(
                    final_npy,
                    final_json,
                    final_ply,
                    grouped_export_info,
                )
                info.update(final_mesh_mapping_info)
                self.episode_records[-1].update(final_mesh_mapping_info)
            part_reward = final_part_miou - self.init_part_miou
            summary = dict(info)
            summary["step_reward"] = float(info["reward"])
            summary["part_reward"] = float(part_reward)
            summary["reward"] = float(part_reward)
            summary["num_steps"] = int(len(self.episode_records))
            trajectory = [dict(record) for record in self.episode_records]
            if trajectory:
                trajectory[-1]["step_reward"] = float(trajectory[-1]["reward"])
                trajectory[-1]["part_reward"] = float(part_reward)
            summary["trajectory"] = trajectory
            summary["group_ids"] = [int(x) for x in self.group_ids]
            summary["bootstrap"] = self._bootstrap_provenance_summary()
            summary.update(grouped_export_info)
            with open(
                os.path.join(self.ep_dir, "episode_summary.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        return (
            self._state(pred_assign=pred_after, soft_assign=soft_after),
            float(reward),
            done,
            info,
        )
