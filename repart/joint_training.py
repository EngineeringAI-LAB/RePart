#!/usr/bin/env python3
"""Jointly fine-tune RePart policies from completed category runs.

Each positional input is a run directory containing:

* ``experiment_config.json``
* ``sq_partnet_rl_epoch_*.pt``
* ``generated_sdf_csv/``
* ``sq_label_cache/``

The source policies are aligned to the largest ``max_prims`` value and merged
with an equal parameter average.  SDF files are used in place from their
source runs, while SQ-label cache files are linked into the joint output.  SDF
generation is disabled and matching reused label caches are trusted; if a
cache was built for a different number of bootstrap primitives, the labels
are recomputed for the current joint MPS result.

Example:

    python -m repart.joint_training \
        runs/sq_partnet_little \
        runs/sq_partnet_container \
        runs/sq_partnet_furniture \
        --out-dir runs/sq_partnet_joint_finetune

Source categories are sampled with equal probability.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .data import PartNetSample, discover_samples, resolve_uid_point_sample
from .environment import ActionCatalog
from .data import _sdf_csv_quick_error
from .policy import StateEncoder
from .ppo import ppo_train, set_seed


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAME = "experiment_config.json"
GLOBAL_FEATURE_DIM = 8


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_run(run_value: str) -> Dict[str, Any]:
    run_dir = _resolve_repo_path(run_value)
    config_path = run_dir / CONFIG_NAME
    checkpoints = sorted(run_dir.glob("sq_partnet_rl_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoint found in {run_dir}")
    checkpoint_path = checkpoints[-1]
    cache_dir = run_dir / "sq_label_cache"
    sdf_dir = run_dir / "generated_sdf_csv"
    for required in (config_path, checkpoint_path, cache_dir, sdf_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Required source-run path does not exist: {required}"
            )

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    source_args = config.get("args")
    if not isinstance(source_args, dict):
        raise ValueError(f"{config_path} has no valid 'args' object")
    sample_ids = config.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError(f"{config_path} has no non-empty 'sample_ids' list")
    if not source_args.get("data_root"):
        raise ValueError(f"{config_path} does not record args.data_root")

    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "cache_dir": cache_dir,
        "sdf_dir": sdf_dir,
        "config": config,
        "args": source_args,
        "sample_ids": [str(value) for value in sample_ids],
        "max_prims": int(source_args["max_prims"]),
    }


def _discover_reused_samples(source: Dict[str, Any]) -> List[PartNetSample]:
    source_args = source["args"]
    data_root = _resolve_repo_path(str(source_args["data_root"]))
    point_root_value = source_args.get("point_sample_root")
    point_root = (
        str(_resolve_repo_path(str(point_root_value))) if point_root_value else None
    )
    discovered = discover_samples(
        str(data_root),
        point_sample_root=point_root,
        sdf_root=None,
    )
    by_id = {sample.sample_id: sample for sample in discovered}
    wanted_ids = list(source["sample_ids"])
    # A source mesh collection may have been reorganized after the original
    # run.  Fine-tuning only needs point samples plus the reused SDF, so recover
    # such entries directly from the stable PartNet point-sample root.
    if point_root is not None:
        for sample_id in wanted_ids:
            if sample_id not in by_id:
                recovered = resolve_uid_point_sample(sample_id, point_root)
                if recovered is not None:
                    by_id[sample_id] = recovered
    missing = [sample_id for sample_id in wanted_ids if sample_id not in by_id]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"{source['run_dir']}: {len(missing)} configured samples are missing "
            f"from {data_root} (first: {preview})"
        )

    samples = [by_id[sample_id] for sample_id in wanted_ids]
    stats = _apply_reused_preprocessed_run_fast(samples, source)
    for sample in samples:
        setattr(sample, "_joint_source_max_prims", int(source["max_prims"]))
        setattr(sample, "_ppo_domain_id", int(source["domain_id"]))
    source["reuse_stats"] = stats
    return samples


def _apply_reused_preprocessed_run_fast(
    samples: Sequence[PartNetSample],
    source: Mapping[str, Any],
) -> Dict[str, int]:
    """Reuse one source run without recursively walking all rollout outputs."""
    run_dir = Path(source["run_dir"])
    sdf_dir = Path(source["sdf_dir"])
    stats = {
        "total_input_samples": len(samples),
        "valid_reused_sdf": 0,
        "missing_reused_sdf": 0,
        "invalid_reused_sdf": 0,
        "reused_initial_sq": 0,
        "invalid_initial_sq": 0,
        "kept_samples": 0,
    }

    for sample in samples:
        sdf_path = sdf_dir / f"{sample.sample_id}.csv"
        if not sdf_path.is_file():
            stats["missing_reused_sdf"] += 1
            raise FileNotFoundError(
                f"Missing reusable SDF for sample {sample.sample_id}: {sdf_path}"
            )
        sdf_error = _sdf_csv_quick_error(sdf_path)
        if sdf_error is not None:
            stats["invalid_reused_sdf"] += 1
            raise RuntimeError(f"Invalid reusable SDF {sdf_path}: {sdf_error}")
        sample.sdf_csv_path = str(sdf_path)
        sample.trust_sdf_csv = True
        stats["valid_reused_sdf"] += 1

        # Expert rollouts are small and have a predictable per-sample layout.
        # Do not rglob(run_dir): PPO/eval history can contain millions of paths.
        expert_sample_dir = run_dir / "expert_rollouts" / sample.sample_id
        initial_candidates = (
            sorted(
                expert_sample_dir.glob(f"episode_*/{sample.sample_id}_initial_sq.npy")
            )
            if expert_sample_dir.is_dir()
            else []
        )
        if initial_candidates:
            initial_path = initial_candidates[0]
            try:
                values = np.asarray(
                    np.load(initial_path, allow_pickle=False),
                    dtype=np.float64,
                )
                if (
                    values.ndim != 2
                    or values.shape[1] != 11
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        f"expected a finite (N, 11) array, got {values.shape}"
                    )
                provenance_path = initial_path.with_name(
                    f"{sample.sample_id}_initial_sq_provenance.json"
                )
                provenance: Dict[str, Any] = {}
                if provenance_path.is_file():
                    with provenance_path.open("r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                    if isinstance(loaded, dict):
                        provenance.update(loaded)
                provenance.setdefault("sample_id", sample.sample_id)
                provenance.setdefault("bootstrap_source", "mps")
                provenance.setdefault("fallback_single_bbox", False)
                provenance["sdf_csv_path"] = str(sdf_path)
                sample.bootstrap_x = np.array(values, copy=True)
                sample.bootstrap_info_cache = provenance
                sample.bootstrap_source = str(provenance["bootstrap_source"])
                sample.initial_sq_path = str(initial_path)
                sample.initial_sq_provenance = dict(provenance)
                stats["reused_initial_sq"] += 1
            except Exception as exc:
                stats["invalid_initial_sq"] += 1
                print(
                    f"[reuse-preprocessed] WARNING invalid initial SQ ignored "
                    f"for {sample.sample_id}: {type(exc).__name__}: {exc}"
                )

    stats["kept_samples"] = len(samples)
    print(
        "[reuse-preprocessed] "
        f"run={run_dir} kept={len(samples)}/{len(samples)} "
        f"valid_sdf={stats['valid_reused_sdf']} "
        "missing_sdf=0 invalid_sdf=0 "
        f"initial_sq={stats['reused_initial_sq']}"
    )
    return stats


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(
            f"Refusing to replace an existing cache entry: {destination}"
        )
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _stage_label_caches(
    sources: Sequence[Mapping[str, Any]],
    output_cache_dir: Path,
) -> int:
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sources:
        cache_dir = Path(source["cache_dir"])
        for sample_id in source["sample_ids"]:
            npz_path = cache_dir / f"{sample_id}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(
                    f"Missing reusable SQ-label cache for sample {sample_id}: {npz_path}"
                )
            # The source training run already produced this cache.  Do not
            # decompress all arrays during startup; the environment validates
            # an entry when that sample is first selected for a rollout.
            _link_or_copy(npz_path, output_cache_dir / npz_path.name)
            json_path = cache_dir / f"{sample_id}.json"
            if json_path.is_file():
                _link_or_copy(json_path, output_cache_dir / json_path.name)
            count += 1
    return count


def _align_tensor(
    key: str,
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    source_max_prims: int,
    target_max_prims: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a target-shaped value and a per-element contribution mask."""
    source_tensor = source_tensor.detach().cpu().to(dtype=target_tensor.dtype)
    target_shape = tuple(target_tensor.shape)
    if tuple(source_tensor.shape) == target_shape:
        return source_tensor, torch.ones_like(target_tensor)

    # max_prims only changes the SQ-mask slice of actor_context.0's input.
    if key != "actor_context.0.weight" or source_tensor.ndim != 2:
        raise ValueError(
            f"Cannot align checkpoint tensor {key}: source={tuple(source_tensor.shape)} "
            f"target={target_shape}"
        )
    if source_max_prims > target_max_prims:
        raise ValueError(
            f"Source max_prims={source_max_prims} exceeds target={target_max_prims}"
        )
    source_prefix = source_tensor.shape[1] - source_max_prims - GLOBAL_FEATURE_DIM
    target_prefix = target_tensor.shape[1] - target_max_prims - GLOBAL_FEATURE_DIM
    if (
        source_tensor.shape[0] != target_tensor.shape[0]
        or source_prefix != target_prefix
    ):
        raise ValueError(
            f"Unexpected actor_context layout: source={tuple(source_tensor.shape)}, "
            f"target={target_shape}"
        )

    aligned = torch.zeros_like(target_tensor)
    present = torch.zeros_like(target_tensor)
    aligned[:, :target_prefix] = source_tensor[:, :source_prefix]
    present[:, :target_prefix] = 1
    aligned[:, target_prefix : target_prefix + source_max_prims] = source_tensor[
        :, source_prefix : source_prefix + source_max_prims
    ]
    present[:, target_prefix : target_prefix + source_max_prims] = 1
    aligned[:, -GLOBAL_FEATURE_DIM:] = source_tensor[:, -GLOBAL_FEATURE_DIM:]
    present[:, -GLOBAL_FEATURE_DIM:] = 1
    return aligned, present


def _average_source_checkpoints(
    sources: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    target_max_prims: int,
    device: torch.device,
) -> StateEncoder:
    catalog = ActionCatalog(max_prims=target_max_prims)
    model = StateEncoder(
        action_dim=catalog.action_dim,
        max_prims=target_max_prims,
    ).to(device)
    target_state = model.state_dict()
    sums = {
        key: torch.zeros_like(value, device="cpu")
        for key, value in target_state.items()
    }
    denominators = {
        key: torch.zeros_like(value, device="cpu")
        for key, value in target_state.items()
    }

    for source, weight in zip(sources, weights):
        try:
            checkpoint = torch.load(
                source["checkpoint_path"], map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(source["checkpoint_path"], map_location="cpu")
        state = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        if set(state) != set(target_state):
            missing = sorted(set(target_state) - set(state))
            extra = sorted(set(state) - set(target_state))
            raise ValueError(
                f"Checkpoint keys differ for {source['checkpoint_path']}: "
                f"missing={missing}, extra={extra}"
            )
        for key, target_tensor in target_state.items():
            aligned, present = _align_tensor(
                key,
                state[key],
                target_tensor.detach().cpu(),
                source_max_prims=int(source["max_prims"]),
                target_max_prims=target_max_prims,
            )
            sums[key].add_(aligned * float(weight))
            denominators[key].add_(present * float(weight))

    averaged: Dict[str, torch.Tensor] = {}
    for key in target_state:
        if bool((denominators[key] <= 0).any()):
            raise RuntimeError(
                f"No source checkpoint contributed to parts of tensor {key}"
            )
        averaged[key] = sums[key] / denominators[key]
    model.load_state_dict(averaged, strict=True)
    return model


def _weighted_source_value(
    sources: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    key: str,
) -> float:
    return float(
        sum(
            float(source["args"][key]) * weight
            for source, weight in zip(sources, weights)
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint RePart PPO fine-tuning with source SDF/label-cache reuse"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        metavar="RUN",
        help="Three completed category-specific run directories",
    )
    parser.add_argument("--out-dir", default="runs/sq_partnet_joint_finetune")
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--rollout-episodes-per-epoch", type=int, default=304)
    parser.add_argument("--rollout-workers", type=int, default=16)
    parser.add_argument(
        "--worker-sdf-cache-size",
        type=int,
        default=2,
        help="Maximum number of decompressed SDF/Grid samples retained by each rollout worker",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--group-excess-penalty-w",
        type=float,
        default=None,
        help="Override the source-weighted dense group-excess penalty weight.",
    )
    parser.add_argument(
        "--stop-group-penalty-w",
        type=float,
        default=None,
        help="Override the source-weighted stop group penalty weight.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive")
    if args.rollout_episodes_per_epoch <= 0 or args.rollout_workers <= 0:
        raise ValueError("rollout episode and worker counts must be positive")
    if args.worker_sdf_cache_size <= 0:
        raise ValueError("--worker-sdf-cache-size must be positive")

    if len(args.run_dirs) != 3:
        raise ValueError("Joint fine-tuning expects three category-specific runs")
    sources = [_load_run(value) for value in args.run_dirs]
    for domain_id, source in enumerate(sources):
        source["domain_id"] = int(domain_id)
        source["domain_name"] = Path(source["run_dir"]).name
    weights = [1.0 / len(sources)] * len(sources)
    all_ids = [sample_id for source in sources for sample_id in source["sample_ids"]]
    if len(all_ids) != len(set(all_ids)):
        duplicates = sorted(
            {sample_id for sample_id in all_ids if all_ids.count(sample_id) > 1}
        )
        raise ValueError(
            "Sample IDs must be unique across source runs; duplicates: "
            + ", ".join(duplicates[:20])
        )

    output_dir = _resolve_repo_path(args.out_dir)
    source_dirs = {Path(source["run_dir"]) for source in sources}
    if output_dir in source_dirs:
        raise ValueError("--out-dir must not be one of the source run directories")
    if list(output_dir.glob("*.pt")):
        raise FileExistsError(
            f"{output_dir} already contains joint checkpoints; choose a fresh --out-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_max_prims = max(int(source["max_prims"]) for source in sources)
    samples: List[PartNetSample] = []
    for source in sources:
        print(
            f"[joint][prepare] loading samples/SDF from {source['run_dir']} ...",
            flush=True,
        )
        source_samples = _discover_reused_samples(source)
        source["num_joint_samples"] = len(source_samples)
        samples.extend(source_samples)

    print("[joint][prepare] linking SQ-label caches ...", flush=True)
    linked_cache_count = _stage_label_caches(sources, output_dir / "sq_label_cache")
    if linked_cache_count != len(samples):
        raise RuntimeError(
            f"Staged {linked_cache_count} caches for {len(samples)} samples"
        )

    print("[joint][prepare] averaging source checkpoints ...", flush=True)
    model = _average_source_checkpoints(
        sources, weights, target_max_prims=target_max_prims, device=device
    )

    max_steps = max(int(source["args"]["max_steps"]) for source in sources)
    global_cfg = {
        "mesh2sdf_script": None,
        "sdf_out_dir": None,
        "bootstrap_source": "mps",
        "max_prims": target_max_prims,
        "max_division": max(
            int(source["args"].get("max_division", 50)) for source in sources
        ),
        "region_timeout_sec": None,
        "max_region_refits": None,
        "heartbeat_sec": 30.0,
        "num_workers": None,
        "parallel_min_regions": 3,
        "chunk_size": 1,
        "sq_label_cache_dir": str(output_dir / "sq_label_cache"),
        "reuse_preprocessed_run": [str(source["run_dir"]) for source in sources],
        "reuse_preprocessed_stats": [
            source.get("reuse_stats", {}) for source in sources
        ],
        "direct_rl_from_run": True,
        "trust_reused_sq_label_cache": True,
        "seed": args.seed,
        "verbose": False,
    }
    env_kwargs = {
        "max_prims": target_max_prims,
        "max_steps": max_steps,
        "point_sample_size": max(
            int(source["args"]["point_sample_size"]) for source in sources
        ),
        "invalid_action_penalty": _weighted_source_value(
            sources, weights, "invalid_action_penalty"
        ),
        "render_group_highlights": False,
        "stop_reward_bonus": _weighted_source_value(
            sources, weights, "stop_reward_bonus"
        ),
        "group_excess_penalty_w": (
            float(args.group_excess_penalty_w)
            if args.group_excess_penalty_w is not None
            else _weighted_source_value(sources, weights, "group_excess_penalty_w")
        ),
        "stop_group_penalty_w": (
            float(args.stop_group_penalty_w)
            if args.stop_group_penalty_w is not None
            else _weighted_source_value(sources, weights, "stop_group_penalty_w")
        ),
        "merge_neighbor_k": max(
            int(source["args"]["merge_neighbor_k"]) for source in sources
        ),
        "sq_label_method": "coverage",
        "final_mesh_mapping_mode": "atom",
    }
    joint_config = {
        "mode": "joint_finetune",
        "device": str(device),
        "num_samples": len(samples),
        "sample_ids": all_ids,
        "source_runs": [
            {
                "run_dir": str(source["run_dir"]),
                "checkpoint": str(source["checkpoint_path"]),
                "config": str(source["config_path"]),
                "num_samples": int(source["num_joint_samples"]),
                "max_prims": int(source["max_prims"]),
                "domain_id": int(source["domain_id"]),
                "domain_name": str(source["domain_name"]),
                "reuse_stats": source.get("reuse_stats", {}),
            }
            for source in sources
        ],
        "args": vars(args),
        "global_cfg": global_cfg,
        "env_kwargs": env_kwargs,
        "initialization": "equal_parameter_average",
        "normalize_advantages_by_domain": True,
        "average_loss_by_domain": True,
    }
    (output_dir / CONFIG_NAME).write_text(
        json.dumps(joint_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[joint] sources={len(sources)} samples={len(samples)} "
        f"max_prims={target_max_prims} linked_label_caches={linked_cache_count}"
    )
    print("[joint] equal source sampling and parameter averaging")

    catalog = ActionCatalog(max_prims=target_max_prims)
    ppo_train(
        samples=samples,
        action_catalog=catalog,
        out_dir=str(output_dir),
        global_cfg=global_cfg,
        env_kwargs=env_kwargs,
        device=device,
        train_epochs=args.train_epochs,
        rollout_episodes_per_epoch=args.rollout_episodes_per_epoch,
        rollout_workers=args.rollout_workers,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        policy_temperature=args.policy_temperature,
        aux_part_loss_w=_weighted_source_value(sources, weights, "aux_part_loss_w"),
        aux_group_loss_w=_weighted_source_value(sources, weights, "aux_group_loss_w"),
        aux_target_count_loss_w=_weighted_source_value(
            sources, weights, "aux_target_count_loss_w"
        ),
        initial_model=model,
        normalize_advantages_by_domain=True,
        average_loss_by_domain=True,
        worker_sdf_cache_size=args.worker_sdf_cache_size,
        rollout_domain_weights={
            int(source["domain_id"]): float(weight)
            for source, weight in zip(sources, weights)
        },
    )
    print(f"[joint] fine-tuning complete: {output_dir}")


if __name__ == "__main__":
    main()
