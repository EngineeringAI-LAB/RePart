from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from .data import PartNetSample, discover_samples
from .environment import ActionCatalog
from .data import _sdf_csv_quick_error
from .policy import StateEncoder
from .ppo import set_seed, ppo_train, evaluate_policy
from .outputs import save_experiment_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RePart SQ grouping with PPO")
    p.add_argument(
        "--data-root",
        default="data/train_shapes",
        help="Mesh root (UID-named meshes) or a PartNet point-sample root with label-*.txt / pts-*.pts / ply-*.ply",
    )
    p.add_argument(
        "--point-sample-root",
        default="partnet_select",
        help="PartNet root containing {uid}/point_sample/{ply,label,pts}-10000.* when --data-root points to meshes",
    )
    p.add_argument(
        "--sdf-root",
        default=None,
        help="Optional root containing precomputed SDF csv files",
    )
    p.add_argument(
        "--mesh2sdf-script",
        default=str(
            Path(__file__).resolve().parent.parent / "scripts" / "mesh_to_sdf.py"
        ),
        help="Mesh-to-SDF script used to build SDF CSV files",
    )
    p.add_argument("--out-dir", default="runs/sq_partnet_rl", help="Output directory")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--checkpoint", default=None, help="Checkpoint for evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reuse-preprocessed-run",
        default=None,
        help=(
            "Reuse generated_sdf_csv, sq_label_cache, and any saved initial SQs "
            "from an earlier run directory."
        ),
    )
    p.add_argument(
        "--reuse-preprocessed-only",
        action="store_true",
        help="When reusing a run, keep only samples that already have a valid reused SDF csv.",
    )
    p.add_argument(
        "--direct-rl-from-run",
        default=None,
        help=(
            "Shortcut for RL training from an earlier processed run: reuse that "
            "run, keep only samples with valid generated SDF csv, reuse its "
            "sq_label_cache, and do not generate new SDF files."
        ),
    )
    p.add_argument(
        "--trust-reused-sq-label-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Load sq_label_cache from the reused run without requiring an exact "
            "cache-key match. Defaults to enabled with --direct-rl-from-run."
        ),
    )

    p.add_argument("--max-prims", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--point-sample-size", type=int, default=512)

    p.add_argument("--max-division", type=int, default=50)
    p.add_argument("--region-timeout-sec", type=float, default=None)
    p.add_argument("--max-region-refits", type=int, default=None)
    p.add_argument("--heartbeat-sec", type=float, default=30.0)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--parallel-min-regions", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=1)

    p.add_argument("--train-epochs", type=int, default=200)
    p.add_argument("--rollout-episodes-per-epoch", type=int, default=304)
    p.add_argument(
        "--rollout-workers",
        type=int,
        default=16,
        help="Parallel PPO rollout workers; 1 keeps the serial collector.",
    )
    p.add_argument(
        "--worker-sdf-cache-size",
        type=int,
        default=2,
        help="Maximum number of decompressed SDF/Grid samples retained by each rollout worker",
    )
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used by GAE/returns. The main PPO default is 0.99.",
    )
    p.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda used by PPO. The main PPO default is 0.95.",
    )
    p.add_argument(
        "--myopic-ppo",
        action="store_true",
        help=(
            "Run the Myopic-PPO ablation: keep the same on-policy PPO, actor/critic, "
            "clipping, and dense environment reward, but set gamma=0 and GAE-lambda=0."
        ),
    )
    p.add_argument("--policy-temperature", type=float, default=1.0)
    p.add_argument(
        "--aux-part-loss-w",
        type=float,
        default=0.5,
        help="Auxiliary critic loss weight for final part mIoU prediction.",
    )
    p.add_argument(
        "--aux-group-loss-w",
        type=float,
        default=0.25,
        help="Auxiliary critic loss weight for final normalized group excess prediction.",
    )
    p.add_argument(
        "--aux-target-count-loss-w",
        type=float,
        default=0.1,
        help="Auxiliary actor loss weight for predicting GT target group-count ratio from visible geometry.",
    )
    p.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=-0.02,
        help="Penalty applied when the chosen action is invalid or degenerates.",
    )
    p.add_argument(
        "--stop-reward-bonus",
        type=float,
        default=0.0,
        help="Small bonus added when the policy chooses stop.",
    )
    p.add_argument(
        "--group-excess-penalty-w",
        type=float,
        default=0.02,
        help="Dense reward weight for reducing normalized excess groups above the GT target part count.",
    )
    p.add_argument(
        "--stop-group-penalty-w",
        type=float,
        default=0.1,
        help="Extra normalized excess-group penalty applied when the policy chooses stop.",
    )
    p.add_argument(
        "--merge-neighbor-k",
        type=int,
        default=4,
        help="Each group may only merge with its k nearest neighbor groups.",
    )
    p.add_argument(
        "--sq-label-method",
        choices=["coverage", "legacy"],
        default="coverage",
        help="SQ ground-truth labeling used for group-aware reward/evaluation: "
        "asymmetric AABB coverage (default) or the original point-to-SQ majority rule.",
    )
    p.add_argument(
        "--final-mesh-mapping-mode",
        choices=["atom", "nearest"],
        default="atom",
        help=(
            "PLY exported at eval episode end: atom uses the default curvature "
            "atom-voting mesh_mapping output; nearest uses the debug nearest-group "
            "baseline mesh. Training rewards/rollouts are unchanged."
        ),
    )
    args = p.parse_args()
    if args.myopic_ppo:
        # Myopic-PPO is deliberately a one-step PPO control.  Keep this
        # override in the parsed namespace so experiment_config.json records
        # The ablation config records the values actually used by ppo_train().
        args.gamma = 0.0
        args.gae_lambda = 0.0
    if not 0.0 <= float(args.gamma) <= 1.0:
        raise ValueError(f"--gamma must be in [0, 1], got {args.gamma}")
    if not 0.0 <= float(args.gae_lambda) <= 1.0:
        raise ValueError(f"--gae-lambda must be in [0, 1], got {args.gae_lambda}")
    return args


def _index_reused_initial_sq(run_root: Path) -> Dict[str, Tuple[Path, Optional[Path]]]:
    indexed: Dict[str, Tuple[Path, Optional[Path]]] = {}
    for sq_path in sorted(run_root.rglob("*_initial_sq.npy")):
        uid = sq_path.name[: -len("_initial_sq.npy")]
        provenance = sq_path.with_name(f"{uid}_initial_sq_provenance.json")
        indexed.setdefault(uid, (sq_path, provenance if provenance.exists() else None))
    return indexed


def apply_reused_preprocessed_run(
    samples: Sequence[PartNetSample],
    run_root: str,
    processed_only: bool = False,
) -> Tuple[List[PartNetSample], Dict[str, int]]:
    root = Path(run_root)
    sdf_dir = root / "generated_sdf_csv"
    initial_sq_index = _index_reused_initial_sq(root)
    kept: List[PartNetSample] = []
    stats = {
        "total_input_samples": int(len(samples)),
        "valid_reused_sdf": 0,
        "missing_reused_sdf": 0,
        "invalid_reused_sdf": 0,
        "reused_initial_sq": 0,
        "invalid_initial_sq": 0,
        "kept_samples": 0,
    }

    for sample in samples:
        reused_sdf: Optional[Path] = None
        candidate_sdf = sdf_dir / f"{sample.sample_id}.csv"
        if candidate_sdf.exists():
            error = _sdf_csv_quick_error(candidate_sdf)
            if error is None:
                reused_sdf = candidate_sdf
                sample.sdf_csv_path = str(candidate_sdf)
                sample.trust_sdf_csv = True
                stats["valid_reused_sdf"] += 1
            else:
                stats["invalid_reused_sdf"] += 1
                print(
                    f"[reuse-preprocessed] WARNING invalid SDF ignored for "
                    f"{sample.sample_id}: {candidate_sdf} ({error})"
                )
        else:
            stats["missing_reused_sdf"] += 1

        initial_entry = initial_sq_index.get(sample.sample_id)
        if initial_entry is not None:
            sq_path, provenance_path = initial_entry
            try:
                x = np.asarray(np.load(sq_path, allow_pickle=False), dtype=np.float64)
                if x.ndim != 2 or x.shape[1] != 11 or not np.isfinite(x).all():
                    raise ValueError(f"expected finite (N, 11), got {x.shape}")
                sample.bootstrap_x = np.array(x, copy=True)
                info: Dict[str, Any] = {}
                if provenance_path is not None:
                    with open(provenance_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        info.update(loaded)
                info.setdefault("sample_id", sample.sample_id)
                info.setdefault("bootstrap_source", info.get("global_optimizer", "mps"))
                info.setdefault("fallback_single_bbox", False)
                info["sdf_csv_path"] = str(reused_sdf or sample.sdf_csv_path or "")
                sample.bootstrap_info_cache = info
                sample.bootstrap_source = str(info.get("bootstrap_source", "mps"))
                sample.initial_sq_path = str(sq_path)
                sample.initial_sq_provenance = dict(info)
                stats["reused_initial_sq"] += 1
            except Exception as exc:
                sample.bootstrap_x = None
                sample.bootstrap_info_cache = None
                stats["invalid_initial_sq"] += 1
                print(
                    f"[reuse-preprocessed] WARNING invalid initial SQ ignored for "
                    f"{sample.sample_id}: {sq_path} ({type(exc).__name__}: {exc})"
                )

        if processed_only and reused_sdf is None:
            continue
        kept.append(sample)

    stats["kept_samples"] = int(len(kept))
    if not kept:
        raise RuntimeError(
            f"No samples left after applying reused preprocessed run {run_root!r}; "
            f"check generated_sdf_csv or disable --reuse-preprocessed-only."
        )
    print(
        "[reuse-preprocessed] "
        f"run={root} kept={stats['kept_samples']}/{stats['total_input_samples']} "
        f"valid_sdf={stats['valid_reused_sdf']} "
        f"missing_sdf={stats['missing_reused_sdf']} "
        f"invalid_sdf={stats['invalid_reused_sdf']} "
        f"initial_sq={stats['reused_initial_sq']}"
    )
    return kept, stats


def main() -> None:
    args = parse_args()
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("--checkpoint is only supported in eval mode")
    if args.mode == "train" and args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive")
    if args.mode == "train" and list(Path(args.out_dir).glob("*.pt")):
        raise FileExistsError(
            "Training output already contains checkpoints; choose a fresh --out-dir"
        )
    if args.worker_sdf_cache_size <= 0:
        raise ValueError("--worker-sdf-cache-size must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    samples = discover_samples(
        args.data_root, point_sample_root=args.point_sample_root, sdf_root=args.sdf_root
    )
    reused_run_root = args.direct_rl_from_run or args.reuse_preprocessed_run
    reuse_stats: Dict[str, int] = {}
    if reused_run_root:
        samples, reuse_stats = apply_reused_preprocessed_run(
            samples,
            run_root=reused_run_root,
            processed_only=bool(
                args.direct_rl_from_run or args.reuse_preprocessed_only
            ),
        )
    action_catalog = ActionCatalog(max_prims=args.max_prims)
    sq_label_cache_base = (
        Path(reused_run_root)
        if reused_run_root and (Path(reused_run_root) / "sq_label_cache").exists()
        else Path(args.checkpoint).resolve().parent
        if args.mode == "eval" and args.checkpoint
        else Path(args.out_dir)
    )
    sdf_out_dir = (
        str(Path(reused_run_root) / "generated_sdf_csv")
        if reused_run_root and (Path(reused_run_root) / "generated_sdf_csv").exists()
        else os.path.join(args.out_dir, "generated_sdf_csv")
    )
    mesh2sdf_script = None if args.direct_rl_from_run else args.mesh2sdf_script

    global_cfg = {
        "mesh2sdf_script": mesh2sdf_script,
        "sdf_out_dir": sdf_out_dir,
        "bootstrap_source": "mps",
        "max_prims": args.max_prims,
        "max_division": args.max_division,
        "region_timeout_sec": args.region_timeout_sec,
        "max_region_refits": args.max_region_refits,
        "heartbeat_sec": args.heartbeat_sec,
        "num_workers": args.num_workers,
        "parallel_min_regions": args.parallel_min_regions,
        "chunk_size": args.chunk_size,
        "sq_label_cache_dir": str(sq_label_cache_base / "sq_label_cache"),
        "reuse_preprocessed_run": str(reused_run_root or ""),
        "reuse_preprocessed_stats": reuse_stats,
        "direct_rl_from_run": bool(args.direct_rl_from_run),
        "trust_reused_sq_label_cache": (
            bool(args.direct_rl_from_run)
            if args.trust_reused_sq_label_cache is None
            else bool(args.trust_reused_sq_label_cache)
        ),
        "ppo_gamma": float(args.gamma),
        "ppo_gae_lambda": float(args.gae_lambda),
        "myopic_ppo": bool(args.myopic_ppo),
        "seed": args.seed,
        "verbose": False,
    }
    env_kwargs = {
        "max_prims": args.max_prims,
        "max_steps": args.max_steps,
        "point_sample_size": args.point_sample_size,
        "invalid_action_penalty": args.invalid_action_penalty,
        "render_group_highlights": False,
        "stop_reward_bonus": args.stop_reward_bonus,
        "group_excess_penalty_w": args.group_excess_penalty_w,
        "stop_group_penalty_w": args.stop_group_penalty_w,
        "merge_neighbor_k": args.merge_neighbor_k,
        "sq_label_method": args.sq_label_method,
        "final_mesh_mapping_mode": args.final_mesh_mapping_mode,
    }
    save_experiment_config(
        out_dir=args.out_dir,
        args=args,
        global_cfg=global_cfg,
        env_kwargs=env_kwargs,
        device=device,
        samples=samples,
    )

    if args.mode == "train":
        model = ppo_train(
            samples=samples,
            action_catalog=action_catalog,
            out_dir=args.out_dir,
            global_cfg=global_cfg,
            env_kwargs=env_kwargs,
            device=device,
            train_epochs=args.train_epochs,
            rollout_episodes_per_epoch=args.rollout_episodes_per_epoch,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            lr=args.lr,
            policy_temperature=args.policy_temperature,
            rollout_workers=args.rollout_workers,
            aux_part_loss_w=args.aux_part_loss_w,
            aux_group_loss_w=args.aux_group_loss_w,
            aux_target_count_loss_w=args.aux_target_count_loss_w,
            worker_sdf_cache_size=args.worker_sdf_cache_size,
        )
        evaluate_policy(
            model, samples, action_catalog, args.out_dir, global_cfg, env_kwargs, device
        )
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in eval mode")
        model = StateEncoder(
            action_dim=action_catalog.action_dim,
            max_prims=args.max_prims,
        ).to(device)
        # Evaluation checkpoints are produced locally by this project and may
        # include NumPy RNG/optimizer metadata.  PyTorch 2.6 otherwise tries
        # weights_only=True and rejects those trusted objects.
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        evaluate_policy(
            model, samples, action_catalog, args.out_dir, global_cfg, env_kwargs, device
        )


if __name__ == "__main__":
    main()
