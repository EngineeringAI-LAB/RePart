from __future__ import annotations

from collections import OrderedDict
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.optim as optim
from .data import PartNetSample
from .environment import ActionCatalog, SQEditEnv
from .policy import StateEncoder
from .outputs import save_training_reward_curve


class RolloutBuffer:
    def __init__(self):
        self.states: List[Dict[str, np.ndarray]] = []
        self.actions: List[int] = []
        self.logps: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.final_part_miou_targets: List[float] = []
        self.final_group_excess_targets: List[float] = []
        self.domain_ids: List[int] = []

    def add(
        self,
        state,
        action,
        logp,
        reward,
        done,
        value,
        final_part_miou_target=0.0,
        final_group_excess_target=0.0,
        domain_id=0,
    ):
        self.states.append(state)
        self.actions.append(int(action))
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.final_part_miou_targets.append(float(final_part_miou_target))
        self.final_group_excess_targets.append(float(final_group_excess_target))
        self.domain_ids.append(int(domain_id))


_ROLLOUT_WORKER: Dict[str, Any] = {}


def _parallel_rollout_worker_init(
    samples_state: Sequence[PartNetSample],
    global_cfg: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    max_prims: int,
    rollout_dir: str,
    policy_temperature: float,
    sdf_cache_samples: int,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _ROLLOUT_WORKER["samples"] = {sample.sample_id: sample for sample in samples_state}
    _ROLLOUT_WORKER["global_cfg"] = global_cfg
    _ROLLOUT_WORKER["env_kwargs"] = env_kwargs
    _ROLLOUT_WORKER["catalog"] = ActionCatalog(max_prims=max_prims)
    _ROLLOUT_WORKER["rollout_dir"] = rollout_dir
    _ROLLOUT_WORKER["temperature"] = float(policy_temperature)
    _ROLLOUT_WORKER["sdf_cache_samples"] = max(1, int(sdf_cache_samples))
    _ROLLOUT_WORKER["sdf_cache_lru"] = OrderedDict()
    _ROLLOUT_WORKER["model"] = StateEncoder(
        action_dim=_ROLLOUT_WORKER["catalog"].action_dim,
        max_prims=max_prims,
    ).to("cpu")
    _ROLLOUT_WORKER["model"].eval()


def _parallel_worker_touch_sdf_cache(sample_id: str) -> None:
    """Bound persistent SDF/Grid memory while retaining small sample caches."""
    limit = int(_ROLLOUT_WORKER["sdf_cache_samples"])
    lru: OrderedDict[str, None] = _ROLLOUT_WORKER["sdf_cache_lru"]
    lru.pop(sample_id, None)
    lru[sample_id] = None
    while len(lru) > int(limit):
        evicted_id, _ = lru.popitem(last=False)
        evicted = _ROLLOUT_WORKER["samples"][evicted_id]
        evicted.cached_sdf_csv_path = None
        evicted.cached_sdf_global = None
        evicted.cached_sdf_grid = None


def _parallel_rollout_worker(
    payload: Tuple[Dict[str, torch.Tensor], List[Tuple[int, str]]]
) -> List[Any]:
    state_dict, assignments = payload
    model: StateEncoder = _ROLLOUT_WORKER["model"]
    model.load_state_dict(state_dict)
    cpu = torch.device("cpu")
    results = []
    for global_ep_idx, sample_id in assignments:
        sample = _ROLLOUT_WORKER["samples"][sample_id]
        domain_id = int(getattr(sample, "_ppo_domain_id", 0))
        env = SQEditEnv(
            sample,
            _ROLLOUT_WORKER["catalog"],
            _ROLLOUT_WORKER["rollout_dir"],
            global_cfg=_ROLLOUT_WORKER["global_cfg"],
            episode_idx=global_ep_idx,
            **_ROLLOUT_WORKER["env_kwargs"],
        )
        state = env.reset()
        done = False
        ep_reward = 0.0
        transitions = []
        last_info: Dict[str, Any] = {}
        while not done:
            batch_state = stack_states([state], cpu)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(
                    batch_state,
                    deterministic=False,
                    temperature=_ROLLOUT_WORKER["temperature"],
                )
            action = int(action_t.item())
            next_state, reward, done, info = env.step(action)
            transitions.append(
                (
                    state,
                    action,
                    float(logp_t.item()),
                    reward,
                    done,
                    float(value_t.item()),
                )
            )
            ep_reward += reward
            state = next_state
            last_info = info
        final_part_target = float(
            last_info.get("final_part_miou", last_info.get("after_part_miou", 0.0))
        )
        final_num_groups = float(
            last_info.get("final_num_groups", last_info.get("num_groups", 0.0))
        )
        final_group_excess_target = env._normalized_group_excess(final_num_groups)
        _parallel_worker_touch_sdf_cache(sample_id)
        results.append(
            (
                transitions,
                final_part_target,
                final_group_excess_target,
                ep_reward,
                domain_id,
            )
        )
    return results


def _round_robin_chunks(seq: List[Any], k: int) -> List[List[Any]]:
    chunks: List[List[Any]] = [[] for _ in range(max(1, int(k)))]
    for index, item in enumerate(seq):
        chunks[index % len(chunks)].append(item)
    return [chunk for chunk in chunks if chunk]


STATE_KEYS = [
    "sq_params",
    "sq_mask",
    "point_sample",
    "action_mask",
    "prim_features",
    "global_features",
    "merge_action_features",
    "merge_pair_indices",
    "merge_pair_action_ids",
    "merge_pair_mask",
    "privileged_prim_stats",
    "privileged_global_features",
    "target_group_count_ratio",
]


ACTOR_STATE_KEYS = [
    "sq_params",
    "sq_mask",
    "point_sample",
    "action_mask",
    "prim_features",
    "global_features",
    "merge_action_features",
    "merge_pair_indices",
    "merge_pair_action_ids",
    "merge_pair_mask",
]


def stack_states(
    states: Sequence[Dict[str, np.ndarray]], device: torch.device
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in STATE_KEYS:
        arr = np.stack([s[k] for s in states], axis=0)
        out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def stack_actor_states(
    states: Sequence[Dict[str, np.ndarray]], device: torch.device
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in ACTOR_STATE_KEYS:
        arr = np.stack([s[k] for s in states], axis=0)
        out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    vals = np.asarray(values + [last_value], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t + 1] * (1.0 - dones[t]) - vals[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    ret = adv + np.asarray(values, dtype=np.float32)
    return adv, ret


def normalize_advantages(
    advantages: np.ndarray,
    domain_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Normalize advantages globally or independently inside each domain."""
    normalized = np.asarray(advantages, dtype=np.float32).copy()
    if normalized.size <= 1:
        return normalized
    domains = (
        np.zeros((normalized.size,), dtype=np.int64)
        if domain_ids is None
        else np.asarray(domain_ids, dtype=np.int64)
    )
    if domains.shape != normalized.shape:
        raise ValueError(
            f"Advantage/domain shape mismatch: {normalized.shape} vs {domains.shape}"
        )
    for domain_id in np.unique(domains):
        mask = domains == domain_id
        values = normalized[mask]
        if values.size == 0:
            continue
        value_std = float(values.std())
        if np.isfinite(value_std) and value_std > 1e-8:
            normalized[mask] = (values - values.mean()) / (value_std + 1e-8)
        else:
            normalized[mask] = values - values.mean()
    return normalized


def domain_macro_mean(
    per_transition_values: torch.Tensor,
    domain_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean inside each present domain, followed by an equal domain mean."""
    values = per_transition_values.reshape(-1)
    domains = domain_ids.reshape(-1)
    if values.shape[0] != domains.shape[0]:
        raise ValueError(
            f"Loss/domain shape mismatch: {tuple(values.shape)} vs {tuple(domains.shape)}"
        )
    domain_means = [
        values[domains == domain_id].mean() for domain_id in torch.unique(domains)
    ]
    if not domain_means:
        raise ValueError("Cannot reduce an empty domain-balanced loss")
    return torch.stack(domain_means).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def make_training_checkpoint_payload(
    model: StateEncoder,
    epoch: int,
    epoch_indices: Sequence[int],
    epoch_avg_rewards: Sequence[float],
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "epoch_indices": [int(x) for x in epoch_indices],
        "epoch_avg_rewards": [float(x) for x in epoch_avg_rewards],
    }


def ppo_train(
    samples: Sequence[PartNetSample],
    action_catalog: ActionCatalog,
    out_dir: str,
    global_cfg: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    device: torch.device,
    train_epochs: int = 50,
    rollout_episodes_per_epoch: int = 4,
    ppo_epochs: int = 4,
    minibatch_size: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    policy_temperature: float = 1.0,
    rollout_workers: int = 1,
    aux_part_loss_w: float = 0.5,
    aux_group_loss_w: float = 0.25,
    aux_target_count_loss_w: float = 0.1,
    initial_model: Optional[StateEncoder] = None,
    normalize_advantages_by_domain: bool = False,
    average_loss_by_domain: bool = False,
    worker_sdf_cache_size: int = 2,
    rollout_domain_weights: Optional[Dict[int, float]] = None,
) -> StateEncoder:
    model = (
        initial_model
        if initial_model is not None
        else StateEncoder(
            action_dim=action_catalog.action_dim,
            max_prims=env_kwargs["max_prims"],
        )
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_indices: List[int] = []
    epoch_avg_rewards: List[float] = []
    rollout_dir = os.path.join(out_dir, "ppo_rollouts")
    train_sample_ids = [sample.sample_id for sample in samples]
    sample_ids_by_domain: Dict[int, List[str]] = {}
    sample_domain_by_id: Dict[str, int] = {}
    for sample in samples:
        domain_id = int(getattr(sample, "_ppo_domain_id", 0))
        sample_ids_by_domain.setdefault(domain_id, []).append(sample.sample_id)
        sample_domain_by_id[sample.sample_id] = domain_id

    normalized_rollout_domain_weights: Optional[Dict[int, float]] = None
    if rollout_domain_weights is not None:
        known_domains = set(sample_ids_by_domain)
        coerced_domain_weights = {
            int(key): float(value) for key, value in rollout_domain_weights.items()
        }
        provided_domains = set(coerced_domain_weights)
        if provided_domains != known_domains:
            raise ValueError(
                "rollout_domain_weights domains must match sample domains: "
                f"provided={sorted(provided_domains)} expected={sorted(known_domains)}"
            )
        raw_domain_weights = {
            int(domain_id): coerced_domain_weights[domain_id]
            for domain_id in sorted(known_domains)
        }
        if any(
            not np.isfinite(weight) or weight < 0.0
            for weight in raw_domain_weights.values()
        ):
            raise ValueError("rollout_domain_weights must be finite and non-negative")
        total_domain_weight = float(sum(raw_domain_weights.values()))
        if total_domain_weight <= 0.0:
            raise ValueError("rollout_domain_weights must have a positive sum")
        normalized_rollout_domain_weights = {
            domain_id: weight / total_domain_weight
            for domain_id, weight in raw_domain_weights.items()
        }
        print(
            "[PPO] rollout domain sampling weights="
            f"{normalized_rollout_domain_weights}"
        )
    use_parallel_rollout = int(rollout_workers) > 1
    pool: Optional[ProcessPoolExecutor] = None
    if use_parallel_rollout:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=int(rollout_workers),
            mp_context=ctx,
            initializer=_parallel_rollout_worker_init,
            initargs=(
                list(samples),
                global_cfg,
                env_kwargs,
                int(env_kwargs["max_prims"]),
                rollout_dir,
                float(policy_temperature),
                worker_sdf_cache_size,
            ),
        )
        print(
            f"[PPO] parallel rollout: {int(rollout_workers)} workers (spawn) "
            f"sdf_cache_samples_per_worker={max(1, int(worker_sdf_cache_size))}"
        )
    else:
        print("[PPO] serial rollout")

    for local_epoch in range(train_epochs):
        epoch = local_epoch
        rollout = RolloutBuffer()
        episode_rewards: List[float] = []
        if normalized_rollout_domain_weights is None:
            selected_sample_ids = [
                random.choice(train_sample_ids)
                for _ in range(rollout_episodes_per_epoch)
            ]
        else:
            selectable_domains = [
                domain_id
                for domain_id, weight in normalized_rollout_domain_weights.items()
                if weight > 0.0
            ]
            selected_domains = random.choices(
                selectable_domains,
                weights=[
                    normalized_rollout_domain_weights[domain_id]
                    for domain_id in selectable_domains
                ],
                k=rollout_episodes_per_epoch,
            )
            selected_sample_ids = [
                random.choice(sample_ids_by_domain[domain_id])
                for domain_id in selected_domains
            ]
        episode_specs = [
            (epoch * rollout_episodes_per_epoch + ep_idx, sample_id)
            for ep_idx, sample_id in enumerate(selected_sample_ids)
        ]
        if normalized_rollout_domain_weights is not None:
            selected_domain_counts = {
                domain_id: sum(
                    sample_domain_by_id[sample_id] == domain_id
                    for sample_id in selected_sample_ids
                )
                for domain_id in normalized_rollout_domain_weights
            }
            print(
                f"[PPO][domain-sampling] epoch={epoch + 1:03d} "
                f"episodes={selected_domain_counts}"
            )
        if use_parallel_rollout:
            assert pool is not None
            state_dict_cpu = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            chunks = _round_robin_chunks(episode_specs, int(rollout_workers))
            futures = [
                pool.submit(_parallel_rollout_worker, (state_dict_cpu, chunk))
                for chunk in chunks
            ]
            for future in futures:
                for (
                    episode_transitions,
                    final_part_target,
                    final_group_excess_target,
                    ep_reward,
                    domain_id,
                ) in future.result():
                    for transition in episode_transitions:
                        rollout.add(
                            *transition,
                            final_part_miou_target=final_part_target,
                            final_group_excess_target=final_group_excess_target,
                            domain_id=domain_id,
                        )
                    episode_rewards.append(ep_reward)
        else:
            for global_ep_idx, sample_id in episode_specs:
                sample = next(
                    sample for sample in samples if sample.sample_id == sample_id
                )
                domain_id = int(getattr(sample, "_ppo_domain_id", 0))
                env = SQEditEnv(
                    sample,
                    action_catalog,
                    rollout_dir,
                    global_cfg=global_cfg,
                    episode_idx=global_ep_idx,
                    **env_kwargs,
                )
                state = env.reset()
                done = False
                ep_reward = 0.0
                episode_transitions: List[
                    Tuple[Dict[str, np.ndarray], int, float, float, float, float]
                ] = []
                last_info: Dict[str, Any] = {}
                while not done:
                    batch_state = stack_states([state], device)
                    with torch.no_grad():
                        action_t, logp_t, value_t = model.act(
                            batch_state,
                            deterministic=False,
                            temperature=policy_temperature,
                        )
                    action = int(action_t.item())
                    next_state, reward, done, info = env.step(action)
                    episode_transitions.append(
                        (
                            state,
                            action,
                            float(logp_t.item()),
                            reward,
                            done,
                            float(value_t.item()),
                        )
                    )
                    ep_reward += reward
                    state = next_state
                    last_info = info
                final_part_target = float(
                    last_info.get(
                        "final_part_miou", last_info.get("after_part_miou", 0.0)
                    )
                )
                final_num_groups = float(
                    last_info.get("final_num_groups", last_info.get("num_groups", 0.0))
                )
                final_group_excess_target = env._normalized_group_excess(
                    final_num_groups
                )
                for transition in episode_transitions:
                    rollout.add(
                        *transition,
                        final_part_miou_target=final_part_target,
                        final_group_excess_target=final_group_excess_target,
                        domain_id=domain_id,
                    )
                episode_rewards.append(ep_reward)
        # bootstrap last value as 0 because episodes are complete in this collector.
        adv, ret = compute_gae(
            rollout.rewards,
            rollout.dones,
            rollout.values,
            last_value=0.0,
            gamma=gamma,
            lam=gae_lambda,
        )
        advantage_domains = (
            rollout.domain_ids if normalize_advantages_by_domain else None
        )
        adv = normalize_advantages(adv, advantage_domains)

        states_all = rollout.states
        actions_all = torch.as_tensor(rollout.actions, dtype=torch.long, device=device)
        old_logps_all = torch.as_tensor(
            rollout.logps, dtype=torch.float32, device=device
        )
        returns_all = torch.as_tensor(ret, dtype=torch.float32, device=device)
        adv_all = torch.as_tensor(adv, dtype=torch.float32, device=device)
        final_part_targets_all = torch.as_tensor(
            rollout.final_part_miou_targets, dtype=torch.float32, device=device
        )
        final_group_targets_all = torch.as_tensor(
            rollout.final_group_excess_targets, dtype=torch.float32, device=device
        )
        domain_ids_all = torch.as_tensor(
            rollout.domain_ids,
            dtype=torch.long,
            device=device,
        )

        if normalize_advantages_by_domain or average_loss_by_domain:
            domain_counts = {
                int(domain_id): int(count)
                for domain_id, count in zip(
                    *np.unique(np.asarray(rollout.domain_ids), return_counts=True)
                )
            }
            print(
                f"[PPO][domain] transitions={domain_counts} "
                f"advantage={'per-domain' if normalize_advantages_by_domain else 'global'} "
                f"loss={'macro-domain' if average_loss_by_domain else 'transition-mean'}"
            )

        idx_all = np.arange(len(states_all))
        for _ in range(ppo_epochs):
            np.random.shuffle(idx_all)
            for s in range(0, len(idx_all), minibatch_size):
                idx = idx_all[s : s + minibatch_size]
                batch_states = stack_states([states_all[i] for i in idx], device)
                batch_actions = actions_all[idx]
                batch_old_logps = old_logps_all[idx]
                batch_returns = returns_all[idx]
                batch_adv = adv_all[idx]
                batch_part_targets = final_part_targets_all[idx]
                batch_group_targets = final_group_targets_all[idx]
                batch_domain_ids = domain_ids_all[idx]

                logps, entropy, values, aux_preds = model.evaluate_actions_with_aux(
                    batch_states,
                    batch_actions,
                    temperature=policy_temperature,
                )
                ratio = torch.exp(logps - batch_old_logps)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_adv
                actor_losses = -torch.min(surr1, surr2)
                critic_losses = (values - batch_returns).square()
                aux_part_losses = (
                    aux_preds["final_part_miou"] - batch_part_targets
                ).square()
                aux_group_losses = (
                    aux_preds["final_group_excess"] - batch_group_targets
                ).square()
                aux_target_count_losses = (
                    aux_preds["target_group_count_ratio"]
                    - batch_states["target_group_count_ratio"].squeeze(-1)
                ).square()
                if average_loss_by_domain:
                    actor_loss = domain_macro_mean(actor_losses, batch_domain_ids)
                    critic_loss = domain_macro_mean(critic_losses, batch_domain_ids)
                    aux_part_loss = domain_macro_mean(aux_part_losses, batch_domain_ids)
                    aux_group_loss = domain_macro_mean(
                        aux_group_losses, batch_domain_ids
                    )
                    aux_target_count_loss = domain_macro_mean(
                        aux_target_count_losses, batch_domain_ids
                    )
                    entropy = domain_macro_mean(entropy, batch_domain_ids)
                else:
                    actor_loss = actor_losses.mean()
                    critic_loss = critic_losses.mean()
                    aux_part_loss = aux_part_losses.mean()
                    aux_group_loss = aux_group_losses.mean()
                    aux_target_count_loss = aux_target_count_losses.mean()
                    entropy = entropy.mean()
                loss = (
                    actor_loss
                    + vf_coef * critic_loss
                    + aux_part_loss_w * aux_part_loss
                    + aux_group_loss_w * aux_group_loss
                    + aux_target_count_loss_w * aux_target_count_loss
                    - ent_coef * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        avg_episode_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        epoch_index = int(epoch + 1)
        epoch_indices.append(epoch_index)
        epoch_avg_rewards.append(avg_episode_reward)
        if epoch_index % 50 == 0 or epoch_index == train_epochs:
            checkpoint_path = os.path.join(
                out_dir, f"sq_partnet_rl_epoch_{epoch_index:04d}.pt"
            )
            atomic_torch_save(
                make_training_checkpoint_payload(
                    model=model,
                    epoch=epoch_index,
                    epoch_indices=epoch_indices,
                    epoch_avg_rewards=epoch_avg_rewards,
                ),
                checkpoint_path,
            )
            print(f"[PPO] saved checkpoint: {checkpoint_path}")
        save_training_reward_curve(out_dir, epoch_avg_rewards, epoch_indices)
        print(
            f"[PPO] epoch={epoch_index:03d} avg_episode_reward={avg_episode_reward:.4f} "
            f"num_transitions={len(states_all)} "
            f"curve={os.path.join(out_dir, 'ppo_epoch_rewards.png')}"
        )

    if pool is not None:
        pool.shutdown()

    save_training_reward_curve(out_dir, epoch_avg_rewards, epoch_indices)
    return model


def _is_nonempty_file(path: Path) -> bool:
    """Return whether *path* is a regular, non-empty output file."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _complete_eval_episode_dir(
    uid_eval_dir: Path,
    sample_id: str,
) -> Optional[Path]:
    """Find a fully written evaluation episode for one UID.

    An evaluation can be interrupted after creating the UID/episode
    directory, or after writing only some artifacts.  Such a directory must
    not be treated as a completed result. ``episode_summary.json`` is written last
    by ``SQEditEnv.step``; we additionally require every deterministic core
    output and validate the summary before considering the UID complete.
    """
    if not uid_eval_dir.is_dir():
        return None

    required_suffixes = (
        "_initial_sq.npy",
        "_initial_sq.ply",
        "_initial_sq_provenance.json",
        "_grouped_multi_sq.npy",
        "_grouped_multi_sq.ply",
        "_grouped_multi_sq.json",
        "_final_sq.npy",
        "_final_sq.ply",
        "_final_sq.json",
    )
    episode_dirs = sorted(
        (path for path in uid_eval_dir.glob("episode_*") if path.is_dir()),
        reverse=True,
    )
    for episode_dir in episode_dirs:
        summary_path = episode_dir / "episode_summary.json"
        if not _is_nonempty_file(summary_path):
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        if str(summary.get("sample_id", "")) != str(sample_id):
            continue
        if not bool(summary.get("done", False)):
            continue
        if not isinstance(summary.get("trajectory"), list) or not summary["trajectory"]:
            continue

        required_files = [
            episode_dir / f"{sample_id}{suffix}" for suffix in required_suffixes
        ]
        if all(_is_nonempty_file(path) for path in required_files):
            return episode_dir
    return None


@torch.no_grad()
def evaluate_policy(
    model: StateEncoder,
    samples: Sequence[PartNetSample],
    action_catalog: ActionCatalog,
    out_dir: str,
    global_cfg: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    eval_env_kwargs = dict(env_kwargs)
    eval_env_kwargs["render_group_highlights"] = True
    eval_root = Path(out_dir) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    for sample_index, sample in enumerate(samples):
        uid_eval_dir = eval_root / str(sample.sample_id)
        complete_episode_dir = _complete_eval_episode_dir(
            uid_eval_dir,
            str(sample.sample_id),
        )
        if complete_episode_dir is not None:
            print(
                f"[EVAL] skip sample={sample.sample_id}: complete result under "
                f"{complete_episode_dir}"
            )
            continue

        env = SQEditEnv(
            sample,
            action_catalog,
            str(eval_root),
            global_cfg=global_cfg,
            episode_idx=sample_index,
            **eval_env_kwargs,
        )
        state = env.reset()
        done = False
        last_info: Dict[str, Any] = {}
        while not done:
            batch = stack_actor_states([state], device)
            action, _ = model.act_actor_only(batch, deterministic=True)
            state, _, done, info = env.step(int(action.item()))
            last_info = info
        final_part_miou = float(
            last_info.get("final_part_miou", last_info.get("after_part_miou", 0.0))
        )
        final_num_sq = float(
            last_info.get("final_num_primitives", last_info.get("num_primitives", 0.0))
        )
        summaries.append(last_info)
        print(
            f"[EVAL] sample={sample.sample_id} final_part_miou={final_part_miou:.4f} "
            f"num_sq={final_num_sq:.0f}"
        )
    return summaries
