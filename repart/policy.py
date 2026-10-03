from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .state_features import PRIMITIVE_FEATURE_DIM


class StateEncoder(nn.Module):
    """
    Asymmetric actor-critic.

    Actor only reads geometry-visible features:
      - point_sample
      - sq_params / sq_mask
      - prim_features
      - global_features
      - merge_action_features
      - action_mask
      - its own predicted target_group_count_ratio, supervised during training

    Critic additionally reads privileged per-primitive supervision statistics built
    from GT labels during training:
      - privileged_prim_stats
      - privileged_global_features

    This keeps the policy deployable without GT labels while still letting the
    critic learn a much less noisy value function.
    """

    def __init__(
        self,
        action_dim: int,
        max_prims: int,
        point_feat_dim: int = 128,
        sq_feat_dim: int = 128,
        prim_feat_dim: int = 64,
        privileged_feat_dim: int = 64,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        actor_token_dim: int = 128,
        prim_feature_input_dim: int = PRIMITIVE_FEATURE_DIM,
        global_feature_input_dim: int = 8,
        merge_action_input_dim: int = 17,
        privileged_feature_input_dim: int = 6,
        privileged_global_input_dim: int = 4,
        privileged_global_feat_dim: int = 32,
    ):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, point_feat_dim),
            nn.ReLU(),
        )
        self.point_pool_norm = nn.LayerNorm(point_feat_dim * 2)
        self.sq_mlp = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, sq_feat_dim),
            nn.ReLU(),
        )
        self.prim_mlp = nn.Sequential(
            nn.Linear(prim_feature_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, prim_feat_dim),
            nn.ReLU(),
        )
        self.privileged_mlp = nn.Sequential(
            nn.Linear(privileged_feature_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, privileged_feat_dim),
            nn.ReLU(),
        )
        self.critic_token_mlp = nn.Sequential(
            nn.Linear(11 + prim_feature_input_dim + privileged_feature_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, privileged_feat_dim),
            nn.ReLU(),
        )
        self.critic_token_score = nn.Linear(privileged_feat_dim, 1)
        self.privileged_global_mlp = nn.Sequential(
            nn.Linear(privileged_global_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, privileged_global_feat_dim),
            nn.ReLU(),
        )

        actor_global_dim = (
            point_feat_dim * 2
            + sq_feat_dim
            + prim_feat_dim
            + max_prims
            + global_feature_input_dim
        )
        self.actor_context = nn.Sequential(
            nn.Linear(actor_global_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
        )
        self.actor_token_mlp = nn.Sequential(
            nn.Linear(sq_feat_dim + prim_feat_dim, actor_token_dim),
            nn.ReLU(),
            nn.Linear(actor_token_dim, actor_token_dim),
            nn.ReLU(),
        )
        self.actor_target_group_count_head = nn.Sequential(
            nn.Linear(actor_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.merge_action_mlp = nn.Sequential(
            nn.Linear(merge_action_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.action_type_embed = nn.Embedding(2, 16)
        self.stop_head = nn.Sequential(
            nn.Linear(actor_hidden_dim + 1 + 16, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1),
        )
        self.merge_head = nn.Sequential(
            nn.Linear(
                actor_hidden_dim + 1 + actor_token_dim * 3 + 32 + 16, actor_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1),
        )

        critic_in_dim = (
            actor_hidden_dim + privileged_feat_dim + privileged_global_feat_dim
        )
        self.critic_trunk = nn.Sequential(
            nn.Linear(critic_in_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
        )
        self.critic = nn.Linear(critic_hidden_dim, 1)
        self.aux_final_part_head = nn.Linear(critic_hidden_dim, 1)
        self.aux_final_group_excess_head = nn.Linear(critic_hidden_dim, 1)

        self.max_prims = int(max_prims)
        self.action_dim = int(action_dim)
        self.privileged_feat_dim = int(privileged_feat_dim)
        self.privileged_global_input_dim = int(privileged_global_input_dim)
        self.actor_token_dim = int(actor_token_dim)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        features = features * mask.unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        return features.sum(dim=1) / denom

    def _actor_context(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points = state["point_sample"].float()
        sq = state["sq_params"].float()
        sq_mask = state["sq_mask"].float()
        prim_features = state["prim_features"].float()
        global_features = state["global_features"].float()

        point_center = points.mean(dim=1, keepdim=True)
        point_rel = points - point_center
        point_radius = torch.clamp(
            torch.linalg.norm(point_rel, dim=-1, keepdim=True).amax(
                dim=1, keepdim=True
            ),
            min=1e-6,
        )
        point_rel_norm = point_rel / point_radius
        point_dist = torch.linalg.norm(point_rel_norm, dim=-1, keepdim=True)
        point_input = torch.cat([points, point_rel_norm, point_dist], dim=-1)
        point_tokens = self.point_mlp(point_input)
        pfeat = self.point_pool_norm(
            torch.cat(
                [point_tokens.max(dim=1).values, point_tokens.mean(dim=1)], dim=-1
            )
        )
        sq_tokens = self.sq_mlp(sq)
        prim_tokens = self.prim_mlp(prim_features)
        sqfeat = self._masked_mean(sq_tokens, sq_mask)
        primfeat = self._masked_mean(prim_tokens, sq_mask)
        actor_tokens = self.actor_token_mlp(torch.cat([sq_tokens, prim_tokens], dim=-1))

        trunk_in = torch.cat(
            [pfeat, sqfeat, primfeat, sq_mask, global_features], dim=-1
        )
        return self.actor_context(trunk_in), actor_tokens

    def _actor_hidden(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        actor_h, _ = self._actor_context(state)
        return actor_h

    def actor_target_group_count_ratio(
        self, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        actor_h = self._actor_hidden(state)
        return torch.sigmoid(self.actor_target_group_count_head(actor_h)).squeeze(-1)

    def actor_logits(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        actor_h, actor_tokens = self._actor_context(state)
        batch_size = actor_h.shape[0]
        merge_action_features = state["merge_action_features"].float()
        merge_pair_indices = (
            state["merge_pair_indices"].long().clamp(0, self.max_prims - 1)
        )
        merge_pair_action_ids = (
            state["merge_pair_action_ids"].long().clamp(0, self.action_dim - 1)
        )
        merge_pair_mask = state["merge_pair_mask"].float()
        target_ratio = torch.sigmoid(
            self.actor_target_group_count_head(actor_h)
        ).squeeze(-1)
        target_ratio_1d = target_ratio.unsqueeze(-1)

        stop_type = self.action_type_embed(
            torch.zeros((batch_size,), dtype=torch.long, device=actor_h.device)
        )
        stop_logits = self.stop_head(
            torch.cat([actor_h, target_ratio_1d, stop_type], dim=-1)
        ).squeeze(-1)

        merge_feat = self.merge_action_mlp(merge_action_features)
        merge_type = self.action_type_embed(
            torch.ones(merge_pair_mask.shape, dtype=torch.long, device=actor_h.device)
        )
        token_dim = actor_tokens.shape[-1]
        idx_i = merge_pair_indices[..., 0].unsqueeze(-1).expand(-1, -1, token_dim)
        idx_j = merge_pair_indices[..., 1].unsqueeze(-1).expand(-1, -1, token_dim)
        merge_token_i = torch.gather(actor_tokens, 1, idx_i)
        merge_token_j = torch.gather(actor_tokens, 1, idx_j)
        num_candidates = merge_pair_mask.shape[1]
        merge_global = actor_h.unsqueeze(1).expand(-1, num_candidates, -1)
        merge_target_ratio = target_ratio_1d.unsqueeze(1).expand(-1, num_candidates, -1)
        merge_delta = torch.abs(merge_token_i - merge_token_j)
        merge_logits = self.merge_head(
            torch.cat(
                [
                    merge_global,
                    merge_target_ratio,
                    merge_token_i,
                    merge_token_j,
                    merge_delta,
                    merge_feat,
                    merge_type,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        logits = torch.full(
            (batch_size, self.action_dim),
            -1e9,
            dtype=actor_h.dtype,
            device=actor_h.device,
        )
        logits[:, 0] = stop_logits
        valid = merge_pair_mask > 0.0
        if bool(valid.any()):
            batch_idx = (
                torch.arange(batch_size, device=actor_h.device)
                .unsqueeze(1)
                .expand_as(merge_pair_action_ids)
            )
            logits[batch_idx[valid], merge_pair_action_ids[valid]] = merge_logits[valid]
        return logits

    def _masked_attention_pool(
        self,
        token_feat: torch.Tensor,
        token_mask: torch.Tensor,
        attn_logits: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = token_mask.float()
        masked_logits = attn_logits.masked_fill(token_mask <= 0, -1e9)
        attn = torch.softmax(masked_logits, dim=1)
        attn = attn * token_mask
        denom = torch.clamp(attn.sum(dim=1, keepdim=True), min=1e-6)
        attn = attn / denom
        return (token_feat * attn.unsqueeze(-1)).sum(dim=1)

    def _critic_hidden(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Keep privileged critic losses from shaping the shared actor encoder.
        actor_h = self._actor_hidden(state).detach()
        sq = state["sq_params"].float()
        prim_features = state["prim_features"].float()
        sq_mask = state["sq_mask"].float()
        privileged = state.get("privileged_prim_stats", None)
        if privileged is None:
            privileged = torch.zeros(
                (actor_h.shape[0], self.max_prims, 6),
                dtype=actor_h.dtype,
                device=actor_h.device,
            )
        else:
            privileged = privileged.float()
        privileged_global = state.get("privileged_global_features", None)
        if privileged_global is None:
            privileged_global = torch.zeros(
                (actor_h.shape[0], self.privileged_global_input_dim),
                dtype=actor_h.dtype,
                device=actor_h.device,
            )
        else:
            privileged_global = privileged_global.float()
        token_in = torch.cat([sq, prim_features, privileged], dim=-1)
        token_feat = self.critic_token_mlp(token_in)
        token_scores = self.critic_token_score(token_feat).squeeze(-1)
        privfeat = self._masked_attention_pool(token_feat, sq_mask, token_scores)
        priv_global_feat = self.privileged_global_mlp(privileged_global)
        critic_in = torch.cat([actor_h, privfeat, priv_global_feat], dim=-1)
        return self.critic_trunk(critic_in)

    def critic_outputs(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = self._critic_hidden(state)
        value = self.critic(hidden).squeeze(-1)
        aux = {
            "final_part_miou": torch.sigmoid(self.aux_final_part_head(hidden)).squeeze(
                -1
            ),
            "final_group_excess": F.softplus(
                self.aux_final_group_excess_head(hidden)
            ).squeeze(-1),
        }
        return value, aux

    def critic_value(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        value, _ = self.critic_outputs(state)
        return value

    def forward(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor_logits(state), self.critic_value(state)

    @staticmethod
    def temperature_scale_logits(
        masked_logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-6)
        if abs(temp - 1.0) < 1e-6:
            return masked_logits
        valid = torch.isfinite(masked_logits)
        scaled = masked_logits.clone()
        scaled[valid] = scaled[valid] / temp
        return scaled

    @staticmethod
    def apply_action_mask(
        logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        return logits.masked_fill(action_mask <= 0, -1e9)

    def act(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(state)
        value = self.critic_value(state)
        action_mask = state["action_mask"].float()
        masked_logits = self.apply_action_mask(logits, action_mask)
        sampled_logits = self.temperature_scale_logits(masked_logits, temperature)
        dist = torch.distributions.Categorical(logits=sampled_logits)
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def act_actor_only(
        self,
        actor_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(actor_state)
        action_mask = actor_state["action_mask"].float()
        masked_logits = self.apply_action_mask(logits, action_mask)
        sampled_logits = self.temperature_scale_logits(masked_logits, temperature)
        dist = torch.distributions.Categorical(logits=sampled_logits)
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp

    def evaluate_actions_with_aux(
        self,
        state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.actor_logits(state)
        value, aux = self.critic_outputs(state)
        aux = dict(aux)
        aux["target_group_count_ratio"] = self.actor_target_group_count_ratio(state)
        action_mask = state["action_mask"].float()
        masked_logits = self.apply_action_mask(logits, action_mask)
        eval_logits = self.temperature_scale_logits(masked_logits, temperature)
        dist = torch.distributions.Categorical(logits=eval_logits)
        logps = dist.log_prob(actions)
        entropy = dist.entropy()
        return logps, entropy, value, aux
