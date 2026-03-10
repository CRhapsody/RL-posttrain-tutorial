from typing import Dict

import torch
import torch.nn as nn

from src.algorithms.base import RLAlgorithm
from src.models.policy import PolicyModel
from src.models.critic import CriticModel
from src.rollout.generator import RolloutBatch


class PPOAlgorithm(RLAlgorithm):
    """Proximal Policy Optimization with clipped surrogate objective and GAE."""

    def __init__(
        self,
        policy: PolicyModel,
        critic: CriticModel,
        policy_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        kl_coef: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.95,
        ppo_epochs: int = 4,
        num_mini_batches: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.critic = critic
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer

        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm

    def compute_advantages(self, batch: RolloutBatch) -> RolloutBatch:
        """Compute GAE advantages using rewards and critic values.

        For token-level RL the reward is sparse (only at the last response token),
        so we distribute it and compute GAE along the response positions.
        """
        values = batch.values  # (B, T)
        labels_mask = batch.labels_mask  # (B, T)
        rewards_scalar = batch.rewards  # (B,)

        B, T = values.shape
        device = values.device

        # Place the scalar reward at the last valid response token for each sample
        token_rewards = torch.zeros(B, T, device=device)
        for i in range(B):
            resp_positions = labels_mask[i].nonzero(as_tuple=True)[0]
            if len(resp_positions) > 0:
                token_rewards[i, resp_positions[-1]] = rewards_scalar[i]

        # Apply KL penalty per token: r_t = r_t - kl_coef * (log_pi - log_ref)
        kl_per_token = batch.log_probs - batch.ref_log_probs  # (B, T-1)
        # Align dimensions: token_rewards is (B, T), kl is (B, T-1)
        token_rewards[:, 1:] = token_rewards[:, 1:] - self.kl_coef * kl_per_token

        # GAE computation
        advantages = torch.zeros(B, T, device=device)
        last_gae = torch.zeros(B, device=device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=device)
            else:
                next_value = values[:, t + 1]

            delta = token_rewards[:, t] + self.gamma * next_value - values[:, t]
            last_gae = delta + self.gamma * self.lam * last_gae
            last_gae = last_gae * labels_mask[:, t].float()
            advantages[:, t] = last_gae

        returns = advantages + values

        batch.advantages = advantages
        batch.returns = returns
        return batch

    def compute_loss(
        self,
        batch: RolloutBatch,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO clipped objective and value function loss."""
        # Recompute current log probs and values
        new_log_probs = self.policy.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels_mask=batch.labels_mask,
        )
        new_values = self.critic.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels_mask=batch.labels_mask,
        )

        # Policy loss (clipped surrogate)
        # Align shapes: advantages is (B, T), log_probs is (B, T-1)
        adv = batch.advantages[:, 1:]  # align with shifted log probs
        response_mask = batch.labels_mask[:, 1:].float()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Value loss (clipped)
        ret = batch.returns
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.clip_eps, self.clip_eps
        )
        vf_loss1 = (new_values - ret) ** 2
        vf_loss2 = (value_pred_clipped - ret) ** 2
        labels_float = batch.labels_mask.float()
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        value_loss = (value_loss * labels_float).sum() / labels_float.sum().clamp(min=1)

        # KL for logging
        approx_kl = (old_log_probs - new_log_probs).mean()

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": policy_loss + self.vf_coef * value_loss,
            "approx_kl": approx_kl,
        }

    def update_step(self, batch: RolloutBatch) -> Dict[str, float]:
        """Run multiple epochs of mini-batch PPO updates."""
        batch = self.compute_advantages(batch)

        old_log_probs = batch.log_probs.detach().clone()
        old_values = batch.values.detach().clone()

        stats_accum = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0}
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(batch.input_ids.shape[0], device=batch.input_ids.device)
            mb_size = max(1, batch.input_ids.shape[0] // self.num_mini_batches)

            for start in range(0, batch.input_ids.shape[0], mb_size):
                end = min(start + mb_size, batch.input_ids.shape[0])
                mb_idx = indices[start:end]

                mb_batch = RolloutBatch(
                    prompt_ids=batch.prompt_ids[mb_idx],
                    response_ids=batch.response_ids[mb_idx],
                    input_ids=batch.input_ids[mb_idx],
                    attention_mask=batch.attention_mask[mb_idx],
                    labels_mask=batch.labels_mask[mb_idx],
                    log_probs=batch.log_probs[mb_idx],
                    ref_log_probs=batch.ref_log_probs[mb_idx],
                    rewards=batch.rewards[mb_idx],
                    values=batch.values[mb_idx],
                )
                mb_batch.advantages = batch.advantages[mb_idx]
                mb_batch.returns = batch.returns[mb_idx]

                losses = self.compute_loss(
                    mb_batch,
                    old_log_probs=old_log_probs[mb_idx],
                    old_values=old_values[mb_idx],
                )

                # Update policy
                self.policy_optimizer.zero_grad()
                losses["policy_loss"].backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                (self.vf_coef * losses["value_loss"]).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                for k in stats_accum:
                    stats_accum[k] += losses[k].item()
                num_updates += 1

        return {k: v / max(num_updates, 1) for k, v in stats_accum.items()}
