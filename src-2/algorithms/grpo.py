from typing import Dict

import torch
import torch.nn as nn

from src.algorithms.base import RLAlgorithm
from src.models.policy import PolicyModel
from src.rollout.generator import RolloutBatch


class GRPOAlgorithm(RLAlgorithm):
    """Group Relative Policy Optimization.

    No critic needed. Advantages are computed by normalizing rewards
    within each group of responses generated for the same prompt.
    """

    def __init__(
        self,
        policy: PolicyModel,
        policy_optimizer: torch.optim.Optimizer,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.1,
        grpo_epochs: int = 1,
        num_mini_batches: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.policy_optimizer = policy_optimizer

        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.grpo_epochs = grpo_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm

    def compute_advantages(self, batch: RolloutBatch) -> RolloutBatch:
        """Compute group-normalized advantages.

        For GRPO, the batch has B*G samples where every `group_size`
        consecutive samples share the same prompt. The advantage for each
        response is: (reward - mean_reward) / (std_reward + eps).
        """
        rewards = batch.rewards  # (B*G,)
        G = self.group_size
        B = rewards.shape[0] // G

        grouped = rewards.view(B, G)
        mean_r = grouped.mean(dim=1, keepdim=True)
        std_r = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        norm_rewards = (grouped - mean_r) / std_r  # (B, G)
        advantages_scalar = norm_rewards.view(-1)  # (B*G,)

        # Expand scalar advantage to per-token: same advantage for all response tokens
        T = batch.labels_mask.shape[1]
        advantages = advantages_scalar.unsqueeze(1).expand(-1, T)
        advantages = advantages * batch.labels_mask.float()

        batch.advantages = advantages
        return batch

    def compute_loss(
        self,
        batch: RolloutBatch,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute GRPO clipped surrogate loss."""
        new_log_probs = self.policy.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels_mask=batch.labels_mask,
        )

        adv = batch.advantages[:, 1:]  # align with shifted log probs
        response_mask = batch.labels_mask[:, 1:].float()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # KL penalty (per-token KL between policy and reference)
        kl = batch.log_probs - batch.ref_log_probs  # old policy vs ref
        kl_loss = (kl * response_mask).sum() / response_mask.sum().clamp(min=1)

        total_loss = policy_loss + self.kl_coef * kl_loss

        approx_kl = (old_log_probs - new_log_probs).mean()

        return {
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "approx_kl": approx_kl,
        }

    def update_step(self, batch: RolloutBatch) -> Dict[str, float]:
        """Run GRPO policy updates."""
        batch = self.compute_advantages(batch)

        old_log_probs = batch.log_probs.detach().clone()

        stats_accum = {"policy_loss": 0.0, "kl_loss": 0.0, "approx_kl": 0.0}
        num_updates = 0

        for epoch in range(self.grpo_epochs):
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
                )
                mb_batch.advantages = batch.advantages[mb_idx]

                losses = self.compute_loss(mb_batch, old_log_probs=old_log_probs[mb_idx])

                self.policy_optimizer.zero_grad()
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                for k in stats_accum:
                    stats_accum[k] += losses[k].item()
                num_updates += 1

        return {k: v / max(num_updates, 1) for k, v in stats_accum.items()}
