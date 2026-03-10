from dataclasses import dataclass
from typing import Optional, List

import torch

from src.models.policy import PolicyModel
from src.models.critic import CriticModel
from src.models.reference import ReferenceModel
from src.rewards.base import RewardFunction


@dataclass
class RolloutBatch:
    """Container for all data produced by a single rollout step."""
    prompt_ids: torch.Tensor          # (B, T_prompt)
    response_ids: torch.Tensor        # (B, T_response)
    input_ids: torch.Tensor           # (B, T_prompt + T_response) full sequence
    attention_mask: torch.Tensor      # (B, T_total)
    labels_mask: torch.Tensor         # (B, T_total) True for response positions
    log_probs: torch.Tensor           # (B, T_total - 1) policy log probs
    ref_log_probs: torch.Tensor       # (B, T_total - 1) reference log probs
    rewards: torch.Tensor             # (B,) scalar reward per sequence
    values: Optional[torch.Tensor] = None  # (B, T_total) critic values, PPO only


class RolloutGenerator:
    """Manages the data collection phase of an RL training iteration.

    Given a batch of prompts, uses the policy to generate responses, then
    collects all signals needed for the policy update step.
    """

    def __init__(
        self,
        policy: PolicyModel,
        ref_model: ReferenceModel,
        reward_fn: RewardFunction,
        tokenizer,
        critic: Optional[CriticModel] = None,
        gen_kwargs: Optional[dict] = None,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.critic = critic
        self.gen_kwargs = gen_kwargs or {}

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> RolloutBatch:
        """Generate responses and collect all training signals.

        Args:
            prompt_ids: (B, T_prompt) tokenized prompts.
            prompt_mask: (B, T_prompt) attention mask for prompts.

        Returns:
            RolloutBatch with all fields populated.
        """
        device = prompt_ids.device
        batch_size, prompt_len = prompt_ids.shape

        # Step 1: Generate responses
        full_ids = self.policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            **self.gen_kwargs,
        )
        total_len = full_ids.shape[1]
        response_ids = full_ids[:, prompt_len:]

        # Build attention mask and labels mask for the full sequence
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
        labels_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        labels_mask[:, prompt_len:] = True
        # Exclude padding from labels
        labels_mask = labels_mask & attention_mask.bool()

        # Step 2: Compute policy log probs
        log_probs = self.policy.forward(
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels_mask=labels_mask,
        )

        # Step 3: Compute reference log probs
        ref_log_probs = self.ref_model.forward(
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels_mask=labels_mask,
        )

        # Step 4: Compute rewards
        prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        reward_list = self.reward_fn.compute(prompt_texts, response_texts)
        rewards = torch.tensor(reward_list, dtype=torch.float32, device=device)

        # Step 5: Compute values (PPO only)
        values = None
        if self.critic is not None:
            values = self.critic.forward(
                input_ids=full_ids,
                attention_mask=attention_mask,
                labels_mask=labels_mask,
            )

        return RolloutBatch(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels_mask=labels_mask,
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            values=values,
        )

    @torch.no_grad()
    def generate_group_rollouts(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        group_size: int = 4,
    ) -> RolloutBatch:
        """Generate multiple responses per prompt for GRPO.

        Repeats each prompt `group_size` times, generates responses, then
        returns a RolloutBatch where batch dimension is B * group_size.
        """
        batch_size = prompt_ids.shape[0]

        # Repeat each prompt group_size times: (B, T) -> (B*G, T)
        expanded_ids = prompt_ids.repeat_interleave(group_size, dim=0)
        expanded_mask = prompt_mask.repeat_interleave(group_size, dim=0)

        return self.generate_rollouts(expanded_ids, expanded_mask)
