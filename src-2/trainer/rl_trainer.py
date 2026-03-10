from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.algorithms.base import RLAlgorithm
from src.algorithms.ppo import PPOAlgorithm
from src.algorithms.grpo import GRPOAlgorithm
from src.rollout.generator import RolloutGenerator
from src.utils.logging import get_logger, log_stats


class RLTrainer:
    """Orchestrates the RL post-training loop.

    Coordinates data loading, rollout generation, and algorithm updates.
    """

    def __init__(
        self,
        algorithm: RLAlgorithm,
        rollout_generator: RolloutGenerator,
        dataloader: DataLoader,
        max_steps: int = 1000,
        log_interval: int = 1,
        save_interval: int = 100,
        save_dir: Optional[str] = None,
        wandb_run: Optional[object] = None,
        group_size: int = 4,
    ):
        self.algorithm = algorithm
        self.rollout_generator = rollout_generator
        self.dataloader = dataloader
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.wandb_run = wandb_run
        self.group_size = group_size
        self.logger = get_logger()

    def train(self):
        """Main training loop."""
        self.logger.info(
            f"Starting RL training | algorithm={type(self.algorithm).__name__} "
            f"| max_steps={self.max_steps}"
        )

        is_grpo = isinstance(self.algorithm, GRPOAlgorithm)
        step = 0
        data_iter = iter(self.dataloader)

        while step < self.max_steps:
            # Get next batch of prompts (cycle if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                if hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(step)
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            prompt_ids = batch["prompt_ids"].cuda()
            prompt_mask = batch["prompt_mask"].cuda()

            # Generate rollouts
            if is_grpo:
                rollout_batch = self.rollout_generator.generate_group_rollouts(
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    group_size=self.group_size,
                )
            else:
                rollout_batch = self.rollout_generator.generate_rollouts(
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                )

            # Policy update
            train_stats = self.algorithm.update_step(rollout_batch)

            # Add reward stats
            train_stats["reward_mean"] = rollout_batch.rewards.mean().item()
            train_stats["reward_std"] = rollout_batch.rewards.std().item()
            train_stats["response_len_mean"] = rollout_batch.response_ids.shape[1]

            step += 1

            if step % self.log_interval == 0:
                log_stats(self.logger, step, train_stats, self.wandb_run)

            if self.save_dir and step % self.save_interval == 0:
                self._save_checkpoint(step)

        self.logger.info("Training complete.")

    def _save_checkpoint(self, step: int):
        """Save model checkpoint (basic implementation)."""
        import os
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

        save_path = os.path.join(self.save_dir, f"step_{step}")
        os.makedirs(save_path, exist_ok=True)

        policy = self.algorithm.policy
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT):
            state_dict = policy.state_dict()
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, os.path.join(save_path, "policy.pt"))
                self.logger.info(f"Checkpoint saved to {save_path}")
