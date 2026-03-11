from src.algorithms.ppo import PPOAlgorithm
from src.algorithms.grpo import GRPOAlgorithm
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
import torch


class RLTrainer():
    def __init__(
        self, 
        algorithm, 
        rollout_generator, 
        train_loader,
        group_size,
        max_train_steps,
        log_interval,
        save_interval,
        save_dir,
    ):
        self.algorithm = algorithm
        self.rollout_generator = rollout_generator
        self.train_loader = train_loader
        self.group_size = group_size
        # self.max_train_steps = max_train_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = save_dir


    def train(self):
        self.step = 0
        

        # while self.step < self.max_train_steps:
            # try:# get next batch of prompts (cycle if exhausted)
            #     batch = next(self.dataloader)
            # except StopIteration:
            #     # TODO: (cycle if exhausted)
            #     return None
        for epoch in range(self.max_train_steps):
            train_loader = self.train_loader.sample.set_epoch(epoch) # TODO prepare for distributed sampling, if do not set, will be the same for all epochs
            for batch in train_loader:
                prompt_ids = batch["prompt_ids"].cuda()
                prompt_mask = batch["prompt_mask"].cuda()
        
                train_stats = {}

                # get next batch of prompts (cycle if exhausted)
                # rollout generator
                if isinstance(self.algorithm, GRPOAlgorithm):
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
                
                # compute advantages
                if isinstance(self.algorithm, GRPOAlgorithm):
                    train_stats["advantages"] = self.algorithm.compute_advantages(rollout_batch)
                else:
                    train_stats["advantages"] = self.algorithm.compute_advantages(rollout_batch)


                # policy update
                train_stats = self.algorithm.update_step(rollout_batch)

                # if ppo, update critic
                if isinstance(self.algorithm, PPOAlgorithm):
                    self.algorithm.update_critic(rollout_batch)
                
                # step +=1
                self.step += 1

                # log stats
                if self.step % self.log_interval == 0:
                    self.logger.info(f"Step {self.step}: {train_stats}")

                # save checkpoint
                if self.save_dir and self.step % self.save_interval == 0:
                    self._save_checkpoint(self, step=self.step)


    def _save_checkpoint(self, step: int):
        """Save model checkpoint (basic implementation for fsdp)."""
        save_path = os.path.join(self.save_dir, f"step_{step}")
        os.makedirs(save_path, exist_ok=True)

        # save policy
        policy = self.algorithm.policy # TODO: may be other objects not in algorithm
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT):
            state_dict = policy.state_dict()
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, os.path.join(save_path, "policy.pt"))
                self.logger.info(f"Checkpoint saved to {save_path}")
