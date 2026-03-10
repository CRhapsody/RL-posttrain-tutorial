"""Entry point for RL post-training.

Usage:
    torchrun --nproc_per_node=N train.py --config configs/ppo_default.yaml
"""

import argparse

import yaml
import torch

from src.distributed import init_distributed, wrap_model_fsdp, cleanup_distributed
from src.models.policy import PolicyModel
from src.models.critic import CriticModel
from src.models.reference import ReferenceModel
from src.rollout.generator import RolloutGenerator
from src.rewards.base import LengthReward, RuleBasedReward
from src.data.dataset import PromptDataset, build_prompt_dataloader
from src.algorithms.ppo import PPOAlgorithm
from src.algorithms.grpo import GRPOAlgorithm
from src.trainer.rl_trainer import RLTrainer
from src.utils.logging import get_logger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_reward_fn(cfg: dict):
    reward_type = cfg["reward"]["type"]
    if reward_type == "length":
        return LengthReward(target_length=cfg["reward"].get("target_length", 200))
    elif reward_type == "rule":
        return RuleBasedReward(
            positive_keywords=cfg["reward"].get("positive_keywords", []),
            negative_keywords=cfg["reward"].get("negative_keywords", []),
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def build_dataset(cfg: dict) -> PromptDataset:
    data_cfg = cfg["data"]
    source = data_cfg["source"]

    if source == "list":
        return PromptDataset(data_cfg["prompts"])
    elif source == "hf_dataset":
        return PromptDataset.from_hf_dataset(
            data_cfg["dataset_name"],
            split=data_cfg.get("split", "train"),
            column=data_cfg.get("column", "prompt"),
        )
    elif source == "jsonl":
        return PromptDataset.from_jsonl(
            data_cfg["path"],
            column=data_cfg.get("column", "prompt"),
        )
    else:
        raise ValueError(f"Unknown data source: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger()

    # Distributed init
    local_rank = init_distributed()
    logger.info(f"Initialized rank {local_rank}")

    model_path = cfg["model"]["model_name_or_path"]
    tokenizer_name = cfg["model"].get("tokenizer_name") or model_path
    algo_name = cfg["algorithm"]["name"]

    # Build models
    policy = PolicyModel(model_path, tokenizer_name)
    ref_model = ReferenceModel(model_path)

    critic = None
    if algo_name == "ppo":
        critic = CriticModel(model_path)

    # Wrap with FSDP
    policy = wrap_model_fsdp(policy, cfg["fsdp"], device_id=local_rank)
    ref_model = wrap_model_fsdp(ref_model, cfg["fsdp"], device_id=local_rank)
    if critic is not None:
        critic = wrap_model_fsdp(critic, cfg["fsdp"], device_id=local_rank)

    # Optimizers
    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg["training"]["lr"]
    )
    critic_optimizer = None
    if critic is not None:
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(), lr=cfg["training"].get("critic_lr", cfg["training"]["lr"])
        )

    # Build algorithm
    algo_cfg = cfg["algorithm"]
    if algo_name == "ppo":
        algorithm = PPOAlgorithm(
            policy=policy,
            critic=critic,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clip_eps=algo_cfg.get("clip_eps", 0.2),
            vf_coef=algo_cfg.get("vf_coef", 0.5),
            kl_coef=algo_cfg.get("kl_coef", 0.1),
            gamma=algo_cfg.get("gamma", 1.0),
            lam=algo_cfg.get("lam", 0.95),
            ppo_epochs=algo_cfg.get("ppo_epochs", 4),
            num_mini_batches=algo_cfg.get("num_mini_batches", 1),
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
        )
    elif algo_name == "grpo":
        algorithm = GRPOAlgorithm(
            policy=policy,
            policy_optimizer=policy_optimizer,
            group_size=algo_cfg.get("group_size", 4),
            clip_eps=algo_cfg.get("clip_eps", 0.2),
            kl_coef=algo_cfg.get("kl_coef", 0.1),
            grpo_epochs=algo_cfg.get("grpo_epochs", 1),
            num_mini_batches=algo_cfg.get("num_mini_batches", 1),
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Build components
    reward_fn = build_reward_fn(cfg)
    dataset = build_dataset(cfg)
    dataloader = build_prompt_dataloader(
        dataset,
        policy.module.tokenizer if hasattr(policy, "module") else policy.tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["data"].get("max_prompt_length", 512),
    )

    gen_kwargs = {
        k: v for k, v in cfg.get("generation", {}).items()
        if k in ("max_new_tokens", "temperature", "top_p", "do_sample")
    }
    rollout_generator = RolloutGenerator(
        policy=policy,
        ref_model=ref_model,
        reward_fn=reward_fn,
        tokenizer=policy.module.tokenizer if hasattr(policy, "module") else policy.tokenizer,
        critic=critic,
        gen_kwargs=gen_kwargs,
    )

    # Optional wandb
    wandb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb
        wandb_run = wandb.init(
            project=cfg["wandb"].get("project", "rl-posttrain"),
            name=cfg["wandb"].get("name"),
            config=cfg,
        )

    # Build trainer and run
    trainer = RLTrainer(
        algorithm=algorithm,
        rollout_generator=rollout_generator,
        dataloader=dataloader,
        max_steps=cfg["training"]["max_steps"],
        log_interval=cfg["training"].get("log_interval", 1),
        save_interval=cfg["training"].get("save_interval", 100),
        save_dir=cfg["training"].get("save_dir"),
        wandb_run=wandb_run,
        group_size=algo_cfg.get("group_size", 4),
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
