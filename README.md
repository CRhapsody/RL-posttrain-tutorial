# RL Post-Training Framework

A minimal reinforcement learning post-training framework for LLMs, supporting PPO and GRPO algorithms with PyTorch FSDP.

## Features

- **PPO** (Proximal Policy Optimization): Actor-Critic architecture with GAE
- **GRPO** (Group Relative Policy Optimization): Critic-free, group-based advantage estimation
- **FSDP**: Full Sharded Data Parallel for multi-GPU training
- **HuggingFace Transformers** integration for model loading

## Quick Start

```bash
pip install -r requirements.txt

# Single-node multi-GPU training
torchrun --nproc_per_node=4 train.py --config configs/ppo_default.yaml
torchrun --nproc_per_node=4 train.py --config configs/grpo_default.yaml
```

## Project Structure

```
├── configs/              # YAML configuration files
├── src/
│   ├── algorithms/       # PPO, GRPO implementations
│   ├── models/           # Policy, Critic, Reference model wrappers
│   ├── rollout/          # Rollout generation
│   ├── rewards/          # Reward functions
│   ├── data/             # Dataset loading
│   ├── distributed/      # FSDP utilities
│   ├── trainer/          # Training loop orchestration
│   └── utils/            # Logging and helpers
└── train.py              # Entry point
```
