import logging
import os
from typing import Dict, Optional

import torch.distributed as dist


def get_logger(name: str = "rl_trainer") -> logging.Logger:
    """Get a logger that only logs on rank 0 in distributed settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_stats(
    logger: logging.Logger,
    step: int,
    stats: Dict[str, float],
    wandb_run: Optional[object] = None,
):
    """Log training statistics to console and optionally wandb."""
    if not is_main_process():
        return

    parts = [f"step={step}"]
    for k, v in stats.items():
        parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(parts))

    if wandb_run is not None:
        wandb_run.log(stats, step=step)
