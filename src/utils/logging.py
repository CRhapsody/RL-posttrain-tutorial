import logging
import os
from typing import Dict, Optional

import torch.distributed as dist

def get_logger(name: str = "rl_trainer") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logging.setLevel(logging.INFO)
    return logging

# logging is only on main process
def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.rank() == 0

# example:
# log_stats(logger, step=42, stats={"loss": 0.352, "reward_mean": 1.23})
# 输出: [2026-03-11 10:30:00] [INFO] step=42 | loss=0.3520 | reward_mean=1.2300
def log_stats(logger: logging.Logger, 
                step: int, 
                stats: Dict[str, float],
                wandb_run: Optional[object] = None
                ):
    if not is_main_process():
        return
    parts = [f"step={step}"]
    for k, v in stats.items():
        parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(parts))

    if wandb_run is not None:
        wandb_run.log(stats, step=step)
