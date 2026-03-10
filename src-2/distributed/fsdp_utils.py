import os
import functools
from typing import Optional, Type

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel


def init_distributed() -> int:
    """Initialize the distributed process group and return local rank."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_transformer_layer_cls(model: PreTrainedModel) -> Optional[Type]:
    """Auto-detect the transformer layer class for FSDP wrapping.

    Walks common HF model architectures to find the repeated decoder layer.
    """
    for attr in ("model", "transformer", "gpt_neox"):
        backbone = getattr(model, attr, None)
        if backbone is not None:
            layers = getattr(backbone, "layers", None) or getattr(backbone, "h", None)
            if layers is not None and len(layers) > 0:
                return type(layers[0])
    return None


def wrap_model_fsdp(
    model: PreTrainedModel,
    cfg: dict,
    device_id: Optional[int] = None,
) -> FSDP:
    """Wrap a HuggingFace model with FSDP.

    Args:
        model: The HuggingFace model to wrap.
        cfg: FSDP configuration dict with keys:
            - sharding_strategy: "full_shard" | "shard_grad_op" | "no_shard"
            - mixed_precision: "bf16" | "fp16" | "none"
            - cpu_offload: bool
        device_id: CUDA device id for this process.
    """
    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(
        cfg.get("sharding_strategy", "full_shard"),
        ShardingStrategy.FULL_SHARD,
    )

    mp_policy = _build_mixed_precision(cfg.get("mixed_precision", "bf16"))

    layer_cls = get_transformer_layer_cls(model)
    auto_wrap = None
    if layer_cls is not None:
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )

    cpu_offload = CPUOffload(offload_params=True) if cfg.get("cpu_offload", False) else None

    wrapped = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        cpu_offload=cpu_offload,
        device_id=device_id,
        use_orig_params=True,
    )
    return wrapped


def _build_mixed_precision(mode: str) -> Optional[MixedPrecision]:
    if mode == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif mode == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    return None
