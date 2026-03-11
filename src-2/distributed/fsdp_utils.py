import os
import functools
from typing import Dict, Optional, Set, Type

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from transformers import PreTrainedModel


def init_distributed(
    tp_size: int = 1,
) -> tuple:
    """Initialize distributed process group and optional 2D device mesh.

    Args:
        tp_size: Tensor parallel degree. 1 means TP disabled (FSDP only).
            world_size must be divisible by tp_size.

    Returns:
        (local_rank, device_mesh) where device_mesh is None when tp_size=1.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    device_mesh = None
    if tp_size > 1:
        world_size = dist.get_world_size()
        assert world_size % tp_size == 0, (
            f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        )
        dp_size = world_size // tp_size
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )

    return local_rank, device_mesh


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ─── Tensor Parallelism ──────────────────────────────────────────────

# Maps (model_attr, layer_attr) pairs for locating decoder layers
_BACKBONE_SEARCH_PATHS = (
    ("model", "layers"),   # Llama, Qwen, Mistral
    ("transformer", "h"),  # GPT-2, GPT-J
    ("gpt_neox", "layers"),  # GPT-NeoX
)

# Common TP plans for different HF model architectures.
# key = module attr path inside one decoder layer → TP style
_TP_PLANS: Dict[str, Dict[str, object]] = {
    "llama": {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    },
    "gpt2": {
        "attn.c_attn": ColwiseParallel(),
        "attn.c_proj": RowwiseParallel(),
        "mlp.c_fc": ColwiseParallel(),
        "mlp.c_proj": RowwiseParallel(),
    },
}


def _detect_tp_plan(layer: nn.Module) -> Optional[Dict[str, object]]:
    """Try to match a decoder layer against known TP plans."""
    cls_name = type(layer).__name__.lower()
    for key, plan in _TP_PLANS.items():
        if key in cls_name:
            return plan
    # first_key = next(iter(_TP_PLANS.values()))
    # sample_attr = next(iter(first_key)).split(".")[0]
    # if hasattr(layer, sample_attr):
    #     return first_key
    return None


def apply_tensor_parallel(
    model: PreTrainedModel,
    device_mesh,
    tp_plan: Optional[Dict[str, object]] = None,
):
    """Apply tensor parallelism to each transformer layer in the model.

    Args:
        model: HuggingFace model (before FSDP wrapping).
        device_mesh: 2D DeviceMesh with ("dp", "tp") dimensions.
        tp_plan: Explicit TP plan mapping submodule paths to parallel styles.
            If None, auto-detects based on model architecture.
    """
    tp_mesh = device_mesh["tp"]

    layers = _get_decoder_layers(model)
    if layers is None:
        raise ValueError(
            "Cannot locate transformer decoder layers for TP. "
            "Pass tp_plan explicitly or set tp_size=1."
        )

    if tp_plan is None:
        tp_plan = _detect_tp_plan(layers[0])
        if tp_plan is None:
            raise ValueError(
                f"No built-in TP plan for {type(layers[0]).__name__}. "
                "Please provide tp_plan explicitly."
            )

    for layer in layers:
        parallelize_module(layer, tp_mesh, tp_plan)


def _get_decoder_layers(model: PreTrainedModel) -> Optional[nn.ModuleList]:
    """Walk HF model hierarchy to find the repeated decoder layer list."""
    for backbone_attr, layer_attr in _BACKBONE_SEARCH_PATHS:
        backbone = getattr(model, backbone_attr, None)
        if backbone is not None:
            layers = getattr(backbone, layer_attr, None)
            if layers is not None and len(layers) > 0:
                return layers
    return None


# ─── FSDP Wrapping ───────────────────────────────────────────────────

def get_transformer_layer_cls(model: PreTrainedModel) -> Optional[Type]:
    """Auto-detect the transformer layer class for FSDP wrapping."""
    layers = _get_decoder_layers(model)
    if layers is not None and len(layers) > 0:
        return type(layers[0])
    return None


def wrap_model_fsdp(
    model: PreTrainedModel,
    cfg: dict,
    device_id: Optional[int] = None,
    device_mesh=None,
) -> FSDP:
    """Wrap a HuggingFace model with FSDP.

    If device_mesh is provided (2D parallelism), FSDP shards along the
    data-parallel dimension only.

    Args:
        model: The HuggingFace model to wrap.
        cfg: FSDP configuration dict with keys:
            - sharding_strategy: "full_shard" | "shard_grad_op" | "no_shard"
            - mixed_precision: "bf16" | "fp16" | "none"
            - cpu_offload: bool
        device_id: CUDA device id for this process.
        device_mesh: Optional 2D DeviceMesh for TP+FSDP.
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

    cpu_offload = (
        CPUOffload(offload_params=True) if cfg.get("cpu_offload", False) else None
    )

    fsdp_kwargs = dict(
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        cpu_offload=cpu_offload,
        device_id=device_id,
        use_orig_params=True,
    )

    if device_mesh is not None:
        fsdp_kwargs["process_group"] = device_mesh["dp"].get_group()

    return FSDP(model, **fsdp_kwargs)


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
