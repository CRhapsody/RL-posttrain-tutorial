import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import torch
import os 
from typing import Optional
import functools
from typing import Dict
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)


def init_distributed(tp_size: int = 1):
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


    # assign device mesh
    device_mesh = None
    if tp_size > 1:
        world_size = dist.get_world_size()
        assert world_size % tp_size == 0
        dp_size = world_size // tp_size
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
    return local_rank, device_mesh

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# llama model: model.model.layers = [
#     layers[0]: LlamaDecoderLayer,  
#     layers[1]: LlamaDecoderLayer,
#     ...
#     layers[31]: LlamaDecoderLayer,
# ]
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

# assign tp plan to each layer
def _assign_tp_plan(layer: nn.Module):
    cls_name = type(layer)
    for key, plan in _TP_PLANS.items():
        if key in cls_name:
            return plan
    return None

# apply tensor parallel to one decoder layer
def apply_tensor_parallel(model: nn.Module, device_mesh: DeviceMesh):
    tp_mesh = device_mesh["tp"]
    layers = _get_decoder_layers(model)
    if layers is None:
        raise ValueError("Cannot locate transformer decoder layers for TP. Pass tp_plan explicitly or set tp_size=1.")
    
    for layer in layers:
        tp_plan = _assign_tp_plan(layer)
        if tp_plan is None:
            raise ValueError(f"No built-in TP plan for {type(layer).__name__}. Please provide tp_plan explicitly.")
        parallelize_module(layer, tp_mesh, tp_plan)


def _get_decoder_layers(model: nn.Module):
    for backbone_attr, layer_attr in _BACKBONE_SEARCH_PATHS:
        backbone = getattr(model, backbone_attr, None)
        if backbone is not None:
            layers = getattr(backbone, layer_attr, None)
            if layers is not None and len(layers) > 0:
                return layers
    return None

def get_transformer_layer_cls(model: nn.Module):
    layers = _get_decoder_layers(model)
    if layers is not None and len(layers) > 0:
        return type(layers[0])
    return None

def _build_mixed_precision(mode: str) -> Optional[MixedPrecision]:
    if mode == 'bf16':
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32, # fp32 or bf16
        )
    elif mode == 'fp16':
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float32, # fp32 or fp16
        )
    return None

def wrap_model_fsdp(model: nn.Module, device_mesh: DeviceMesh, cfg: dict):
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    if device_mesh is not None:
        process_group = device_mesh["dp"].get_group()
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(
        cfg.get("sharding_strategy", "full_shard"), 
        ShardingStrategy.FULL_SHARD)

    mp_policy= _build_mixed_precision(cfg.get("mixed_precision", "bf16"))
    layer_cls = get_transformer_layer_cls(model) # auto-detect the transformer layer class for FSDP wrapping, in this case, llama
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )


    cpu_offload = (
        CPUOffload(offload_params=True) if cfg.get("cpu_offload", False) else None
    )

    return FSDP(model, 
                auto_wrap_policy=auto_wrap, 
                cpu_offload=cpu_offload,
                device_mesh=device_mesh,
                sharding_strategy=sharding_strategy,
                mixed_precision=mp_policy,
                use_orig_params=True,
                process_group=process_group,
                device_id=device_id,
            )