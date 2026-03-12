# RL 后训练仓库 - 模块编写顺序指南

## 当前仓库状态

- `src/` 目录：只有 `algorithms/` 和 `trainer/` 的骨架/桩代码，大量模块缺失
- `src-2/` 目录：包含完整的参考实现
- `train.py`：入口文件已完成，导入的是 `src.*` 模块

## 模块依赖关系图

```
train.py (入口)
  ├── src/utils/logging.py          ← 被几乎所有模块使用
  ├── src/distributed/fsdp_utils.py ← 被 train.py 直接调用
  ├── src/models/
  │   ├── policy.py      ← 被 rollout、algorithms 使用
  │   ├── reference.py   ← 被 rollout 使用
  │   └── critic.py      ← 被 rollout、PPO 使用
  ├── src/rewards/base.py           ← 被 rollout 使用
  ├── src/data/dataset.py           ← 被 train.py 直接调用
  ├── src/rollout/generator.py      ← 依赖 models + rewards
  ├── src/algorithms/
  │   ├── base.py        ← 抽象基类
  │   ├── ppo.py         ← 依赖 models + rollout
  │   └── grpo.py        ← 依赖 models + rollout
  └── src/trainer/rl_trainer.py     ← 依赖以上所有模块
```

## 推荐编写顺序（自底向上）

### 第一层：无依赖的基础工具模块

| 顺序 | 模块 | 说明 |
|------|------|------|
| **1** | `src/utils/logging.py` | 日志工具，被全局使用，零外部依赖，最先写 |
| **2** | `src/distributed/fsdp_utils.py` | 分布式初始化/FSDP 包装，只依赖 PyTorch，独立性强 |

**理由**：这两个是"基础设施"模块，几乎被所有上层模块引用。先完成它们，后续开发时日志和分布式环境就可以直接使用，也便于调试。

### 第二层：模型封装层

| 顺序 | 模块 | 说明 |
|------|------|------|
| **3** | `src/models/policy.py` | Policy 模型，核心角色，提供 `forward()` 和 `generate()` |
| **4** | `src/models/reference.py` | Reference 模型（冻结副本），用于 KL 散度计算 |
| **5** | `src/models/critic.py` | Critic 模型（值函数头），PPO 专用 |

**理由**：模型层是 RL 管线的"原子单元"。Policy 是最核心的模型，Reference 和 Critic 都是它的变体。写完后可以独立测试——加载一个小模型验证 forward/generate 是否正常工作。

### 第三层：数据与奖励

| 顺序 | 模块 | 说明 |
|------|------|------|
| **6** | `src/rewards/base.py` | 奖励函数（`LengthReward`, `RuleBasedReward`），纯 Python 逻辑 |
| **7** | `src/data/dataset.py` | Prompt 数据集和 DataLoader 构建 |

**理由**：奖励函数几乎没有依赖（纯字符串/数值计算），数据模块只依赖 tokenizer。这两个可以并行开发，写完后可以独立做单元测试。

### 第四层：Rollout 生成器

| 顺序 | 模块 | 说明 |
|------|------|------|
| **8** | `src/rollout/generator.py` | 串联 policy/ref/critic/reward，生成 `RolloutBatch` |

**理由**：Rollout 是整个 RL 管线的"数据收集阶段"，它调用模型生成响应、计算 log probs、计算奖励、收集 values。必须在模型层和奖励层都完成后才能编写。`RolloutBatch` 数据类定义了后续算法所需的所有张量格式。

### 第五层：RL 算法

| 顺序 | 模块 | 说明 |
|------|------|------|
| **9** | `src/algorithms/base.py` | 抽象基类，定义 `compute_advantages` / `compute_loss` / `update_step` 接口 |
| **10** | `src/algorithms/grpo.py` | GRPO 算法，比 PPO 简单（无 critic，组内归一化优势） |
| **11** | `src/algorithms/ppo.py` | PPO 算法，包含 GAE、clipped surrogate loss、value loss、mini-batch 更新 |

**理由**：先写抽象基类确定接口契约，再实现具体算法。**GRPO 比 PPO 简单**（不需要 critic、不需要 GAE），建议先实现 GRPO 验证管线跑通，再实现 PPO。

### 第六层：训练器（最后编写）

| 顺序 | 模块 | 说明 |
|------|------|------|
| **12** | `src/trainer/rl_trainer.py` | 训练主循环，编排所有组件 |

**理由**：Trainer 是最顶层的"编排者"，它调用 dataloader → rollout → algorithm → logging → checkpoint。只有当所有下层模块都完成后，才能正确编写和测试 Trainer。

## 关键原则

1. **自底向上**：从无依赖的叶子模块开始，逐层向上构建
2. **每层可测试**：每完成一层，写个简单脚本验证该层的功能
3. **接口先行**：先定义好 `RLAlgorithm` 基类和 `RolloutBatch` 数据结构，再填充实现
4. **先 GRPO 后 PPO**：GRPO 更简单，适合先跑通整个管线

## logging 模块详解

### Python `logging` 基础知识

Python 的 `logging` 模块是标准库自带的日志系统，比 `print()` 更强大。核心概念有四个：

| 概念 | 说明 | 类比 |
|------|------|------|
| **Logger** | 日志记录器，负责"说什么" | 你自己（说话的人） |
| **Handler** | 日志处理器，负责"输出到哪里" | 喇叭 / 笔 / 传真机 |
| **Formatter** | 日志格式器，负责"长什么样" | 消息的排版模板 |
| **Level** | 日志级别，负责"过滤什么" | 消息的重要程度 |

#### 日志级别（从低到高）

```
DEBUG    (10) → 调试细节，开发时用
INFO     (20) → 正常运行信息，如 "训练开始"
WARNING  (30) → 警告，程序还能跑但有隐患
ERROR    (40) → 出错了，但程序没崩
CRITICAL (50) → 严重错误，程序要崩了
```

设置 `logger.setLevel(logging.INFO)` 后，只有 >= INFO 级别的消息才会输出，DEBUG 会被过滤掉。

#### 最简单的用法

```python
import logging

# 方法 1：直接用 logging 模块（全局根 logger）
logging.basicConfig(level=logging.INFO)
logging.info("训练开始")   # 输出: INFO:root:训练开始

# 方法 2：创建命名 logger（推荐，大型项目用）
logger = logging.getLogger("my_project")
logger.setLevel(logging.INFO)
logger.info("训练开始")    # 此时还不会输出！因为没有 handler
```

**关键陷阱**：`getLogger()` 创建的 logger 默认没有 handler，不会输出任何东西。必须手动添加 handler。

#### Handler 和 Formatter

```python
import logging

logger = logging.getLogger("my_project")
logger.setLevel(logging.INFO)

# 创建一个 handler：输出到控制台（StreamHandler）
handler = logging.StreamHandler()

# 创建一个 formatter：定义输出格式
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",  # 格式模板
    datefmt="%Y-%m-%d %H:%M:%S",                  # 时间格式
)
handler.setFormatter(formatter)   # 把 formatter 绑到 handler
logger.addHandler(handler)        # 把 handler 绑到 logger

logger.info("训练开始")
# 输出: [2026-03-11 10:30:00] [INFO] 训练开始
```

常用的格式占位符：
- `%(asctime)s` — 时间戳
- `%(levelname)s` — 级别名（INFO / WARNING 等）
- `%(name)s` — logger 的名字
- `%(message)s` — 你传入的消息

常用的 Handler 类型：
- `StreamHandler()` — 输出到控制台（stderr）
- `FileHandler("train.log")` — 输出到文件

#### `getLogger()` 的单例特性

```python
logger_a = logging.getLogger("rl_trainer")
logger_b = logging.getLogger("rl_trainer")
print(logger_a is logger_b)  # True！同名的 logger 是同一个对象
```

这就是为什么 `get_logger()` 要检查 `if not logger.handlers`——防止多次调用时重复添加 handler，导致同一条消息打印多次。

### 逐行解析 `src-2/utils/logging.py`

```python
# === get_logger 函数 ===

def get_logger(name: str = "rl_trainer") -> logging.Logger:
    logger = logging.getLogger(name)      # 获取（或创建）一个命名 logger
    if not logger.handlers:               # 防止重复添加 handler
        handler = logging.StreamHandler() # 输出到控制台
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)     # 只输出 INFO 及以上级别
    return logger
```

在项目中任何地方调用 `get_logger()`，拿到的都是**同一个配置好的 logger**。

```python
# === is_main_process 函数 ===

def is_main_process() -> bool:
    if not dist.is_initialized():   # 没用分布式 → 当前就是唯一进程
        return True
    return dist.get_rank() == 0     # 分布式下只有 rank 0 是主进程
```

多 GPU 训练时（比如 4 张卡），每张卡上都跑一个进程。如果不过滤，同一条日志会打印 4 遍。这个函数让只有 rank 0（主进程）执行日志操作。

```python
# === log_stats 函数 ===

def log_stats(
    logger: logging.Logger,
    step: int,
    stats: Dict[str, float],           # 如 {"loss": 0.5, "reward_mean": 1.2}
    wandb_run: Optional[object] = None, # 可选的 wandb 实例
):
    if not is_main_process():           # 非主进程直接跳过
        return

    parts = [f"step={step}"]
    for k, v in stats.items():
        parts.append(f"{k}={v:.4f}")    # 每个指标保留 4 位小数
    logger.info(" | ".join(parts))      # 拼接成一行日志

    if wandb_run is not None:
        wandb_run.log(stats, step=step) # 同时上报到 wandb 面板
```

输出效果：`[2026-03-11 10:30:00] [INFO] step=42 | loss=0.3521 | reward_mean=1.2345`

### 为什么不用 print？

| | `print()` | `logging` |
|--|-----------|-----------|
| 级别过滤 | 不支持 | DEBUG/INFO/WARNING 等 |
| 时间戳 | 需手动加 | Formatter 自动加 |
| 输出目标 | 只有 stdout | 控制台、文件、网络都行 |
| 全局控制 | 无法关闭 | 改一下 level 就能静音 |
| 分布式 | 每个进程都打印 | 配合 `is_main_process` 过滤 |

---

## FSDP 分布式模块详解

### 为什么需要 FSDP？

7B 参数模型训练时的显存开销：

| 占用项 | 大小 (fp32) |
|--------|------------|
| 参数 | 28 GB |
| 梯度 | 28 GB |
| 优化器状态 (Adam m+v) | 56 GB |
| **合计** | > 112 GB |

一张 A100 (80GB) 放不下。FSDP 的核心思想：把参数/梯度/优化器状态切成 N 份，每张卡只存 1/N，用的时候再 all-gather 拼回来。

### 四个函数解析

#### 1. `init_distributed()` — 多卡建立通信
- `torchrun --nproc_per_node=4` 启动 4 个独立进程
- `dist.init_process_group("nccl")` 让它们互相认识（NCCL 是 NVIDIA GPU 通信库）
- `LOCAL_RANK` 环境变量由 torchrun 自动设置（0, 1, 2, 3）
- `torch.cuda.set_device(local_rank)` 绑定每个进程到对应的 GPU

#### 2. `cleanup_distributed()` — 释放通信资源

#### 3. `get_transformer_layer_cls()` — 自动探测 Transformer 重复层
- FSDP 需要知道以什么粒度切分模型
- 不同 HF 模型命名不同：Llama 用 `.model.layers`，GPT-2 用 `.transformer.h`
- 按 Transformer 层切分 vs 整体切分：前者内存峰值更低，计算通信可重叠

#### 4. `wrap_model_fsdp()` — 四个关键配置
| 配置 | 作用 |
|------|------|
| `ShardingStrategy` | FULL_SHARD(全切) / SHARD_GRAD_OP(只切梯度+优化器) / NO_SHARD(=DDP) |
| `MixedPrecision` | bf16/fp16 混合精度，省显存+加速计算 |
| `auto_wrap_policy` | 指定按哪种子模块层级做 FSDP 包装 |
| `CPUOffload` | 显存不够时把参数卸载到 CPU（会变慢） |

`use_orig_params=True` 保留原始参数名，兼容 HF 的 checkpoint 保存/加载。

### `transformer_auto_wrap_policy` 详解

核心逻辑极其简单——就是一个类型判断：

```python
def transformer_auto_wrap_policy(module, recurse, nonwrapped_numel, transformer_layer_cls):
    if recurse:
        return True                               # 继续递归遍历子模块
    return type(module) in transformer_layer_cls   # 命中目标类 → 包装！
```

FSDP 初始化时递归遍历所有子模块，对每个调用此函数：
- embed_tokens → 不命中 → 不独立包装（归属外层 FSDP）
- layers[0] (LlamaDecoderLayer) → 命中 → 包装成独立 FSDP 单元
- layers[1] → 命中 → 包装
- ...
- norm → 不命中 → 归属外层

代码中用 `functools.partial` 预绑定 `transformer_layer_cls` 参数，因为 FSDP 要求 policy 签名是 `(module, recurse, numel) -> bool`。

### 张量并行 (Tensor Parallelism) 新增内容

#### FSDP vs TP 的区别

| | FSDP (数据并行) | TP (张量并行) |
|--|-----------------|--------------|
| 切什么 | 每张卡存参数的 1/N 分片 | 每张卡存权重矩阵的 1/N 列或行 |
| 何时通信 | forward 前 all-gather，backward 后 reduce-scatter | forward/backward 中 all-reduce |
| 数据 | 每张卡处理不同的 batch | 每张卡处理相同的 batch（同一个输入） |

#### TP 的核心操作

对于一个 Linear 层 Y = XW：
- **ColwiseParallel**：按列切 W → 每卡算一部分输出 → 结果拼接
- **RowwiseParallel**：按行切 W → 输入也要切 → 每卡算部分结果 → all-reduce 求和

Transformer 中的 TP 分配规则（以 Llama 为例）：
- Q/K/V projection → ColwiseParallel（天然按 head 切分）
- O projection → RowwiseParallel（汇聚多 head 结果）
- MLP gate/up → ColwiseParallel
- MLP down → RowwiseParallel

#### 2D 并行 = TP + FSDP

8 张卡，tp_size=2 时的 DeviceMesh：

```
            TP 维度
           ┌──────┐
    GPU 0 ─┤      ├─ GPU 1     DP group 0
    GPU 2 ─┤      ├─ GPU 3     DP group 1
    GPU 4 ─┤      ├─ GPU 5     DP group 2
    GPU 6 ─┤      ├─ GPU 7     DP group 3
           └──────┘
    ← DP 维度 →
```

- TP 组 (横向)：GPU 0&1 共同处理一个 batch，各持有一半的权重列/行
- DP 组 (纵向)：GPU 0,2,4,6 处理不同 batch，FSDP 分片参数

### TP Plan 的切分规则（以 Llama Attention 为例）

```
输入 X [batch, seq, hidden=4096]
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    │
  Q_proj K_proj V_proj    ← ColwiseParallel：按列切
  [4096, [4096, [4096,       tp_size=2 时，每卡只存 [4096, 2048]
   4096]  4096]  4096]
    │    │    │
    ▼    ▼    ▼
  Attention 计算（每卡算一半 head）→ 中间无需通信
    │
    ▼
  O_proj                  ← RowwiseParallel：按行切
  [4096, 4096]               tp_size=2 时，每卡只存 [2048, 4096]
    │                        结果做 all-reduce 求和 → 唯一的通信点
    ▼
  输出 [batch, seq, 4096]
```

Col→Row 配对使得中间不需要额外通信，只在 RowwiseParallel 结束时做一次 all-reduce。

### `wrap_model_fsdp` 中 `process_group` 的关键作用

当 `device_mesh` 不为 None 时：
```python
fsdp_kwargs["process_group"] = device_mesh["dp"].get_group()
```
这确保 FSDP 只在 DP 维度的 GPU 之间做分片。
如果不加这行，FSDP 会在全部 GPU 之间分片，破坏 TP 的权重切分逻辑。

### 全局流程（8 卡 tp_size=2）

```
torchrun --nproc_per_node=8 train.py
  → init_distributed(tp_size=2)
    → init_process_group: 8 进程加入 NCCL 通信组
    → init_device_mesh: 排列成 4×2 矩阵 (dp=4, tp=2)
  → 每个进程创建完整模型
  → apply_tensor_parallel(model, mesh)
    → 对每个 DecoderLayer: parallelize_module 按 plan 切分权重
    → q_proj [4096,4096] → 每卡只存 [4096,2048]
  → wrap_model_fsdp(model, cfg, mesh)
    → FSDP 只在 DP 维度分片（4路），不碰 TP 伙伴
    → q_proj [4096,2048] 再被 4 路 DP 分片 → 每卡实际存 [4096,512]
  → 训练循环
  → cleanup_distributed()
```

---

## 分布式编程常见误解 Q&A

### Q1: 分布式训练是只有 rank 0 在执行代码吗？

**不是。** 所有进程都在执行同一份代码（SPMD 模式：Single Program, Multiple Data）。

```
torchrun --nproc_per_node=4 train.py
```

启动 4 个**完全独立的 Python 进程**，每个从头到尾执行 `train.py` 的全部代码：

| 进程 | `LOCAL_RANK` | 绑定 GPU | 执行的代码 |
|------|-------------|---------|-----------|
| 进程 0 | 0 | GPU 0 | train.py 全部 |
| 进程 1 | 1 | GPU 1 | train.py 全部 |
| 进程 2 | 2 | GPU 2 | train.py 全部 |
| 进程 3 | 3 | GPU 3 | train.py 全部 |

`torch.cuda.set_device(local_rank)` 让每个进程绑定**自己那张卡**。不调这行的话，4 个进程全部默认用 GPU 0。

**只有极少数操作是 rank 0 独享的**：打印日志、保存 checkpoint。模型创建、FSDP 包装、训练循环，每个进程都必须执行。

### Q2: 每个进程都要执行 `init_device_mesh` 吗？

**是的。** 但每个进程得到的是**同一个逻辑网格的不同视角**：

```python
device_mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
```

所有进程都执行这行，但 `DeviceMesh` 内部知道"我是网格中的哪个位置"：

```
         tp=0   tp=1
dp=0  [  GPU0,  GPU1 ]
dp=1  [  GPU2,  GPU3 ]
dp=2  [  GPU4,  GPU5 ]
dp=3  [  GPU6,  GPU7 ]
```

- GPU0 的进程执行 `device_mesh["tp"]` → 拿到 TP 组 `[GPU0, GPU1]`
- GPU2 的进程执行 `device_mesh["tp"]` → 拿到 TP 组 `[GPU2, GPU3]`

**同一行代码，每个进程拿到的子组不同**——`DeviceMesh` 根据当前 rank 自动返回所属的子组。

这就是分布式编程的核心范式：**SPMD（Single Program, Multiple Data）——同一份程序，不同的数据/视角。**

### Q3: `device_mesh` 不用单独拆出 `dp_mesh` 和 `tp_mesh` 吗？

不需要。`DeviceMesh` 是一个**可以按维度索引的容器**，需要时直接切片：

```python
device_mesh["tp"]              # 取 TP 子网格，用于 apply_tensor_parallel
device_mesh["dp"].get_group()  # 取 DP 通信组，用于 FSDP 的 process_group
```

不需要提前创建两个变量分别传递，一个 `device_mesh` 对象就够了。

### Q4: TP plan 检测 `_detect_tp_plan(layers[0])` 中的 `layers[0]` 是什么？

`layers[0]` 是**第一个 DecoderLayer**（不是第一个 Linear），它内部包含 q/k/v/o_proj + gate/up/down_proj 共 7 个 Linear。

`_detect_tp_plan` 返回的是**一整个字典**，为每个 Linear 分别指定了不同的并行方式：

```python
{
    "self_attn.q_proj": ColwiseParallel(),   # Q 按列切
    "self_attn.k_proj": ColwiseParallel(),   # K 按列切
    "self_attn.v_proj": ColwiseParallel(),   # V 按列切
    "self_attn.o_proj": RowwiseParallel(),   # O 按行切 ← 和上面不一样
    "mlp.gate_proj":    ColwiseParallel(),
    "mlp.up_proj":      ColwiseParallel(),
    "mlp.down_proj":    RowwiseParallel(),   # ← 也不一样
}
```

只用 `layers[0]` 检测一次就够了，因为 `layers[0]` 到 `layers[31]` 的类型和结构完全相同，同一个 plan 对所有层都适用。

### Q5: 在循环中检测 vs 循环前检测有什么区别？

**循环中检测（不推荐）**：

```python
for layer in layers:
    tp_plan = _assign_tp_plan(layer)   # 每层检测一次
    if tp_plan is None:
        raise ValueError(...)          # 第 15 层才报错，前 14 层已被修改
    parallelize_module(layer, tp_mesh, tp_plan)
```

问题：（1）32 层做 32 次完全相同的检测，浪费；（2）如果中途报错，模型处于"一半切了一半没切"的不一致状态。

**循环前检测（推荐）**：

```python
tp_plan = _detect_tp_plan(layers[0])   # 只检测一次
if tp_plan is None:
    raise ValueError(...)              # 在修改任何东西之前就报错

for layer in layers:
    parallelize_module(layer, tp_mesh, tp_plan)
```

优点：（1）避免重复计算；（2）fail-fast——要么全部成功，要么一个都不做。

---

## RLVR 数学验证 Reward 详解

### 什么是 RLVR？

RLVR（Reinforcement Learning with Verifiable Rewards）的核心思想：**不用人类标注偏好，而是用可程序化验证的方式给奖励**。

传统 RLHF 流程：模型回答 → 人类打分 → 训练 reward model → 用 reward model 打分
RLVR 流程：模型回答 → **程序自动验证答案是否正确** → 直接给 0/1 奖励

数学题是天然的 RLVR 场景——答案有唯一正确解，可以用 sympy 做符号化验证。

### 实现架构

```
Prompt (含标准答案)          Response (模型输出)
"Solve: 2x+6=20              "Let me solve step by step.
 <answer>7</answer>"           2x = 14, so x = 7.
       │                       The answer is \\boxed{7}"
       ▼                              │
extract_ground_truth()         extract_answer()
       │                              │
       ▼                              ▼
   gold_str = "7"              pred_str = "7"
       │                              │
       └──────────┬───────────────────┘
                  ▼
          sympy_equal("7", "7")
                  │
          normalize_expr → sympy.sympify
          simplify(pred - gold) == 0 ?
                  │
                  ▼
          reward = 1.0 (correct)
          + format_reward bonus
```

### 三层答案提取

**从模型响应中提取预测答案** `extract_answer(response)`：

| 优先级 | 模式 | 示例 | 说明 |
|--------|------|------|------|
| 1 | `\boxed{...}` | `\boxed{7}` | LaTeX 标准，DeepSeek-R1 等模型常用 |
| 2 | "The answer is ..." | `The final answer is 7.` | 自然语言表述 |
| 3 | 最后一个数字 | `...so x = 7` | 兜底：取响应中最后出现的数字 |

**从 prompt 中提取标准答案** `extract_ground_truth(prompt)`：

| 格式 | 示例 | 说明 |
|------|------|------|
| `<answer>...</answer>` | `<answer>7</answer>` | 自定义 XML 标签 |
| `[ANSWER: ...]` | `[ANSWER: 3/4]` | 方括号格式 |
| `#### ...` | `#### 42` | GSM8K 数据集格式 |

### sympy 验证的三重比较

`sympy_equal(pred_str, gold_str)` 进行三重验证，确保各种等价形式都能被识别：

```
"3/6" vs "1/2"

第 1 重：sympy.simplify(pred - gold) == 0
  → simplify(1/2 - 1/2) = 0  ✓

第 2 重：pred_expr.equals(gold_expr)
  → 处理 simplify 搞不定的复杂表达式

第 3 重：数值比较 float(evalf())
  → |0.5 - 0.5| < 1e-6  ✓ 兜底
```

能正确处理的等价对：
- `3/6` vs `1/2` （分数化简）
- `\frac{3}{4}` vs `0.75` （LaTeX vs 小数）
- `x^2 + 2x + 1` vs `(x+1)^2` （多项式展开）
- `3*x^2 + 2` vs `3x² + 2` （符号表示差异）

### 奖励分数设计

| 情况 | 默认奖励 | 说明 |
|------|---------|------|
| 答案正确 | `+1.0` | 核心信号 |
| 答案错误 | `-0.5` | 负激励，比0更有效 |
| 无法解析 | `-0.1` | 轻微惩罚，鼓励输出规范格式 |
| 格式奖励 | `+0.1` | 有推理步骤 + `\boxed{}` 获得额外加分 |

格式奖励是 RLVR 的重要补充——不仅要答对，还要**展示推理过程**，避免模型学会直接猜答案。

### 示例配置（configs/grpo_math_verify.yaml）

```yaml
reward:
  type: "math_verify"
  correct_reward: 1.0
  incorrect_reward: -0.5
  format_reward: 0.1
  unparseable_reward: -0.1

data:
  source: "list"
  prompts:
    - "What is 2 + 3 * 4? <answer>14</answer>"
    - "Solve for x: 2x + 6 = 20. <answer>7</answer>"
    - "What is 1/3 + 1/6? <answer>1/2</answer>"
```

---

## 当前 `src/` 中已有代码的问题

| 文件 | 问题 |
|------|------|
| `trainer/rl_trainer.py` | `self.step` 未初始化、参数名不匹配 (`max_train_steps` vs `max_steps`)、缺少 `wandb_run` 参数、`os.mkdir` 应为 `os.makedirs`、dataloader 未循环复用 |
| `algorithms/ppo.py` | 只有空的 `__init__`，需要完整实现 |
| `algorithms/grpo.py` | 只有空的 `__init__`，需要完整实现 |
| 缺失模块 | `models/`、`rewards/`、`data/`、`rollout/`、`distributed/`、`utils/` 全部缺失 |
