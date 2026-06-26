# TCOD 项目复现计划

> 基于 8×RTX 3090 (24GB) 的完整全流程复现方案  
> 项目：TCOD (Temporal Curriculum for On-Policy Distillation)  
> 论文：arXiv:2604.24005, COLM 2026

---

## 一、资源评估与可行性分析

### 1.1 原始硬件要求

| 配置项 | 论文主实验 | 论文附录 |
|--------|-----------|---------|
| GPU | 8× NVIDIA H20 (96GB) | 8× NVIDIA A100 (80GB) |
| GPU 分配 | 4× actor + 2× trainer + 2× teacher | 同上 |
| 训练时间 | ~18h (ALFWorld) | ~12-24h |

### 1.2 我们的硬件

| 配置项 | 可用资源 |
|--------|---------|
| GPU | 8× NVIDIA RTX 3090 (24GB) |
| 显存差距 | 24GB vs 96GB = **4倍差距** |
| CPU 内存 | 需要 160GB+ 用于 offload |

### 1.3 可行性矩阵

| 实验 | Student | Teacher | 3090 可行性 | 关键瓶颈 |
|------|---------|---------|------------|---------|
| **ALFWorld TCOD-B2F** | Qwen2.5-1.5B | Qwen2.5-7B-RL | ✅ **可行** | 已验证类似配置 |
| **ALFWorld TCOD-F2B** | Qwen2.5-1.5B | Qwen2.5-7B-RL | ✅ **可行** | 同上 |
| **ALFWorld OPD** | Qwen2.5-3B | Qwen2.5-7B-RL | ✅ **可行** | 需要 offload |
| **ScienceWorld TCOD-B2F** | Qwen2.5-1.5B | Qwen2.5-7B-RL | ✅ **可行** | 长 trajectory |
| **ScienceWorld OPD** | Qwen2.5-3B | Qwen2.5-7B-RL | ⚠️ **紧张** | 需要 aggressive 优化 |
| **WebShop OPD** | Qwen2.5-3B | Teacher | ⚠️ **紧张** | WebShop 需要 ~1TB RAM |
| **WebShop TCOD-B2F** | Qwen2.5-7B | Teacher | ❌ **困难** | 7B student + teacher |
| **Cross-benchmark 4B** | Qwen3-4B | Qwen3-30B-A3B | ❌ **困难** | 30B teacher 加载 |

### 1.4 推荐复现策略

**核心原则**：先跑通最小可行实验，再逐步扩展

```
Phase 1: ALFWorld 1.5B (TCOD-B2F/F2B)  ← 优先级最高，已验证可行
Phase 2: ALFWorld 3B (OPD baseline)     ← 对比实验
Phase 3: ScienceWorld 1.5B              ← 扩展到第二个环境
Phase 4: WebShop (如果资源允许)          ← 挑战性实验
```

---

## 二、详细复现计划

### Phase 0: 环境搭建（预计 2-4 小时）

#### 0.1 创建虚拟环境

```bash
cd /home/chenyizhou/TCOD
conda create -n tcod python=3.10 -y
conda activate tcod
```

#### 0.2 安装项目

```bash
# 安装 Trinity-RFT
pip install -e ".[dev]"

# 安装 flash-attn（如果 CUDA 版本兼容）
pip install flash-attn==2.8.1 --no-build-isolation

# 如果 flash-attn 安装失败，使用 sdpa 替代
# 需要修改配置：attn_implementation: sdpa
```

#### 0.3 安装环境依赖

```bash
# ALFWorld
pip install alfworld
alfworld-download

# ScienceWorld（可选）
git clone https://github.com/allenai/ScienceWorld.git
cd ScienceWorld && pip install . && cd ..
```

#### 0.4 启动 Ray

```bash
ray start --head
```

#### 0.5 验证环境

```bash
# 检查 GPU
nvidia-smi

# 检查 Python/CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# 检查 Ray
ray status
```

---

### Phase 1: ALFWorld 1.5B TCOD 复现（预计 12-18 小时）

**目标**：复现 ALFWorld 上 TCOD-B2F 和 TCOD-F2B 的核心结果

#### 1.1 数据准备

```bash
cd TCOD_examples/alfworld
python get_alfworld_data.py
```

**预期输出**：
- `TCOD_examples/alfworld/alfworld_data/train.jsonl`
- `TCOD_examples/alfworld/alfworld_data/test.jsonl`

#### 1.2 下载模型

```bash
# Student 模型
# Qwen2.5-1.5B-Instruct（约 3GB）

# Teacher 模型
# 需要 GRPO-trained Qwen2.5-7B on ALFWorld
# 如果没有，可以用通用 Qwen2.5-7B-Instruct 作为 teacher（效果会差一些）
```

#### 1.3 3090 适配配置

**原始配置 vs 3090 配置**：

| 参数 | 原始值 | 3090 值 | 原因 |
|------|--------|---------|------|
| `gpu_memory_utilization` | 0.7 | 0.35 | 留出 FSDP + teacher 空间 |
| `trainer.max_token_len_per_gpu` | 16384 | 4096 | 减少 trainer 显存 |
| `trainer.ulysses_sequence_parallel_size` | 2 | 1 | 减少并行度 |
| `buffer.train_batch_size` | 64 | 16 | 减少 batch 显存 |
| `model.max_response_tokens` | 512 | 256 | 减少 KV cache |
| FSDP param_offload | False | True | 卸载到 CPU |
| FSDP optimizer_offload | False | True | 卸载到 CPU |
| `explorer.rollout_model.tensor_parallel_size` | 2 | 1 | 1.5B 模型单卡够 |
| `explorer.rollout_model.engine_num` | 2 | 2 | 保持 2 个 rollout engine |
| `explorer.auxiliary_models[0].tensor_parallel_size` | 2 | 2 | 7B teacher 需要 TP=2 |

#### 1.4 创建 3090 配置文件

**文件**：`TCOD_examples/alfworld/tcod_b2f_3090.yaml`

```yaml
project: "ALFWORLD_TCOD_3090"
name: "alfworld_tcod_b2f_3090"
checkpoint_root_dir: ./checkpoints
continue_from_checkpoint: false

algorithm:
  sample_strategy: staleness_control
  sample_strategy_args:
    max_staleness: 2
  algorithm_type: on_policy_distill
  advantage_fn: multi_turn_opd
  repeat_times: 1
  advantage_fn_args:
    kl_coef: 1.0
  optimizer:
    lr: 1e-6

model:
  model_path: Qwen/Qwen2.5-1.5B-Instruct
  max_prompt_tokens: 4096
  max_response_tokens: 256

cluster:
  node_num: 1
  gpu_per_node: 8

buffer:
  total_steps: 100  # 减少步数用于验证
  batch_size: 8
  train_batch_size: 16
  explorer_input:
    taskset:
      name: alfworld
      storage_type: file
      path: /home/chenyizhou/TCOD/TCOD_examples/alfworld/alfworld_data/train.jsonl
      split: train
      format:
        prompt_key: 'game_file'
      rollout_args:
        temperature: 1.0
        logprobs: 0
      workflow_args:
        temperature: 1.0
        max_env_steps: 30
        total_steps: 100
        checkpoint_strategy: linear
        checkpoint_steps: 5
    eval_tasksets:
      - name: alfworld_eval_seen
        storage_type: file
        path: /home/chenyizhou/TCOD/TCOD_examples/alfworld/alfworld_data/test.jsonl
        split: test
        total_steps: 5
        task_selector:
          selector_type: random
          seed: 42
        format:
          prompt_key: 'game_file'
        rollout_args:
          temperature: 0.4
          logprobs: 0
    default_workflow_type: 'TCOD_b2f_alfworld_workflow'
  trainer_input:
    experience_buffer:
      name: alfworld_tcod_b2f_buffer
      storage_type: queue
      path: 'sqlite:///alfworld_tcod_b2f_buffer.db'

explorer:
  eval_interval: 10
  runner_per_model: 4
  max_timeout: 3600
  rollout_model:
    engine_num: 2
    tensor_parallel_size: 1  # 1.5B 模型单卡够
    enable_prefix_caching: false
    enforce_eager: true
    dtype: bfloat16
    seed: 42
    gpu_memory_utilization: 0.35
    enable_chunked_prefill: true
  auxiliary_models:
    - model_path: /path/to/teacher  # 需要替换为实际路径
      engine_num: 1
      tensor_parallel_size: 2  # 7B teacher 需要 TP=2
      enable_prefix_caching: false
      enforce_eager: true
      dtype: bfloat16
      seed: 42
      max_model_len: 4096
      max_prompt_tokens: 4096
      max_response_tokens: 256
  env_vars:
    TMPDIR: /dev/shm/tmp
    RAY_TMPDIR: /dev/shm/ray_tmp

synchronizer:
  sync_method: 'nccl'
  sync_style: 'dynamic_by_explorer'
  sync_interval: 1
  sync_timeout: 3600

trainer:
  total_steps: 100
  save_interval: 50
  grad_clip: 1.0
  use_dynamic_bsz: true
  max_token_len_per_gpu: 4096
  ulysses_sequence_parallel_size: 1

monitor:
  monitor_type: none  # 3090 可能没有 wandb
```

#### 1.5 运行 TCOD-B2F

```bash
cd /home/chenyizhou/TCOD
trinity run --config TCOD_examples/alfworld/tcod_b2f_3090.yaml
```

**监控指标**：
- `actor/pg_loss`：应该非零，说明蒸馏信号存在
- `critic/score/mean`：负 KL，应该向 0 靠近
- `topk/overlap_ratio`：student/teacher token 重叠率
- `env_done`：任务完成率
- `kl_divergence`：KL 散度

#### 1.6 运行 TCOD-F2B

修改配置：
```yaml
name: "alfworld_tcod_f2b_3090"
buffer.explorer_input.default_workflow_type: 'TCOD_f2b_alfworld_workflow'
```

```bash
trinity run --config TCOD_examples/alfworld/tcod_f2b_3090.yaml
```

#### 1.7 运行 OPD Baseline

修改配置：
```yaml
name: "alfworld_opd_3090"
model.model_path: Qwen/Qwen2.5-3B-Instruct  # 使用 3B student
buffer.explorer_input.default_workflow_type: 'OPD_alfworld_workflow'
```

```bash
trinity run --config TCOD_examples/alfworld/opd_3090.yaml
```

#### 1.8 评估

```bash
# 评估会在训练过程中自动运行（eval_interval: 10）
# 也可以手动评估 checkpoint
```

**预期结果**：
- TCOD-B2F/F2B 应该比 OPD 有更高的成功率
- KL 曲线应该更稳定
- 训练时间应该更短

---

### Phase 2: ALFWorld 3B 扩展实验（预计 18-24 小时）

**目标**：验证 TCOD 在更大 student 模型上的效果

#### 2.1 修改配置

```yaml
model:
  model_path: Qwen/Qwen2.5-3B-Instruct
  max_prompt_tokens: 4096
  max_response_tokens: 256

explorer:
  rollout_model:
    tensor_parallel_size: 2  # 3B 模型需要 TP=2
    gpu_memory_utilization: 0.3
```

#### 2.2 运行实验

```bash
# TCOD-B2F with 3B student
trinity run --config TCOD_examples/alfworld/tcod_b2f_3090_3b.yaml

# TCOD-F2B with 3B student
trinity run --config TCOD_examples/alfworld/tcod_f2b_3090_3b.yaml

# OPD with 3B student
trinity run --config TCOD_examples/alfworld/opd_3090_3b.yaml
```

---

### Phase 3: ScienceWorld 扩展（预计 18-24 小时）

**目标**：验证 TCOD 在不同环境上的泛化能力

#### 3.1 数据准备

```bash
cd TCOD_examples/scienceworld

# 修改 get_sciworld_data.py 中的 jar_path
python get_sciworld_data.py
```

#### 3.2 运行实验

```bash
# 使用与 ALFWorld 类似的 3090 配置
trinity run --config TCOD_examples/scienceworld/tcod_b2f_3090.yaml
```

---

### Phase 4: WebShop 挑战（可选，需要大量 RAM）

**注意**：WebShop 需要 ~1TB 系统 RAM，如果机器 RAM 不足可以跳过

#### 4.1 环境准备

```bash
git clone https://github.com/princeton-nlp/webshop.git webshop
cd webshop
./setup.sh -d small  # 使用小数据集
```

#### 4.2 运行实验

```bash
trinity run --config TCOD_examples/webshop/opd_3090.yaml
```

---

## 三、关键问题与解决方案

### 3.1 OOM 问题

**症状**：CUDA out of memory

**解决方案**（按优先级）：
1. 降低 `gpu_memory_utilization`（0.7 → 0.35）
2. 降低 `max_token_len_per_gpu`（16384 → 4096）
3. 降低 `train_batch_size`（64 → 16）
4. 启用 FSDP offload（param_offload + optimizer_offload）
5. 降低 `max_response_tokens`（512 → 256）
6. 减少 `runner_per_model`（8 → 4）

### 3.2 Teacher 模型问题

**问题**：没有 GRPO-trained Qwen2.5-7B-RL teacher

**解决方案**：
1. 使用通用 Qwen2.5-7B-Instruct 作为 teacher（效果会差）
2. 先用 GRPO 训练一个 ALFWorld teacher（需要额外时间）
3. 使用其他开源 teacher 模型

### 3.3 WebShop RAM 问题

**问题**：WebShop 需要 ~1TB 系统 RAM

**解决方案**：
1. 跳过 WebShop，专注于 ALFWorld 和 ScienceWorld
2. 使用 WebShop 的 small 数据集
3. 如果有足够 RAM 的机器，可以远程运行

### 3.4 flash-attn 兼容性

**问题**：CUDA 版本与 flash-attn 不兼容

**解决方案**：
```bash
# 方案 1：安装 flash-attn
pip install flash-attn==2.8.1 --no-build-isolation

# 方案 2：使用 sdpa 替代
# 在配置中设置 attn_implementation: sdpa
```

---

## 四、时间估算

| Phase | 实验 | 预计时间 | 依赖 |
|-------|------|---------|------|
| 0 | 环境搭建 | 2-4h | 无 |
| 1.1 | ALFWorld 1.5B TCOD-B2F | 12-18h | Phase 0 |
| 1.2 | ALFWorld 1.5B TCOD-F2B | 12-18h | Phase 0 |
| 1.3 | ALFWorld 3B OPD | 18-24h | Phase 0 |
| 2 | ALFWorld 3B TCOD | 18-24h | Phase 1 |
| 3 | ScienceWorld 1.5B | 18-24h | Phase 1 |
| 4 | WebShop (可选) | 24-36h | Phase 1 |

**总计**：约 80-120 小时（如果并行运行可缩短）

---

## 五、验收标准

### 5.1 最低标准（必须达到）

- [ ] 环境搭建成功，`trinity run` 可以启动
- [ ] ALFWorld 1.5B TCOD-B2F 训练完成（100 steps）
- [ ] 训练日志中 `actor/pg_loss` 非零
- [ ] 评估结果中 TCOD 优于 OPD（或至少相当）

### 5.2 完整标准（理想达到）

- [ ] ALFWorld 1.5B TCOD-B2F/F2B 和 OPD 全部完成
- [ ] ALFWorld 3B 实验完成
- [ ] ScienceWorld 实验完成
- [ ] 成功率对比表完整
- [ ] KL 曲线对比图完整

### 5.3 学习标准（核心收获）

- [ ] 理解 OPD 的核心原理和实现
- [ ] 理解 TCOD 的 temporal curriculum 设计
- [ ] 理解 Trinity-RFT 的 Explorer-Trainer 架构
- [ ] 理解多轮 Agent 训练的挑战和解决方案
- [ ] 能够独立修改配置和调试问题

---

## 六、学习路径

### 6.1 第一周：环境与基础

**Day 1-2**：环境搭建
- 安装 Trinity-RFT
- 安装 ALFWorld
- 验证环境

**Day 3-4**：理解架构
- 阅读 Trinity-RFT 文档
- 理解 Explorer-Trainer 架构
- 理解 Workflow 注册机制

**Day 5-7**：运行第一个实验
- 配置 3090 适配
- 运行 ALFWorld TCOD-B2F
- 监控训练指标

### 6.2 第二周：深入理解

**Day 8-10**：分析训练结果
- 分析 KL 曲线
- 分析成功率
- 对比 TCOD vs OPD

**Day 11-14**：扩展实验
- 运行 TCOD-F2B
- 运行 OPD baseline
- 准备对比表

### 6.3 第三周：扩展与总结

**Day 15-17**：ScienceWorld
- 数据准备
- 运行实验
- 分析结果

**Day 18-21**：总结与文档
- 整理实验结果
- 编写学习笔记
- 准备面试材料

---

## 七、附录：快速参考

### 7.1 常用命令

```bash
# 启动 Ray
ray start --head

# 运行实验
trinity run --config <config.yaml>

# 监控 GPU
watch -n 1 nvidia-smi

# 查看 Ray 状态
ray status

# 停止 Ray
ray stop
```

### 7.2 关键配置参数

```yaml
# 显存相关
gpu_memory_utilization: 0.35  # vLLM KV cache 预算
max_token_len_per_gpu: 4096   # Trainer 每 GPU token 预算
train_batch_size: 16          # 训练 batch size

# 模型相关
model_path: Qwen/Qwen2.5-1.5B-Instruct
tensor_parallel_size: 1       # 模型并行度

# TCOD 相关
checkpoint_strategy: linear
checkpoint_steps: 5
total_steps: 100
```

### 7.3 监控指标

| 指标 | 含义 | 期望值 |
|------|------|--------|
| `actor/pg_loss` | Policy gradient loss | 非零 |
| `critic/score/mean` | 负 KL 散度 | 向 0 靠近 |
| `topk/overlap_ratio` | Top-K token 重叠率 | 上升 |
| `env_done` | 任务完成率 | 上升 |
| `kl_divergence` | KL 散度 | 稳定或下降 |

---

*文档版本：v1.0*  
*最后更新：2026-06-26*  
*基于 8×RTX 3090 (24GB) 硬件配置*
