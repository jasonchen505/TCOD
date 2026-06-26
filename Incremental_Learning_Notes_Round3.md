# TCOD 项目增量学习笔记

> 第三轮学习：基于复现计划制定过程中的新发现  
> 对比前两轮（面试准备文档 + 五类能力文档）的增量知识

---

## 一、框架架构层面的新理解

### 1.1 Trinity-RFT 的 GPU 分配机制

**前两轮理解**：
- 知道 Explorer-Trainer 架构
- 知道用 Ray 做分布式

**本轮新增**：
```
Total GPUs = cluster.node_num * cluster.gpu_per_node
Explorer GPUs = engine_num * tensor_parallel_size (student rollout)
Auxiliary GPUs = sum(engine_num * tensor_parallel_size) for each auxiliary model (teacher)
Trainer GPUs = Total - Explorer - Auxiliary
```

**具体到 8 卡配置**：
- 原始 H20 配置：4× actor + 2× trainer + 2× teacher
- 3090 适配：2× actor (TP=1) + 2× teacher (TP=2) + 4× trainer

**关键洞察**：
> Trainer GPU 数量可以动态调整，更多 Trainer GPU 意味着更大的 `max_token_len_per_gpu` 预算，但会减少 rollout 并行度。

### 1.2 `max_token_len_per_gpu` 是真正的显存瓶颈

**前两轮理解**：
- 知道 `gpu_memory_utilization` 控制 vLLM KV cache
- 知道需要降低 batch size

**本轮新增**：
```python
# Trinity 自动计算默认值
trainer.max_token_len_per_gpu = ceil(2 * model.max_model_len / trainer.ulysses_sequence_parallel_size)
```

**关键洞察**：
> `max_token_len_per_gpu` 才是 Trainer 显存的真正驱动因素，不是 `gpu_memory_utilization`。原配置 16384 在 3090 上会 OOM，需要降到 4096-8192。

### 1.3 Sequence Parallelism (SP) 的作用

**前两轮理解**：
- 不知道 SP 的存在

**本轮新增**：
- SP 是 Trinity 支持的序列并行技术
- SP=2 意味着序列被分成 2 份，分别在 2 个 GPU 上计算
- 可以显著降低单 GPU 显存压力

**GPU 配置表（A100 80GB）**：
```
Qwen3-1.7B, max_model_len=20480, 2 GPUs: vanilla（不需要特殊配置）
Qwen3-4B, max_model_len=20480, 2 GPUs: SP=2（需要序列并行）
Qwen3-8B, max_model_len=20480, 2 GPUs: Env + Offload（需要环境变量 + CPU 卸载）
```

**关键洞察**：
> 3090 24GB 相当于 A100 80GB 的约 30% 显存，所以 A100 上需要 Offload 的配置在 3090 上肯定需要更激进的优化。

---

## 二、模型与数据层面的新理解

### 2.1 Teacher 模型的获取问题

**前两轮理解**：
- 知道需要 teacher 模型
- 假设 teacher 模型可以直接下载

**本轮新增**：
- 论文中 ALFWorld 的 teacher 是 **GRPO-trained Qwen2.5-7B**
- 这是一个 **领域特化的 teacher**，不是通用模型
- 通用 Qwen2.5-7B-Instruct 作为 teacher 效果会差很多

**论文关键发现**：
> "Teacher quality strongly affects the upper bound of TCOD. A domain-adapted teacher is more useful than a larger general one."

**解决方案**：
1. 先用 GRPO 训练一个 ALFWorld teacher（需要额外时间）
2. 使用通用 teacher 作为 baseline，但效果会受限
3. 寻找开源的 GRPO-trained teacher checkpoint

### 2.2 ALFWorld 数据结构

**前两轮理解**：
- 知道 ALFWorld 是文本化家庭环境
- 知道需要下载数据

**本轮新增**：
```python
# get_alfworld_data.py 的数据结构
train_data = [
    {"game_file": game_file_path, "target": ""} 
    for game_file_path in selected_train_files
]
```

- 每个样本是一个 `game_file` 路径，指向 `.tw-pddl` 文件
- 数据量：train 有数千个任务，test 有数百个
- 可以通过 `train_size` 和 `test_size` 参数控制数据量

**关键洞察**：
> ALFWorld 的数据是动态加载的，训练时会根据 `game_file` 路径实时创建环境。这意味着数据量不大，但每个任务的交互可能很长（最多 30 步）。

### 2.3 WebShop 的资源瓶颈

**前两轮理解**：
- 知道 WebShop 是电商购物环境
- 知道需要 Java 17+

**本轮新增**：
- WebShop 需要 **~1TB 系统 RAM**（不是 GPU 显存）
- 这是因为 WebShop 的完整数据集很大
- 可以使用 `-d small` 参数下载小数据集

**关键洞察**：
> WebShop 的瓶颈是系统 RAM 而非 GPU 显存。如果机器 RAM 不足（<1TB），应该跳过 WebShop，专注于 ALFWorld 和 ScienceWorld。

---

## 三、配置与调优层面的新理解

### 3.1 配置参数的优先级

**前两轮理解**：
- 知道有很多配置参数
- 不知道调整的优先级

**本轮新增**：

**OOM 排查顺序**：
1. `gpu_memory_utilization`（0.7 → 0.35）：最快见效
2. `max_token_len_per_gpu`（16384 → 4096）：影响最大
3. `train_batch_size`（64 → 16）：减少 batch 显存
4. FSDP offload（param + optimizer）：用 CPU 换 GPU
5. `max_response_tokens`（512 → 256）：减少 KV cache
6. `runner_per_model`（8 → 4）：减少并行任务

**关键洞察**：
> 调优应该从影响最大、改动最小的参数开始。`max_token_len_per_gpu` 是 Trainer 显存的主控参数，应该优先调整。

### 3.2 GPU 分配的 trade-off

**前两轮理解**：
- 知道需要分配 GPU 给 actor、trainer、teacher
- 不知道具体怎么分配

**本轮新增**：

**1.5B student + 7B teacher 的最优分配**：
```
Student rollout: 2×3090 (engine_num=2, TP=1)  → 2 GPUs
Teacher rollout: 2×3090 (engine_num=1, TP=2)  → 2 GPUs
Trainer:         4×3090                        → 4 GPUs
```

**关键洞察**：
> 3090 配置下 Trainer 有 4 个 GPU（比原始 H20 的 2 个更多），这反而是一个优势，可以支持更大的 `max_token_len_per_gpu`。但 rollout 并行度降低了，会影响数据收集速度。

### 3.3 TCOD 特有参数

**前两轮理解**：
- 知道 `checkpoint_strategy: linear`
- 知道 `checkpoint_steps` 控制课程进度

**本轮新增**：

**线性 pacing 公式**：
```python
k = k_start + floor(n / η)  # n: 当前训练步, η: 增长率
```

**参数含义**：
- `checkpoint_steps`：每多少步减少一个 teacher checkpoint step
- `total_steps`：总训练步数，用于计算当前进度
- `k_start`：初始步数（通常为 1）

**关键洞察**：
> `checkpoint_steps` 和 `total_steps` 的比例决定了课程增长速度。如果 `checkpoint_steps=5, total_steps=250`，那么每 5 步减少一个 teacher step，总共可以减少 50 个 step。

---

## 四、工程实践层面的新理解

### 4.1 Ray 临时目录问题

**前两轮理解**：
- 知道需要启动 Ray
- 不知道临时目录的问题

**本轮新增**：
```bash
# 避免根分区空间不足
export RAY_TMPDIR=/mnt/sdb2/ray_tmp
export TMPDIR=/dev/shm/tmp
```

**关键洞察**：
> Ray 默认使用 `/tmp` 作为临时目录，如果根分区空间不足会导致训练失败。需要提前设置 `RAY_TMPDIR` 到大容量磁盘。

### 4.2 vLLM 与 FSDP 的内存共享

**前两轮理解**：
- 知道 vLLM 用于推理
- 知道 FSDP 用于训练
- 不知道它们如何共享内存

**本轮新增**：

**Hybrid Engine 设计**：
```
Rollout 阶段：vLLM 使用 GPU 内存生成序列
  → 需要模型权重 + KV cache
  
Training 阶段：FSDP 使用 GPU 内存训练
  → 需要模型权重 + 梯度 + 优化器 + 激活值
  
切换时：vLLM 释放 KV cache → FSDP 加载优化器状态
```

**关键洞察**：
> `gpu_memory_utilization` 控制的是 vLLM KV cache 预算，不是总内存。需要为 FSDP 留出足够空间，所以 3090 上要降到 0.35。

### 4.3 Staleness Control 的实现

**前两轮理解**：
- 知道 staleness control 用于异步训练
- 不知道具体实现

**本轮新增**：
```yaml
algorithm:
  sample_strategy: staleness_control
  sample_strategy_args:
    max_staleness: 2
```

**实现机制**：
- 每个 trajectory 打上策略版本号
- Trainer 检查版本差异
- 丢弃版本差异 > `max_staleness` 的 experience

**关键洞察**：
> `max_staleness=2` 是经验值，在样本效率和 on-policy 约束之间取得平衡。太小会导致丢弃太多数据，太大会导致 off-policy 问题。

---

## 五、评估与分析层面的新理解

### 5.1 TCOD 的评估指标

**前两轮理解**：
- 知道用成功率（SR）评估
- 知道 KL 散度是重要指标

**本轮新增**：

**完整指标体系**：
| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `env_done` | 任务完成率 | 上升 |
| `kl_divergence` | KL 散度 | 稳定或下降 |
| `env_rounds` | 平均交互步数 | 下降（更高效） |
| `student_env_rounds` | 学生交互步数 | 下降 |
| `teacher_env_rounds` | 教师交互步数 | B2F 中下降 |
| `if_teacher` | 是否使用教师 | B2F 中从 1 到 0 |

**关键洞察**：
> TCOD 的成功不仅体现在更高的成功率，还体现在更少的交互步数。这意味着学生学会了更高效的策略。

### 5.2 KL 散度的解读

**前两轮理解**：
- 知道 KL 散度衡量 student 和 teacher 的差异
- 不知道如何解读 KL 曲线

**本轮新增**：

**KL 曲线的三种模式**：
1. **稳定下降**：理想情况，student 逐步对齐 teacher
2. **先升后降**：常见，student 先探索后收敛
3. **持续上升**：失败信号，student 偏离 teacher

**TCOD vs OPD 的 KL 对比**：
- OPD：KL 持续上升，导致训练崩溃
- TCOD：KL 稳定或下降，训练更稳定

**关键洞察**：
> KL 散度的稳定性比绝对值更重要。TCOD 的核心贡献就是让 KL 曲线更稳定。

### 5.3 训练时间的分析

**前两轮理解**：
- 知道 TCOD 比 OPD 快 32%
- 不知道为什么

**本轮新增**：

**时间节省的来源**：
1. **早期 trajectory 更短**：F2B 限制最大步数，B2F 从中间开始
2. **数据收集更快**：短 trajectory 意味着更快的 rollout
3. **训练更稳定**：不需要重跑失败的实验

**关键洞察**：
> TCOD 的时间节省主要来自课程学习的早期阶段，而不是整个训练过程。随着课程推进到完整 trajectory，时间优势会减小。

---

## 六、项目管理层面的新理解

### 6.1 复现的优先级策略

**前两轮理解**：
- 知道需要复现多个实验
- 不知道如何安排优先级

**本轮新增**：

**优先级排序**：
```
P0: ALFWorld 1.5B TCOD-B2F  ← 最小可行实验
P1: ALFWorld 1.5B TCOD-F2B  ← 对比实验
P2: ALFWorld 3B OPD          ← Baseline
P3: ScienceWorld 1.5B        ← 泛化验证
P4: WebShop (可选)            ← 挑战性实验
```

**关键洞察**：
> 复现应该从最小可行实验开始，验证核心思想后再扩展。不要一开始就追求完整复现所有实验。

### 6.2 风险与缓解

**前两轮理解**：
- 知道 3090 显存有限
- 不知道具体风险点

**本轮新增**：

**主要风险**：
1. **OOM**：最常见，通过配置调优解决
2. **Teacher 模型缺失**：需要寻找替代或自己训练
3. **WebShop RAM 不足**：跳过或使用小数据集
4. **训练时间过长**：减少 total_steps 用于验证
5. **环境依赖问题**：alfworld、ScienceWorld 安装可能有问题

**缓解策略**：
- 先跑 100 steps 验证流程，再跑完整实验
- 准备多个配置文件（debug、quick、full）
- 记录每次运行的配置和结果

---

## 七、与前两轮文档的对比总结

### 7.1 知识层次的递进

| 层次 | 第一轮（面试准备） | 第二轮（五类能力） | 第三轮（复现计划） |
|------|-------------------|-------------------|-------------------|
| 概念理解 | ✅ 核心思想 | ✅ 深入原理 | ✅ 实现细节 |
| 代码理解 | ✅ 关键函数 | ✅ 数据流 | ✅ 配置参数 |
| 实验理解 | ✅ 结果分析 | ✅ 验证方法 | ✅ 资源规划 |
| 工程理解 | ❌ 缺乏 | ✅ 部分 | ✅ 完整 |

### 7.2 新增关键知识点

**架构层面**：
- GPU 分配公式
- `max_token_len_per_gpu` 的作用
- Sequence Parallelism (SP)
- vLLM 与 FSDP 的内存共享

**配置层面**：
- OOM 排查优先级
- GPU 分配 trade-off
- TCOD 特有参数含义

**工程层面**：
- Ray 临时目录问题
- Staleness control 实现
- 训练时间分析

**项目管理层面**：
- 复现优先级策略
- 风险与缓解
- 分阶段验证方法

### 7.3 面试回答的增强

**之前**：
> TCOD 用 temporal curriculum 控制 trajectory 深度，从短到长逐步扩展。

**现在**：
> TCOD 用 temporal curriculum 控制 trajectory 深度，从短到长逐步扩展。具体实现上，F2B 通过 `_get_effective_max_steps()` 动态计算最大步数，B2F 通过 `_linear_checkpoint_step()` 计算 teacher checkpoint 位置。课程增长速度由 `checkpoint_steps` 和 `total_steps` 的比例控制，线性 pacing 公式为 `k = k_start + floor(n / η)`。在 8×3090 上复现时，需要将 `max_token_len_per_gpu` 从 16384 降到 4096，启用 FSDP offload，并调整 GPU 分配为 2×actor + 2×teacher + 4×trainer。

---

## 八、下一步行动

### 8.1 立即行动

1. **环境搭建**：创建 conda 环境，安装 Trinity-RFT
2. **数据准备**：运行 `get_alfworld_data.py`
3. **模型下载**：下载 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct
4. **配置文件**：创建 3090 适配的 YAML 配置

### 8.2 短期目标（1 周内）

1. **跑通最小实验**：ALFWorld 1.5B TCOD-B2F (100 steps)
2. **验证核心指标**：`actor/pg_loss` 非零，`env_done` 上升
3. **记录问题**：遇到的 OOM、环境问题等

### 8.3 中期目标（2-3 周）

1. **完成 ALFWorld 实验**：TCOD-B2F/F2B + OPD baseline
2. **分析结果**：KL 曲线、成功率、训练时间
3. **扩展到 ScienceWorld**：验证泛化能力

---

*文档版本：v1.0*  
*最后更新：2026-06-26*  
*基于第三轮学习（复现计划制定）的增量知识*
