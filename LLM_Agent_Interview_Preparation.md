# LLM & Agent 应用/后训练 面试准备指南

> 基于 TCOD 项目（Temporal Curriculum for On-Policy Distillation）的深度技术梳理与面试考察点整理

---

## 一、项目核心概述

### 1.1 研究背景与动机

**TCOD** (Temporal Curriculum On-Policy Distillation) 是阿里巴巴通义实验室发表于 COLM 2026 的研究工作，解决的核心问题是：

> **在多轮交互式 Agent 任务中，标准 On-Policy Distillation (OPD) 会出现 Trajectory-Level KL Instability，导致训练崩溃或性能下降。**

### 1.2 核心问题定义

**Trajectory-Level KL Instability** 的两个关键观察：
1. **KL 升级与成功率崩溃共现**：训练过程中 KL 散度持续上升，任务成功率崩溃到接近零
2. **即使收敛，初始 KL 过高**：KL 最终收敛但初始值（~1000）比收敛值（~60）高一个数量级

**根本原因**：多轮交互中的 **Compounding Error Amplification**（复合误差放大）
- 学生模型在第 t 步的错误会影响第 t+1 步的状态
- 随着 trajectory 深度增加，学生进入 teacher 不熟悉的状态
- teacher 的监督信号变得不可靠甚至无法学习

### 1.3 解决方案

**TCOD** 提出 **Temporal Curriculum**（时间课程学习）：
- 控制暴露给学生的 trajectory 深度
- 从短到长逐步扩展，使用课程调度策略

**两个变体**：
1. **TCOD-F2B** (Forward-to-Backward)：从 trajectory 开始处学习，逐步扩展到完整长度
2. **TCOD-B2F** (Backward-to-Forward)：从 trajectory 末尾开始，teacher 导航到接近成功状态，学生从那里接管

---

## 二、核心技术细节

### 2.1 OPD 基础公式

**标准 OPD 目标函数**：
```
L_OPD(θ) = E_{τ~π_θ} [ Σ_{t=0}^{T-1} D_KL( π_φ(a_t|h_t) || π_θ(a_t|h_t) ) ]
```

其中：
- `π_φ`：teacher 策略
- `π_θ`：student 策略
- `h_t`：到时间步 t 的完整交互历史
- `D_KL`：KL 散度

**Advantage 计算**（代码实现）：
```python
# on_policy_distill_advantage.py:35
advantages = kl_coef * (teacher_log_probs - old_log_probs)
```

### 2.2 TCOD-F2B 算法

**核心思想**：限制学生在 trajectory 开始处的最大交互步数

**目标函数**：
```
L_TCOD_F2B(θ) = E_{τ~π_θ} [ Σ_{t=0}^{k-1} D_KL( π_φ(a_t|h_t) || π_θ(a_t|h_t) ) ]
```

**Pacing 策略**（线性增长）：
```python
k = k_start + floor(n / η)  # n: 当前训练步, η: 增长率
```

**代码实现**（TCOD_f2b_workflow.py）：
```python
def _get_effective_max_steps(self, current_step: int) -> int:
    """动态计算当前训练步的最大交互步数"""
    k = self.k_start + current_step // self.eta
    return min(k, self.max_env_steps)
```

### 2.3 TCOD-B2F 算法

**核心思想**：teacher 先导航到接近成功状态，学生从那里接管

**目标函数**：
```
L_TCOD_B2F(θ) = E_{τ~(π_φ,π_θ)} [ Σ_{t=L-k+1}^{T-1} D_KL( π_φ(a_t|h_t) || π_θ(a_t|h_t) ) ]
```

**关键步骤**：
1. 使用 teacher 预收集成功 trajectory τ*
2. Teacher 执行前 L-k 步（stop gradient）
3. 学生从第 L-k 步开始接管

**代码实现**（TCOD_b2f_workflow.py）：
```python
def _linear_checkpoint_step(self, predefined_actions: Optional[List[str]] = None) -> Optional[int]:
    """计算 checkpoint step：线性衰减策略"""
    if predefined_actions is None or len(predefined_actions) == 0:
        return 0
    current_step = self._current_training_step
    max_expert_actions = len(predefined_actions) - 1
    reduction = current_step // self.checkpoint_steps
    checkpoint_step = max(0, min(max_expert_actions - reduction, max_expert_actions))
    return checkpoint_step
```

### 2.4 异步训练与 Staleness Control

**异步架构**：
- **Explorer**（Ray actor）：运行 workflows，收集 experiences
- **Trainer**（Ray actor）：从 buffer 读取数据，执行梯度更新
- **Synchronizer**：管理 Explorer 和 Trainer 之间的权重同步

**Staleness-Aware Experience Replay**：
```python
# 配置 staleness filter
algorithm:
  sample_strategy: staleness_control
  sample_strategy_args:
    max_staleness: 2  # 丢弃过期的 experience
```

**作用**：在异步训练中保持 on-policy 特性，避免使用过期策略收集的数据

---

## 三、框架架构理解

### 3.1 Explorer-Trainer 架构

```
┌─────────────────┐    ┌─────────────────┐
│   Explorer 1    │    │   Explorer 2    │
│  (Ray Actor)    │    │  (Ray Actor)    │
│  - 运行 workflows │    │  - 运行 workflows │
│  - 收集 experiences│    │  - 收集 experiences│
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────┐
         │     Buffer      │
         │  (Queue/File)   │
         │  - 解耦 Explorer │
         │    和 Trainer   │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │    Trainer      │
         │  (Ray Actor)    │
         │  - 计算 advantages│
         │  - 梯度更新       │
         └─────────────────┘
```

### 3.2 核心组件

**Experience 类**（experience.py）：
```python
@dataclass
class Experience:
    eid: EID                          # 唯一标识符
    tokens: Optional[Tensor]          # [seq_length]
    prompt_length: int                # prompt 长度
    logprobs: Optional[Tensor]        # [resp_length] student logprobs
    teacher_logprobs: Optional[Tensor] # [resp_length] teacher logprobs
    reward: Optional[float]           # 奖励
    advantages: Optional[Tensor]      # [resp_length] 优势值
    # ... 其他字段
```

**Workflow 注册机制**（workflows/__init__.py）：
```python
WORKFLOWS: Registry = Registry(
    "workflows",
    default_mapping={
        "OPD_alfworld_workflow": "...",
        "TCOD_f2b_alfworld_workflow": "...",
        "TCOD_b2f_alfworld_workflow": "...",
        # ... 9 个 TCOD workflow（3 环境 × 3 方法）
    },
)
```

### 3.3 关键配置参数

```yaml
# TCOD 配置示例（tcod_b2f.yaml）
algorithm:
  advantage_fn: multi_turn_opd      # 使用 MultiTurnOpdAdvantage
  advantage_fn_args:
    kl_coef: 1.0                    # KL 散度系数

buffer:
  total_steps: 250                  # 总训练步数
  batch_size: 16                    # Explorer batch size
  train_batch_size: 64              # Trainer batch size

workflow_args:
  checkpoint_strategy: linear       # checkpoint 策略
  checkpoint_steps: 5               # checkpoint 步数
  total_steps: 250                  # 总步数（用于计算 k）

explorer:
  auxiliary_models:                 # Teacher 模型配置
    - model_path: /path/to/teacher
```

---

## 四、面试考察点与深挖问题

### 4.1 基础概念考察

#### Q1: 什么是 On-Policy Distillation？与 Off-Policy Distillation 的区别？

**标准答案**：
- **On-Policy Distillation**：student 自己生成 trajectory，teacher 在 student 生成的 trajectory 上提供监督信号
- **Off-Policy Distillation**：使用 teacher 预先收集的 trajectory 进行训练

**深挖点**：
- OPD 的优势：避免 exposure bias（训练和推理分布一致）
- OPD 的劣势：需要 teacher 实时推理，计算开销大
- 为什么 OPD 在单轮任务（数学推理）中成功但在多轮 Agent 中失败？

#### Q2: 解释 KL 散度在 OPD 中的作用

**标准答案**：
```python
# 核心公式
advantages = kl_coef * (teacher_log_probs - student_log_probs)
```

- KL 散度衡量 teacher 和 student 策略的差异
- 作为 advantage 信号：student 应该增加 teacher 概率高的 action 的概率
- `kl_coef` 控制蒸馏信号的强度

**深挖点**：
- 为什么用 `teacher_log_probs - student_log_probs` 而不是反过来？
- Forward KL vs Backward KL 的区别？
- KL 散度为 0 意味着什么？

#### Q3: 多轮 Agent 与单轮任务的本质区别是什么？

**标准答案**：
1. **状态依赖**：多轮 Agent 的每一步都会改变环境状态，影响后续决策
2. **复合误差**：早期错误会累积，导致后续状态偏离 teacher 的经验分布
3. **长期信用分配**：需要将最终奖励分配到整个 trajectory 的每一步

**深挖点**：
- 为什么 Long-CoT（长链推理）不会导致同样的问题？
- 答案：Long-CoT 在同一个环境状态上增加响应长度，不改变环境状态

### 4.2 TCOD 方法论考察

#### Q4: TCOD-F2B 和 TCOD-B2F 的核心区别？各自适用场景？

**标准答案**：

| 特性 | TCOD-F2B | TCOD-B2F |
|------|----------|----------|
| 起始点 | trajectory 开始 | trajectory 末尾附近 |
| 学生控制 | 从头开始，逐步扩展 | 从中间接管，逐步向前扩展 |
| Teacher 参与 | 不参与 rollout | 导航到 checkpoint 状态 |
| 计算效率 | 更高效（短 trajectory） | 较低（需要 teacher 导航） |
| 适用场景 | 小模型、简单任务 | 大模型、困难任务 |

**深挖点**：
- 为什么小模型更适合 F2B？（从基础开始，避免早期错误累积）
- 为什么大模型更适合 B2F？（利用 teacher 的 scaffolding，在困难任务上获得更好的初始化）
- 如何选择 `k_start` 和 `η`？

#### Q5: 解释 linear pacing strategy 的设计

**标准答案**：
```python
k = k_start + floor(n / η)  # n: 当前训练步, η: 增长率
```

- `k_start`：初始步数（通常为 1）
- `η`：控制课程增长速度（越大越慢）
- `n / η`：每 η 步增加 1 步

**深挖点**：
- 为什么用线性增长而不是指数增长？
- 答案：线性增长更稳定，易于调参
- `η` 的选择对性能的影响？（论文中 η ∈ {2, 4, 6}，变化 <2%）

#### Q6: TCOD-B2F 中的 train-test mismatch 如何解决？

**标准答案**：
- 训练时：学生从 teacher 导航的 checkpoint 开始
- 测试时：学生需要从头开始执行完整 trajectory

**解决方案**：
- 逐步减少 teacher 的前缀：从 L-1 步减少到 0 步
- 训练结束时，学生从初始状态执行完整 trajectory

**深挖点**：
- 如何验证这种 smooth curriculum transition 有效？
- 答案：测试集上的 end-to-end success rate 随训练步数稳步上升

### 4.3 实现细节考察

#### Q7: 为什么需要 Staleness Control？

**标准答案**：
```yaml
algorithm:
  sample_strategy: staleness_control
  sample_strategy_args:
    max_staleness: 2
```

- 异步训练中，Explorer 和 Trainer 使用不同版本的策略
- 过期的 experience 是 off-policy 的，会损害训练效果
- Staleness filter 丢弃版本差异过大的 experience

**深挖点**：
- `max_staleness = 2` 是如何确定的？
- 答案：经验上在样本效率和 on-policy 约束之间取得平衡
- 为什么不直接用同步训练？
- 答案：异步训练提高 GPU 利用率，Explorer 和 Trainer 可以并行工作

#### Q8: Teacher 和 Student 的 logprobs 如何对齐？

**标准答案**（OPD_workflow.py）：
```python
# 1. Student 生成 response，记录 logprobs
responses = await self.model.chat_async(memory, **kwargs)

# 2. Teacher 在相同输入上计算 logprobs
teacher_logprobs = await self.teacher_model.logprobs_async(
    tokens=response.tokens.tolist(),
    temperature=self.temperature,
)

# 3. 对齐 response 部分
resp_start = response.prompt_length - 1
teacher_resp_logprobs = teacher_logprobs[resp_start:]
student_resp_logprobs = response.logprobs

# 4. 验证长度一致
assert len(teacher_resp_logprobs) == len(student_resp_logprobs)
```

**深挖点**：
- 为什么用 `response.prompt_length - 1` 而不是 `response.prompt_length`？
- 答案：logprobs 是 next-token prediction，prompt 的最后一个 token 的 logprob 对应 response 的第一个 token
- 如果长度不一致怎么办？
- 答案：assert 失败，说明实现有 bug 或 tokenizer 不一致

#### Q9: MultiTurnOpdAdvantage 与 OnPolicyDistillAdvantage 的区别？

**标准答案**：
```python
# OnPolicyDistillAdvantage：单轮 OPD
# batch = [num_samples]

# MultiTurnOpdAdvantage：多轮 OPD
# batch = [num_turns]（每个 turn 一行）
```

**关键区别**：
1. MultiTurn 版本需要按 trajectory 分组计算 KL
2. 使用 `unique_ids` 标识 trajectory 和 turn
3. 添加 trajectory-level metrics（kl/trajectory_mean）

**深挖点**：
- `unique_ids` 的格式是什么？
- 答案：`batch/task/run/step/suffix`
- 如何计算 trajectory-level KL？
- 答案：按 `run_id` 分组，累加每个 turn 的 KL

#### Q10: 如何处理 teacher 和 student 的长度不匹配？

**标准答案**：
```python
teacher_valid_mask = exps.batch.get("teacher_logprobs_valid_mask")
if teacher_valid_mask is not None:
    # 使用 mask 处理 padding
    effective_mask = response_mask & teacher_valid_mask
else:
    # 标准 OPD：teacher 和 student 使用相同 mask
    effective_mask = response_mask
```

**深挖点**：
- 什么时候会出现长度不匹配？
- 答案：hint workflow 中，teacher 可能生成不同长度的 response
- 如何处理？
- 答案：使用 `teacher_logprobs_valid_mask` 标记有效位置

### 4.4 系统设计考察

#### Q11: 描述 Trinity-RFT 的整体架构

**标准答案**：
```
┌─────────────────────────────────────────────────────┐
│                    Trinity-RFT                       │
├─────────────────────────────────────────────────────┤
│  CLI Layer: trinity run --config config.yaml        │
├─────────────────────────────────────────────────────┤
│  Manager Layer:                                     │
│    - ConfigManager: 配置管理                          │
│    - StateManager: 状态管理                           │
│    - Synchronizer: 权重同步                           │
├─────────────────────────────────────────────────────┤
│  Explorer Layer:                                    │
│    - Explorer (Ray Actor): 运行 workflows            │
│    - WorkflowRunner: 执行单个 workflow               │
│    - ModelProxy: 与 vLLM 交互                       │
├─────────────────────────────────────────────────────┤
│  Trainer Layer:                                     │
│    - Trainer (Ray Actor): 模型训练                   │
│    - VerlTrainer: 基于 verl 的实现                   │
│    - AdvantageFn: 计算 advantages                   │
├─────────────────────────────────────────────────────┤
│  Buffer Layer:                                      │
│    - Buffer: 数据缓冲                               │
│    - BufferReader/Writer: 读写接口                   │
│    - TaskScheduler: 任务调度                         │
└─────────────────────────────────────────────────────┘
```

**深挖点**：
- 为什么用 Ray 而不是 PyTorch DDP？
- 答案：Ray 更灵活，支持异构任务，易于扩展
- 权重同步如何工作？
- 答案：使用 NCCL，Explorer 定期从 Trainer 拉取最新权重

#### Q12: 如何设计一个可扩展的 Workflow 系统？

**标准答案**：
```python
# 1. 定义基类
class Workflow:
    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False
    
    async def run_async(self) -> List[Experience]:
        raise NotImplementedError

# 2. 注册机制
WORKFLOWS = Registry("workflows")

@WORKFLOWS.register_module("my_workflow")
class MyWorkflow(Workflow):
    ...

# 3. 配置驱动
workflow_type = config["buffer"]["explorer_input"]["default_workflow_type"]
workflow_class = WORKFLOWS.get(workflow_type)
```

**深挖点**：
- 为什么用注册机制而不是直接 import？
- 答案：解耦配置和代码，支持插件化扩展
- 如何处理 workflow 的状态管理？
- 答案：通过 `reset()` 和 `set_training_progress()` 方法

---

## 五、LLM 后训练核心知识

### 5.1 训练范式对比

| 方法 | 数据来源 | 监督信号 | 优点 | 缺点 |
|------|----------|----------|------|------|
| SFT | Teacher 轨迹 | Token-level CE | 简单高效 | Exposure bias |
| OPD | Student 轨迹 | Teacher logprobs | On-policy | 计算开销大 |
| RL (PPO/GRPO) | Student 轨迹 | Reward model | 可优化任意目标 | 训练不稳定 |
| DPO | Paired 数据 | Preference | 无需 reward model | 需要高质量数据 |

### 5.2 Advantage Function 设计

**PPO Advantage**：
```python
# 使用 GAE (Generalized Advantage Estimation)
advantages = compute_gae(rewards, values, gamma, lam)
```

**GRPO Advantage**：
```python
# Group Relative Policy Optimization
advantages = (rewards - group_mean) / group_std
```

**OPD Advantage**：
```python
# On-Policy Distillation
advantages = kl_coef * (teacher_logprobs - student_logprobs)
```

### 5.3 KL 散度控制

**Forward KL**：
```
D_KL(π_θ || π_φ) = Σ π_θ(x) log(π_θ(x) / π_φ(x))
```
- 模式覆盖（mode covering）
- 倾向于覆盖 teacher 的所有模式

**Backward KL**：
```
D_KL(π_φ || π_θ) = Σ π_φ(x) log(π_φ(x) / π_θ(x))
```
- 模式寻求（mode seeking）
- 倾向于集中在 teacher 的高概率区域

**TCOD 使用 Backward KL**：因为希望 student 集中学习 teacher 认为好的 action

### 5.4 分布式训练关键概念

**FSDP (Fully Sharded Data Parallel)**：
- 模型参数分片到多个 GPU
- 减少单 GPU 内存占用
- 通信开销换取内存效率

**NCCL 通信**：
- GPU-to-GPU 高速通信
- 支持 AllReduce, AllGather 等集合操作
- 用于权重同步和梯度聚合

**vLLM 推理优化**：
- PagedAttention：高效的 attention 实现
- Continuous Batching：动态批处理
- Prefix Caching：复用 prompt 的 KV cache

---

## 六、Agent 环境理解

### 6.1 ALFWorld 环境

**任务类型**：文本化家庭环境导航与物体操作
- 6 个类别：Pick, Clean, Heat, Cool, Examine, Put
- 观察：文本描述的环境状态
- 动作：`go to X`, `take X`, `put X in/on Y` 等

**关键挑战**：
- 长 horizon（最多 30 步）
- 状态空间大
- 需要多步推理

### 6.2 WebShop 环境

**任务类型**：电商购物平台
- 目标：根据用户需求找到合适商品
- 观察：网页文本
- 动作：搜索、点击、选择等

**关键挑战**：
- 需要理解用户需求
- 需要在大量商品中筛选
- 需要多轮交互

### 6.3 ScienceWorld 环境

**任务类型**：科学实验推理
- 目标：完成科学实验
- 观察：实验环境描述
- 动作：操作实验器材

**关键挑战**：
- 需要科学知识
- 实验步骤复杂
- 需要精确操作

---

## 七、面试实战技巧

### 7.1 如何介绍 TCOD 项目

**30 秒版本**：
> TCOD 是一个多轮 Agent 蒸馏框架，解决了标准 OPD 在多轮任务中的 KL 不稳定问题。核心思想是使用时间课程学习，从短到长逐步扩展 trajectory 深度，避免复合误差累积。

**3 分钟版本**：
> 1. 问题：标准 OPD 在多轮 Agent 任务中会出现 KL 散度飙升、成功率崩溃
> 2. 原因：复合误差放大，学生进入 teacher 不熟悉的状态
> 3. 方案：Temporal Curriculum，控制 trajectory 深度
> 4. 两个变体：F2B（从头开始）和 B2F（从末尾开始）
> 5. 结果：在 ALFWorld、WebShop、ScienceWorld 上提升 18 点

### 7.2 常见追问与应对

**Q: 为什么不直接用 RL？**
> A: RL 在多轮任务中面临稀疏奖励和低样本效率问题。OPD 提供 dense token-level 信号，收敛更快。

**Q: 与 SFT 相比有什么优势？**
> A: SFT 只在 teacher 轨迹上训练，有 exposure bias。OPD 在 student 自己的轨迹上训练，分布一致。

**Q: 如何处理 teacher 不够强的情况？**
> A: 论文发现 domain-specific teacher（如 GRPO 训练的 7B）比更大的 general teacher（如 30B）更有效。关键是 teacher 在目标任务上的表现。

**Q: TCOD 的局限性是什么？**
> A: 1. 需要 teacher 预收集成功轨迹（B2F）；2. 线性 pacing 可能不是最优；3. 目前只在文本环境验证。

### 7.3 代码实现亮点

**1. 模块化设计**：
```python
# 清晰的组件分离
- AdvantageFn：计算 advantage
- Workflow：定义交互逻辑
- Buffer：数据管理
- Trainer：模型训练
```

**2. 注册机制**：
```python
@WORKFLOWS.register_module("TCOD_b2f_alfworld_workflow")
class TCOD_b2f_alfworld_workflow(Workflow):
    ...
```

**3. 异步支持**：
```python
async def run_async(self) -> List[Experience]:
    # 支持异步环境交互
    responses = await self.model.chat_async(memory, **kwargs)
```

**4. 配置驱动**：
```yaml
# 一个 YAML 文件定义完整实验
algorithm:
  advantage_fn: multi_turn_opd
workflow_args:
  checkpoint_strategy: linear
```

---

## 八、扩展知识准备

### 8.1 相关论文

1. **On-Policy Distillation**：
   - "On-Policy Distillation of Language Models" (Agarwal et al., 2024)
   - "Stable Distillation" (Jang et al., 2026)

2. **Curriculum Learning**：
   - "Curriculum Learning" (Bengio et al., 2009)
   - "Beyond Random: Difficulty-Aware Curriculum" (Zhang et al., 2026)

3. **Agent Training**：
   - "ReAct" (Yao et al., 2022)
   - "GRPO" (Guo et al., 2025)

4. **RLHF/DPO**：
   - "Training language models to follow instructions" (Ouyang et al., 2022)
   - "Direct Preference Optimization" (Rafailov et al., 2023)

### 8.2 面试高频概念

**1. Exposure Bias**：
- 训练时看到 teacher 轨迹，推理时需要自己生成
- 错误累积，性能下降

**2. Credit Assignment**：
- 如何将最终奖励分配到每一步
- 多轮任务中的核心挑战

**3. On-policy vs Off-policy**：
- On-policy：用当前策略收集数据
- Off-policy：用历史数据训练
- On-policy 更稳定但样本效率低

**4. KL Divergence Control**：
- 防止策略偏离太远
- 保持训练稳定性

**5. Advantage Estimation**：
- 衡量某个 action 相对于平均的好坏
- PPO 用 GAE，GRPO 用 group normalization

---

## 九、项目代码导航

### 9.1 核心文件清单

```
trinity/
├── algorithm/
│   └── advantage_fn/
│       └── on_policy_distill_advantage.py  # OPD advantage 计算
├── common/
│   ├── experience.py                       # Experience 数据类
│   └── workflows/
│       ├── __init__.py                     # Workflow 注册
│       └── envs/TCOD/
│           ├── alfworld/
│           │   ├── OPD_workflow.py         # ALFWorld OPD
│           │   ├── TCOD_b2f_workflow.py    # ALFWorld B2F
│           │   └── TCOD_f2b_workflow.py    # ALFWorld F2B
│           ├── webshop/
│           └── scienceworld/
```

### 9.2 关键代码段

**1. Advantage 计算**（on_policy_distill_advantage.py:35）：
```python
advantages = kl_coef * (teacher_log_probs - old_log_probs)
```

**2. Checkpoint 计算**（TCOD_b2f_workflow.py:86-94）：
```python
def _linear_checkpoint_step(self, predefined_actions):
    reduction = current_step // self.checkpoint_steps
    checkpoint_step = max(0, min(max_expert_actions - reduction, max_expert_actions))
    return checkpoint_step
```

**3. Teacher Logprobs 获取**（OPD_workflow.py:196-211）：
```python
teacher_logprobs = await self.teacher_model.logprobs_async(
    tokens=response.tokens.tolist(),
    temperature=self.temperature,
)
resp_start = response.prompt_length - 1
teacher_resp_logprobs = teacher_logprobs[resp_start:]
```

**4. Workflow 注册**（workflows/__init__.py:58-66）：
```python
"OPD_alfworld_workflow": "...OPD_workflow.OnPolicyDistillVerlAgentAlfworldWorkflow",
"TCOD_f2b_alfworld_workflow": "...TCOD_f2b_workflow.TCOD_f2b_alfworld_workflow",
"TCOD_b2f_alfworld_workflow": "...TCOD_b2f_workflow.TCOD_b2f_alfworld_workflow",
```

---

## 十、模拟面试问答

### 问题 1：请解释 TCOD 的核心思想

**回答框架**：
1. 问题背景：多轮 Agent 蒸馏的挑战
2. 关键观察：KL 不稳定现象
3. 根本原因：复合误差放大
4. 解决方案：时间课程学习
5. 两个变体：F2B 和 B2F
6. 实验结果：提升 18 点

### 问题 2：TCOD-B2F 如何解决 train-test mismatch？

**回答要点**：
1. 训练时：teacher 导航到 checkpoint，学生从那里开始
2. 测试时：学生从头开始
3. 解决方案：逐步减少 teacher 前缀
4. 验证：测试集 success rate 稳步上升

### 问题 3：如何选择 F2B 和 B2F？

**回答要点**：
1. 小模型（≤3B）：F2B 更好
   - 从基础开始，避免早期错误累积
   - 建立 distributional robustness
2. 大模型（≥7B）：B2F 更好
   - 利用 teacher scaffolding
   - 在困难任务上获得更好的初始化
3. 困难任务：B2F 更好
   - teacher 导航提供 dense reward signal

### 问题 4：描述一下代码的整体架构

**回答要点**：
1. Explorer-Trainer 架构
2. Ray 分布式计算
3. Buffer 解耦数据生产消费
4. Workflow 定义交互逻辑
5. AdvantageFn 计算训练信号

### 问题 5：如何处理异步训练中的 staleness 问题？

**回答要点**：
1. 问题：Explorer 和 Trainer 使用不同版本策略
2. 解决：Staleness filter，丢弃过期 experience
3. 配置：`max_staleness: 2`
4. 效果：平衡样本效率和 on-policy 约束

---

## 十一、总结与建议

### 11.1 面试准备重点

1. **理解核心问题**：KL 不稳定的根本原因
2. **掌握解决方案**：F2B 和 B2F 的区别与适用场景
3. **熟悉代码实现**：关键函数和数据流
4. **了解系统设计**：Explorer-Trainer 架构
5. **准备扩展问题**：与其他方法的对比

### 11.2 常见误区

1. ❌ 认为 TCOD 是全新的方法
   - ✅ TCOD 是 OPD 的改进，核心是课程学习思想

2. ❌ 认为 B2F 总是比 F2B 好
   - ✅ 取决于模型大小和任务难度

3. ❌ 忽视实现细节
   - ✅ Staleness control、logprobs 对齐等细节很重要

### 11.3 加分项

1. **能解释论文中的数学公式**
2. **能描述代码的具体实现**
3. **能讨论局限性和改进方向**
4. **能联系其他相关工作**

---

## 附录：关键术语表

| 术语 | 解释 |
|------|------|
| OPD | On-Policy Distillation，在线策略蒸馏 |
| KL | Kullback-Leibler Divergence，KL 散度 |
| Trajectory | 交互轨迹，包含状态-动作序列 |
| Compounding Error | 复合误差，早期错误累积放大 |
| Curriculum Learning | 课程学习，从易到难训练 |
| Staleness | 过期程度，衡量数据的新鲜度 |
| Advantage | 优势值，衡量 action 的好坏 |
| F2B | Forward-to-Backward，从前往后扩展 |
| B2F | Backward-to-Forward，从后往前扩展 |
| Pacing | 步调，控制课程增长速度 |

---

*文档版本：v1.0*
*最后更新：2026-06-26*
*基于 TCOD 项目 commit: [查看最新 commit]*
