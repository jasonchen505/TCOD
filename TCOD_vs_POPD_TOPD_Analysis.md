# TCOD vs "Are Full Rollouts Necessary?" 深度对比分析

> 两篇关于 On-Policy Distillation 效率优化的同期工作  
> TCOD：多轮Agent的时间课程学习  
> POPD/TOPD：单轮推理的Horizon控制

---

## 一、论文基本信息对比

| 维度 | TCOD | POPD/TOPD |
|------|------|-----------|
| **标题** | Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents | Are Full Rollouts Necessary for On-Policy Distillation? |
| **来源** | 阿里巴巴通义实验室 + 香港中文大学 | 中科院自动化所 + 美团 |
| **发表** | COLM 2026 | Preprint (arXiv) |
| **arXiv** | 2604.24005 | - |
| **核心问题** | 多轮Agent中的Trajectory-Level KL Instability | 长horizon推理中的OPD效率问题 |
| **应用场景** | 多轮交互式Agent（ALFWorld, WebShop, ScienceWorld） | 单轮数学推理（AIME, AMC, MATH） |

---

## 二、问题定义对比

### 2.1 TCOD 识别的问题

**Trajectory-Level KL Instability**（轨迹级KL不稳定性）：

```
观察1：KL升级与成功率崩溃共现
  - 训练过程中KL散度持续上升
  - 任务成功率崩溃到接近零

观察2：即使收敛，初始KL过高
  - KL最终收敛但初始值（~1000）比收敛值（~60）高一个数量级
```

**根本原因**：多轮交互中的 **Compounding Error Amplification**（复合误差放大）
- 学生在第t步的错误会影响第t+1步的状态
- 随着trajectory深度增加，学生进入teacher不熟悉的状态
- Teacher的监督信号变得不可靠

**关键洞察**：
> Long-CoT不会导致同样的问题，因为Long-CoT在同一个环境状态上增加响应长度，不改变环境状态。而多轮Agent的每一步都会改变环境状态。

### 2.2 POPD/TOPD 识别的问题

**Long-Horizon OPD Inefficiency**（长horizon OPD的低效性）：

```
问题1：计算成本高
  - 生成完整rollout需要大量解码时间
  - KV cache内存占用大
  - Log-probability计算开销大

问题2：监督质量下降
  - 后期rollout位置的teacher反馈可能有噪声
  - 早期训练时student偏离teacher分布
  - 错误累积导致后期token进入不可靠区域
```

**根本原因**：
1. **Teacher Reliability Degradation**：teacher在远离其高概率区域时变得不可靠
2. **Future Noise Accumulation**：sequence-level OPD中，后期的噪声信号会传播到早期token

**关键洞察**：
> Token-level OPD不需要完整rollout来提供学习信号。因此，full rollouts may not always be necessary for OPD。

### 2.3 问题本质的异同

**相同点**：
- 都识别出 **后期rollout位置的teacher信号不可靠**
- 都认为 **full rollouts在早期训练中是浪费的**
- 都提出 **控制rollout长度** 作为解决方案

**不同点**：

| 维度 | TCOD | POPD/TOPD |
|------|------|-----------|
| **问题根源** | 多轮交互中的状态转移导致复合误差 | 单轮推理中的长序列导致teacher信号衰减 |
| **不可靠信号的来源** | 学生进入teacher不熟悉的状态空间 | 学生token偏离teacher的高概率区域 |
| **影响范围** | 影响整个trajectory的成功率 | 影响训练效率和收敛速度 |
| **任务类型** | 多轮Agent（需要环境交互） | 单轮推理（不需要环境交互） |

---

## 三、解决方案对比

### 3.1 TCOD 的解决方案

**核心思想**：Temporal Curriculum（时间课程学习）

**两个变体**：

**1. TCOD-F2B (Forward-to-Backward)**：
```python
# 从trajectory开始处学习，逐步扩展到完整长度
k = k_start + floor(n / η)  # n: 当前训练步, η: 增长率

# 学生执行前k步，teacher执行剩余步
for t in range(k):
    student_action = student.act(state)
    state = env.step(student_action)

# 只在学生执行的步上计算KL loss
loss = sum(KL(teacher || student) for t in range(k))
```

**2. TCOD-B2F (Backward-to-Forward)**：
```python
# Teacher先导航到接近成功状态，学生从那里接管
checkpoint_step = max(0, len(expert_actions) - reduction)

# Teacher执行前checkpoint_step步（stop gradient）
for t in range(checkpoint_step):
    env.step(expert_actions[t])

# 学生从checkpoint_step开始执行
for t in range(checkpoint_step, max_steps):
    student_action = student.act(state)
    state = env.step(student_action)

# 只在学生执行的步上计算KL loss
loss = sum(KL(teacher || student) for t in range(checkpoint_step, max_steps))
```

### 3.2 POPD/TOPD 的解决方案

**核心思想**：Horizon Control（horizon控制）

**两个变体**：

**1. POPD (Progressive OPD)**：
```python
# 渐进式扩展rollout horizon
H_k = min(T, H_0 + ΔH * floor(k / Δk))

# 只生成和蒸馏前H_k个token
for t in range(H_k):
    student_token = student.generate(context)
    teacher_logprob = teacher.logprob(context + student_token)
    student_logprob = student.logprob(context + student_token)
    
    # Token-level KL reward
    reward_t = teacher_logprob - student_logprob

# Token-level OPD loss
loss = sum(reward_t * student_score_t for t in range(H_k))
```

**2. TOPD (Truncated OPD)**：
```python
# 永久使用截断的rollout horizon
H = ρ * T  # ρ < 1, 截断比例

# 只生成和蒸馏前H个token
for t in range(H):
    student_token = student.generate(context)
    teacher_logprob = teacher.logprob(context + student_token)
    student_logprob = student.logprob(context + student_token)
    
    # Token-level KL reward
    reward_t = teacher_logprob - student_logprob

# Token-level OPD loss
loss = sum(reward_t * student_score_t for t in range(H))
```

### 3.3 解决方案的本质差异

**TCOD**：
- **控制什么**：学生执行的步数（在多轮环境中）
- **谁执行剩余步**：Teacher（B2F）或环境直接终止（F2B）
- **课程策略**：线性增长 `k = k_start + floor(n / η)`
- **最终目标**：学生最终能执行完整trajectory

**POPD/TOPD**：
- **控制什么**：生成和蒸馏的token数量（在单轮推理中）
- **谁执行剩余步**：不需要执行（直接截断）
- **课程策略**：POPD线性增长，TOPD固定截断
- **最终目标**：POPD最终生成完整rollout，TOPD永远使用截断

**关键区别**：
> TCOD需要teacher执行剩余步来提供"正确"的状态序列，而POPD/TOPD直接截断不需要teacher执行。这是因为多轮Agent的任务完成依赖于完整trajectory，而单轮推理的任务完成（答案正确性）不依赖于完整rollout。

---

## 四、理论分析对比

### 4.1 POPD/TOPD 的理论贡献

**Proposition: Accumulation of Future Noise**（未来噪声累积）：

```math
MSE_t(Â_seq_t - A*_t) = δ² Σ_{k=t}^{T} k²
```

**含义**：
- 当噪声随位置增长（σ_k = δk）时，第一个token的MSE为O(δ²T³)
- 早期token累积更多未来噪声
- Sequence-level OPD比token-level OPD更敏感

**设计原则**：
1. **缩短log-ratio horizon**：使用token-level OPD而非sequence-level
2. **控制rollout horizon**：避免在不可靠的后期位置蒸馏

### 4.2 TCOD 的理论分析

TCOD没有严格的数学命题，但提供了经验观察：

**观察1：Per-turn KL随turn index增长**
```
KL(t) 随 t 增长
→ 说明复合误差导致后期turn的teacher信号更不可靠
```

**观察2：TCOD保持KL稳定**
```
TCOD训练中KL曲线比OPD更稳定
→ 说明控制trajectory深度能避免KL不稳定
```

### 4.3 理论深度对比

| 维度 | TCOD | POPD/TOPD |
|------|------|-----------|
| **理论命题** | 无严格命题 | 有Proposition证明噪声累积 |
| **分析方法** | 经验观察 + 直觉解释 | 数学推导 + 实验验证 |
| **核心洞察** | 复合误差导致KL不稳定 | 未来噪声累积导致梯度污染 |
| **可解释性** | 中等 | 强 |

---

## 五、实验设计对比

### 5.1 TCOD 的实验设计

**环境**：
- ALFWorld（家居导航，30步）
- WebShop（电商购物，15步）
- ScienceWorld（科学实验，30步）

**模型**：
- Student: Qwen2.5-{1.5B, 3B, 7B}
- Teacher: Qwen2.5-7B-RL (GRPO-trained) 或 Qwen3-30B-A3B

**评估指标**：
- Success Rate (SR)
- Action Rounds（平均交互步数）
- KL Divergence

**关键结果**：
```
ALFWorld (Qwen2.5-3B student, 7B-RL teacher):
- Vanilla OPD: SR=65.72% (seen), 60.45% (unseen)
- TCOD-F2B:    SR=81.43% (seen), 79.19% (unseen)  → +15.71%
- TCOD-B2F:    SR=77.86% (seen), 70.90% (unseen)  → +12.14%
```

### 5.2 POPD/TOPD 的实验设计

**环境**：
- AIME24, AIME25, AMC23（数学推理）

**模型**：
- Student: R1-Distill-1.5B, OpenMath-1.5B
- Teacher: JustRL-R1-1.5B, JustRL-Nemotron-1.5B

**评估指标**：
- AIME/AMC Accuracy (avg@16)
- Training Time
- GPU Memory

**关键结果**：
```
R1-Distill-1.5B student, JustRL-R1-1.5B teacher:
- OPD:  AIME24=49.3%, Time=29.7h
- POPD: AIME24=51.0%, Time=10.1h  → 3× 效率提升
- TOPD (ρ=0.25): AIME24=51.3%, Time=10.2h
- TOPD (ρ=0.10): AIME24=50.7%, Time=5.3h  → 82% 时间减少
```

### 5.3 实验设计对比

| 维度 | TCOD | POPD/TOPD |
|------|------|-----------|
| **任务类型** | 多轮Agent | 单轮推理 |
| **环境交互** | 需要（ALFWorld, WebShop, ScienceWorld） | 不需要（数学题） |
| **成功定义** | 任务完成（二值） | 答案正确（二值） |
| **Trajectory长度** | 15-30步 | 15360 tokens |
| **效率指标** | 训练时间减少32% | 训练时间减少82% |
| **性能指标** | SR提升15-18点 | Accuracy提升1-2点 |

---

## 六、核心洞察对比

### 6.1 关于"为什么full rollouts不必要"

**TCOD的观点**：
> 在多轮Agent中，full rollouts会导致学生进入teacher不熟悉的状态，产生不可靠的监督信号。通过控制trajectory深度，可以让学生先学习可靠的早期步骤，再逐步扩展到完整任务。

**POPD/TOPD的观点**：
> 在单轮推理中，full rollouts会浪费计算在后期不可靠的token上。Token-level OPD不需要完整rollout来提供学习信号，因为：
> 1. 早期rollout段包含足够的teacher特征
> 2. 后期rollout段的teacher信号可能有噪声
> 3. 截断rollout已经能改变student的策略分布

### 6.2 关于"什么时候需要full rollouts"

**TCOD的隐含观点**：
> 最终还是需要full rollouts，因为多轮Agent的任务完成依赖于完整trajectory。课程学习只是让训练过程更稳定，最终目标是让学生能独立完成完整任务。

**POPD/TOPD的明确观点**：
> TOPD证明了在数学推理任务中，**永远不需要full rollouts**。即使只蒸馏10%的rollout，也能获得大部分性能提升。这是因为推理模式是horizon-independent的。

### 6.3 关于"teacher信号的可靠性"

**TCOD的分析**：
> Teacher信号的可靠性取决于**状态空间的熟悉度**。当学生进入teacher未探索过的状态时，teacher的信号变得不可靠。这在多轮交互中特别严重，因为每一步都会改变状态。

**POPD/TOPD的分析**：
> Teacher信号的可靠性取决于**token位置**。随着rollout位置加深，KL散度增大，teacher信号变得不可靠。这是因为在长序列中，student的分布逐渐偏离teacher的高概率区域。

---

## 七、方法论互补性分析

### 7.1 可以互相借鉴的点

**从POPD/TOPD到TCOD**：
1. **Token-level OPD的理论分析**：TCOD可以借鉴POPD/TOPD的噪声累积分析，为多轮Agent中的KL不稳定提供更严格的理论解释
2. **TOPD的截断策略**：在多轮Agent中，可以考虑永久使用截断trajectory（不一定要最终完成任务），只要能学到有用的skill
3. **Segment-level分析**：POPD/TOPD的segment分析方法可以用于分析多轮Agent中哪些turn最有价值

**从TCOD到POPD/TOPD**：
1. **B2F的teacher scaffolding**：在单轮推理中，可以让teacher先生成"解题思路"（类似checkpoint），student从那里接管
2. **环境状态的概念**：虽然单轮推理没有显式环境，但可以将"解题进度"视为隐式状态，控制student暴露的"状态深度"
3. **成功率指标**：POPD/TOPD只用了accuracy，可以引入更细粒度的指标（如解题步骤数）

### 7.2 潜在的结合方案

**方案1：Multi-turn Agent + TOPD**
```python
# 在多轮Agent中使用截断trajectory
# 不要求student完成任务，只要求学到有用的skill
for t in range(H):  # H < max_steps
    student_action = student.act(state)
    state = env.step(student_action)
    reward_t = KL(teacher || student)
    
# 只在截断的trajectory上蒸馏
loss = sum(reward_t for t in range(H))
```

**方案2：Single-turn Reasoning + B2F**
```python
# 在单轮推理中使用teacher scaffolding
# Teacher先生成解题框架，student填充细节
teacher_outline = teacher.generate_outline(problem)
student_solution = student.solve(problem, outline=teacher_outline)

# 只在student生成的部分蒸馏
loss = KL(teacher || student) on student_solution
```

**方案3：Adaptive Horizon Control**
```python
# 根据KL散度动态调整horizon
current_kl = compute_kl(teacher, student)
if current_kl > threshold:
    reduce_horizon()  # KL太高，缩短horizon
else:
    increase_horizon()  # KL稳定，扩展horizon
```

---

## 八、局限性对比

### 8.1 TCOD的局限性

1. **需要teacher预收集成功轨迹**（B2F）：如果teacher成功率低则数据不足
2. **线性pacing可能不是最优**：没有自适应机制
3. **只在文本环境验证**：没有扩展到多模态
4. **没有理论保证**：课程学习的有效性缺乏理论支撑

### 8.2 POPD/TOPD的局限性

1. **只在单轮推理验证**：没有扩展到多轮Agent
2. **TOPD在控制任务中失败**：当teacher特征依赖于horizon时，截断会丢失关键信息
3. **没有考虑环境交互**：假设rollout可以任意截断
4. **Teacher质量假设**：假设teacher是reasonably good的

### 8.3 共同局限性

1. **没有自适应机制**：都是预定义的schedule或固定的truncation ratio
2. **没有考虑teacher成本**：Teacher forward的计算开销没有优化
3. **没有跨任务迁移**：在一个任务上学到的schedule不一定适用于其他任务

---

## 九、对复现的启示

### 9.1 从POPD/TOPD学到的

**1. Token-level OPD的优势**：
- TCOD当前使用的是per-turn KL，可以考虑更细粒度的token-level KL
- Token-level OPD避免了future noise accumulation

**2. 效率优化的思路**：
- POPD的3×效率提升说明课程学习确实有效
- TOPD的82%时间减少说明截断策略可以大幅降低成本

**3. 评估方法**：
- 使用avg@16评估更稳定
- 控制评估的max_tokens很重要

### 9.2 TCOD的独特价值

**1. 多轮Agent的挑战**：
- 状态转移导致的复合误差是多轮Agent特有的
- 需要环境交互，不能简单截断

**2. B2F的teacher scaffolding**：
- 让teacher导航到checkpoint是创新点
- 可以在其他场景中应用

**3. 超越teacher的能力**：
- TCOD-B2F在hard set上超越了teacher
- 说明课程学习可以激发student的潜力

---

## 十、总结与展望

### 10.1 核心贡献对比

| 维度 | TCOD | POPD/TOPD |
|------|------|-----------|
| **问题发现** | 多轮Agent的KL不稳定 | 长horizon OPD的低效性 |
| **解决方案** | 时间课程学习（F2B/B2F） | Horizon控制（POPD/TOPD） |
| **理论分析** | 经验观察 | 数学证明 |
| **效率提升** | 32%时间减少 | 82%时间减少 |
| **性能提升** | 15-18点SR提升 | 1-2点Accuracy提升 |

### 10.2 未来方向

**共同方向**：
1. **自适应horizon控制**：根据KL散度、entropy等动态调整
2. **跨任务泛化**：在一个任务上学到的策略迁移到其他任务
3. **与RL结合**：将horizon控制思想应用到RLVR

**TCOD特有方向**：
1. **多模态Agent**：扩展到视觉-语言Agent
2. **真实世界部署**：在生产环境中验证
3. **Teacher-free curriculum**：不需要teacher的自监督课程学习

**POPD/TOPD特有方向**：
1. **代码生成**：验证在代码任务上的效果
2. **多轮对话**：扩展到对话场景
3. **自适应truncation ratio**：根据任务难度动态调整ρ

---

## 附录：关键公式对比

### TCOD 的 Pacing 公式
```python
# F2B: 限制学生执行的最大步数
k = k_start + floor(n / η)

# B2F: 计算teacher checkpoint位置
checkpoint_step = max(0, len(expert_actions) - reduction)
```

### POPD 的 Horizon 公式
```python
# 渐进式扩展rollout horizon
H_k = min(T, H_0 + ΔH * floor(k / Δk))
```

### TOPD 的 Truncation 公式
```python
# 固定截断比例
H = ρ * T  # ρ < 1
```

### Token-level OPD 的梯度公式
```math
∇J_tok = E[Σ_{t=1}^{H} r_t · s_t]
```
其中：
- r_t = log(π_teacher(y_t|y_{<t},x) / π_student(y_t|y_{<t},x))
- s_t = ∇log π_student(y_t|y_{<t},x)

### Sequence-level OPD 的梯度公式
```math
∇J_seq = E[Σ_{t=1}^{T} (Σ_{k=t}^{T} r_k) · s_t]
```

---

*文档版本：v1.0*  
*最后更新：2026-06-26*  
*基于两篇论文的深度对比分析*
