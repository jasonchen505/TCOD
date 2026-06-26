# 技术面试五类能力深度应对手册

> 基于 TCOD（多轮Agent蒸馏）与 OPD（单轮数学推理蒸馏）两个项目的深度实践  
> 每个问题：背景 → 核心机制 → 局限与风险 → 改进与验证  
> 面试官考察的不是"你知道什么"，而是"你为什么这么做"、"你怎么证明"、"你遇到什么问题"

---

## 目录

- [能力一：底层原理与方法设计理解](#能力一底层原理与方法设计理解)
- [能力二：实验和方案验证能力](#能力二实验和方案验证能力)
- [能力三：问题定位与排查能力](#能力三问题定位与排查能力)
- [能力四：工程落地能力](#能力四工程落地能力)
- [能力五：业务与实际场景理解](#能力五业务与实际场景理解)

---

## 能力一：底层原理与方法设计理解

### 面试官想看什么

面试官不是只想听你背概念。他想看到：
- 这个方法解决什么**真实痛点**
- 为什么 reward **要这样设计**而不是那样
- 这个方法**什么时候会失败**
- 如果要改进，你**知道往哪个方向**

---

### Q1.1: OPD 到底解决什么问题？为什么不能只用 GRPO 或 SFT？

**问题背景**：

传统知识蒸馏（SFT）在 teacher 固定数据上训练，存在 **distribution mismatch**：学生部署时面对的是自己的分布，而非 teacher 的分布。GRPO 是 outcome-level RL，每个 response 只有一个最终 reward（正确/错误），在数学推理任务上早期 reward 极度稀疏。

**核心机制**：

OPD 让 student 自己 rollout（on-policy），然后在每个 token 位置用 teacher 的分布计算 KL reward。这样即使最后答案错了，中间每个 token 都有 teacher 的分布信号。

**实测证据**（OPD 项目）：
```
GRPO step 4:  actor/pg_loss=0.00497, critic/score/mean=0.03125（32 个样本约 1 个正确）
GRPO step 10: actor/pg_loss=0.0, critic/score/mean=0.0（全错，无梯度）

OPD step 1:   actor/pg_loss=0.2616, critic/score/mean=-0.2616（dense signal，非零）
OPD step 10:  actor/pg_loss=0.2738, critic/score/mean=-0.2614（稳定非零）
```

**对比表**：

| 方法 | 数据来源 | Reward/Loss | On-Policy | 信号密度 |
|------|----------|-------------|-----------|----------|
| SFT | 固定示范数据 | CE loss | 否 | token-level（固定答案） |
| DPO | preference pair | preference obj | 否 | pair-level |
| GRPO | student rollout | outcome reward | 是 | response-level sparse |
| OPD | student rollout + teacher | token-level KL | 是 | token-level dense |

**局限与风险**：
1. Teacher 成本高：每个 token 都要 teacher forward，比 GRPO 贵
2. Teacher/student thinking pattern 必须兼容：格式不兼容会错对齐
3. Teacher 不一定正确：OPD 会蒸馏 teacher 的 bias
4. Token KL ≠ 任务正确性：可能学到 teacher 风格而非正确答案

**改进方向**：
- OPD + GRPO hybrid：`combined_adv = direct_adv + weight * grpo_adv`
- Teacher-aligned prompt selection：只在 student 薄弱分布上蒸馏
- 多 teacher ensemble 或 verifier 过滤 teacher 错误

---

### Q1.2: TCOD 解决什么问题？为什么标准 OPD 在多轮 Agent 中会失败？

**问题背景**：

标准 OPD 在单轮数学推理中成功，但在多轮 Agent 任务（ALFWorld、WebShop、ScienceWorld）中会出现 **Trajectory-Level KL Instability**。

**核心观察**（TCOD 论文）：
1. **KL 升级与成功率崩溃共现**：训练过程中 KL 散度持续上升，任务成功率崩溃到接近零
2. **即使收敛，初始 KL 过高**：KL 最终收敛但初始值（~1000）比收敛值（~60）高一个数量级

**根本原因**：多轮交互中的 **Compounding Error Amplification**（复合误差放大）
- 学生模型在第 t 步的错误会影响第 t+1 步的状态
- 随着 trajectory 深度增加，学生进入 teacher 不熟悉的状态
- Teacher 的监督信号变得不可靠甚至无法学习

**关键洞察**：
> Long-CoT（长链推理）不会导致同样的问题，因为 Long-CoT 在同一个环境状态上增加响应长度，不改变环境状态。而多轮 Agent 的每一步都会改变环境状态。

**解决方案**：

TCOD 提出 **Temporal Curriculum**（时间课程学习）：
- 控制暴露给学生的 trajectory 深度
- 从短到长逐步扩展，使用课程调度策略

**两个变体**：

| 特性 | TCOD-F2B | TCOD-B2F |
|------|----------|----------|
| 起始点 | trajectory 开始 | trajectory 末尾附近 |
| 学生控制 | 从头开始，逐步扩展 | 从中间接管，逐步向前扩展 |
| Teacher 参与 | 不参与 rollout | 导航到 checkpoint 状态 |
| 计算效率 | 更高效（短 trajectory） | 较低（需要 teacher 导航） |
| 适用场景 | 小模型、简单任务 | 大模型、困难任务 |

**代码实现**（TCOD_b2f_workflow.py:86-94）：
```python
def _linear_checkpoint_step(self, predefined_actions):
    """计算 checkpoint step：线性衰减策略"""
    reduction = current_step // self.checkpoint_steps
    checkpoint_step = max(0, min(max_expert_actions - reduction, max_expert_actions))
    return checkpoint_step
```

---

### Q1.3: 为什么用 reverse KL 而不是 forward KL？

**问题背景**：

KL 散度有两种形式：
- Forward KL: `KL(π_teacher || π_student)` — mode-covering
- Reverse KL: `KL(π_student || π_teacher)` — mode-seeking

**核心机制**：

OPD/TCOD 使用 reverse KL `KL(π_student || π_teacher)`。这意味着：
- 惩罚 student 在 teacher 低概率 token 上放太多概率
- 鼓励 student 把概率质量集中在 teacher 高概率区域
- 更适合 on-policy 修正 student 当前错误分布

**直觉理解**：
- Forward KL 会让 student "试图覆盖 teacher 的所有模式"，可能导致 student 在自己不擅长的区域也分配概率
- Reverse KL 会让 student "只在自己选择的区域对齐 teacher"，更保守

**局限**：
- 如果 teacher 的关键 token 不在 student top-K 中，`only_stu` 策略看不到
- 可能降低 diversity，student 被拉向 teacher 的单一模式

**面试回答模板**：
> 我用 reverse KL 是因为 OPD 的 on-policy 设定里，重点纠正 student 自己会选择的分布。Reverse KL 更 mode-seeking，惩罚 student 在 teacher 不认可的 token 上放概率。但如果 teacher 很强，关键 token 可能不在 student top-K 里，这时候 `only_stu` 策略会漏掉信号，可以考虑 `only_tch` 或 `union`。

---

### Q1.4: top-K 的 K 怎么选？不同 top-K strategy 的 trade-off？

**问题背景**：

完整词表 KL 对每个 token 位置都要在 10 万+ vocab 上算 log-softmax 和聚合，计算和显存都很贵。

**核心机制**：

top-K 只取 student 或 teacher 分布中概率最高的 K 个 token 近似 KL。论文观察 97-99% 的概率质量集中在少量共享 token 上。

**K 的 trade-off**：

| K 值 | 优点 | 缺点 |
|------|------|------|
| K 小（如 8） | 计算快，显存省 | KL 估计偏差大，可能漏掉 teacher 重要 token |
| K 中（如 16） | 平衡精度和效率 | 默认配置 |
| K 大（如 64） | 更接近 full KL | teacher forward 后 gather/log-softmax 显存压力大 |
| K=0 | 完整 KL | 显存和计算不可接受 |

**Top-K Strategies**：

| Strategy | KL source | Valid tokens | When to use |
|----------|-----------|-------------|-------------|
| `only_stu` | `S_logp - T_on_S` | All student top-K | Default; student-driven |
| `only_tch` | `S_on_T - T_logp` | All teacher top-K | Teacher-driven; useful if teacher is much stronger |
| `intersection` | `S_logp - T_on_S` | Only overlapping tokens | Conservative; only where both agree tokens exist |
| `union` | Concatenate S & T sets, compute KL | All unique tokens | Most complete; higher compute |

**Pitfall**：`intersection` 可以产生非常少的 valid token，在 student 和 teacher 分布完全不重叠的位置，导致 sparse/zero rewards。

**实测配置**（OPD 项目）：
```bash
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
```

---

### Q1.5: `token_reward_direct` 为什么不需要 critic？PPO clipping 在 OPD 下怎么工作？

**问题背景**：

标准 PPO 需要 value function 估计 baseline，用 GAE 计算 advantage。

**核心机制**：

`token_reward_direct` 的实现非常简单：
```python
# core_algos.py:855-880
advantages = token_level_rewards * response_mask
returns = advantages.clone()
```

它不需要 gamma、lambda、value baseline，因为 reward 本身已经是每个 token 的 dense teacher signal。

**PPO clipping 在 top-K 下的工作**：

当 `log_prob_top_k > 0` 时，advantages 是 3D `(batch, seq_len, K)`，不是普通的 2D：
```python
# core_algos.py:1118-1155
if log_prob.dim() == 3 and old_log_prob.dim() == 3:
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses = torch.sum(pg_losses, dim=-1)  # sum across K tokens
```

**关键设计**：
- 对 K 维求和，不是只更新实际采样 token
- 对 student top-K token 分布整体做调整
- 更接近 distribution-level distillation，不是 sample-level imitation

**为什么还需要 PPO clipping？**
- 即使 reward 来自 teacher KL，policy update 仍需要约束
- Token-level reward 很密集，没有约束可能快速把 student 拉向 teacher
- PPO clipping 限制新旧策略变化，保持训练稳定

---

### Q1.6: OPD/TCOD 什么情况下会失败？

**论文指出的条件**：

1. **Student 和 teacher 要有兼容 thinking pattern**
   - 如果一个是 thinking 模型、一个是 non-thinking 模型，token-level KL 会强行对齐不合适的轨迹
   - `enable_thinking` 设置错误会导致格式不兼容

2. **Teacher 要提供 student 没有的新能力**
   - 如果 teacher 只是同分布略强，student 可能学不到有价值的东西
   - 同 family 1.5B 和 7B teacher 可能 distributionally indistinguishable

3. **多轮 Agent 中的 Trajectory-Level KL Instability**
   - 长 horizon 任务中复合误差累积
   - Student 进入 teacher 不熟悉的状态

**实测中的风险**：
- Teacher/student 的 chat template 必须一致
- Teacher temperature 太高/太低会影响信号质量
- Top-K strategy 选择不当会导致稀疏/零 reward

**TCOD 的额外风险**：
- B2F 需要预收集成功轨迹，如果 teacher 成功率低则数据不足
- F2B 限制步数可能导致 student 学不到完整任务
- 线性 pacing 可能不是最优策略

---

## 能力二：实验和方案验证能力

### 面试官想看什么

面试官会从"你做了什么"追问到：
- 你怎么证明训练**真的有效**，不只是跑完了？
- 你怎么证明不是**评估脚本 bug**？
- 你知道哪些 **confounder**？
- 结果和预期不一致时，你怎么排查？

---

### Q2.1: 你怎么证明 OPD 训练是有效的？

**四层证据链**（OPD 项目实测）：

**第一层：训练确实跑完**
```
checkpoint/.../latest_checkpointed_iteration.txt = 1119
```

**第二层：训练信号存在**
```
step 1:   actor/pg_loss=0.2616（非零，dense signal）
step 2:   actor/pg_loss=0.2745
step 10:  actor/pg_loss=0.2738
```

**第三层：分布对齐趋势**
```
step 1:    critic/score/mean=-0.2616, topk/overlap_ratio=0.7223
step 1119: critic/score/mean=-0.2040, topk/overlap_ratio=0.7625
```
- `critic/score/mean` 从 -0.26 到 -0.20（负 KL 向 0 靠近 = 更对齐）
- `topk/overlap_ratio` 从 0.72 到 0.76（student/teacher 高概率 token 集更一致）

**第四层：离线评估结果**
```
MAX_TOKENS=16384:
AIME24: mean_score=28.1%, best_score=63.3%
AIME25: mean_score=24.0%, best_score=46.7%
AMC23: mean_score=72.2%, best_score=97.5%
```

**TCOD 项目的验证**：
```
ALFWorld (Qwen2.5-3B student, GRPO-trained 7B teacher):
- Vanilla OPD: SR=65.72% (seen), 60.45% (unseen)
- TCOD-F2B:    SR=81.43% (seen), 79.19% (unseen)  → +15.71%
- TCOD-B2F:    SR=77.86% (seen), 70.90% (unseen)  → +12.14%
```

**面试回答模板**：
> 我会分训练过程和最终评估两层证明。训练过程中看 `actor/pg_loss` 是否非零、`critic/score/mean` 作为负 KL 是否向 0 靠近、`topk/overlap_ratio` 是否上升。最终评估用 AIME/AMC N=16 规则评分。但我也会控制 confounder，比如 max tokens、format error、JSONL 完整性、mean_score vs best_score。

---

### Q2.2: max_tokens 对评估结果的影响有多大？如何识别这种 confounder？

**问题背景**（OPD 项目实测）：

一开始用 MAX_TOKENS=2048 做离线评测，AIME 指标非常低。

**排查过程**：
1. 查看输出发现大量 response 没有最终 `\boxed{}`
2. Format error 接近 98%（AIME24）、99%（AIME25）
3. 本质是生成被截断，不是模型不会

**结果对比**：

| 任务 | MAX_TOKENS=2048 mean | MAX_TOKENS=16384 mean | format_error 变化 |
|------|---------------------|----------------------|------------------|
| AMC23 | 23.6% | 72.2% | 76% → 18% |
| AIME24 | 1.9% | 28.1% | 98% → 43% |
| AIME25 | 1.3% | 24.0% | 99% → 41% |

**面试回答模板**：
> 一开始用 2048 做离线评测，AIME 指标非常低。但查看输出发现大量 response 没有最终 boxed 答案，本质是生成被截断。把评测 max tokens 调到 16384 后，format error 大幅下降，指标也上来了。这说明评估时 max tokens 是数学推理任务里的重要 confounder。

---

### Q2.3: mean_score 和 best_score 怎么解读？哪个更可靠？

**区别**：
- `mean_score`：所有 rollout 的平均正确率，接近单次采样准确率
- `best_score`：每题 N 个 rollout 至少一个正确的比例，类似 pass@N

**实测数据**：
```
AIME24: mean_score=28.1%, best_score=63.3%
```

**解读**：
- Mean_score 28.1% = 平均每次 rollout 有 28.1% 概率答对
- Best_score 63.3% = 30 道题中约 19 道至少有一个 rollout 答对

**面试要点**：
> 简历和面试里应该重点讲 mean_score，同时说明 best_score 反映多样采样潜力，不能直接当单次 accuracy。Best_score 更适合衡量模型的"能力上限"，mean_score 更适合衡量"稳定输出能力"。

---

### Q2.4: 如何判断评估结果可靠？已知的评估陷阱有哪些？

**三件事**：

1. **输出完整性**
   - AIME24/25 应分别有 30×16=480 行
   - AMC23 应有 40×16=640 行

2. **评分逻辑**
   - `grade.py` 先抽取最后一个 `\boxed{}`
   - 归一化 + sympy 等价检查
   - 抽样人工检查正确和错误案例

3. **Confounder 控制**
   - Max tokens 是否太短
   - Best_score 是否被误当成 mean_score
   - 是否保存了不完整 JSONL

**已知问题**（OPD 项目 gen_vllm.py:185-187）：
- Worker 异常只 print，不退出
- 没有校验 `len(results) == samples * N`

**verl v0.7.0 验证 bug**：
- 会 under-estimate 性能 5-7 个点
- 解决：设置 `trainer.test_freq=-1`，单独评估 checkpoint

---

### Q2.5: 你会怎么做更严谨的 ablation？

**按优先级**：

1. **Baseline 对比**
   - Base model 直接评估
   - GRPO 同步数训练
   - OPD 同步数训练
   - 全部用同一套 eval max tokens、N、temperature、评分器

2. **Top-K ablation**
   - K=0/8/16/32
   - 观察 reward 方差、topk/overlap_ratio、训练吞吐

3. **Top-K strategy ablation**
   - `only_stu` vs `only_tch` vs `union` vs `intersection`

4. **Teacher temperature ablation**
   - T=0.7/1.0/1.3

5. **训练/评估长度口径实验**
   - Train MAX_RESP_LENGTH=2048/4096
   - Eval MAX_TOKENS=2048/8192/16384

**TCOD 特有的 ablation**：
- Pacing strategy: linear vs exponential vs constant
- Growth rate η: {2, 4, 6}
- F2B vs B2F 在不同模型大小上的对比
- Checkpoint_steps 的影响

---

## 能力三：问题定位与排查能力

### 面试官想看什么

面试官会给你故障场景：
- 模型上线后能力突然下降
- 系统上线后突然十分缓慢
- 训练 loss 变成 0 或 NaN
- 评估结果和预期不一致

他想看你是否能**分层定位**，而不是直接调超参。

---

### Q3.1: 通用排查框架（五层）

```
指标是否真实 → 数据是否一致 → 模型/算法信号是否异常 → 系统资源是否异常 → 代码版本/环境是否变化
```

| 层 | 检查项 |
|---|--------|
| 指标层 | 评估脚本、样本数、format error、mean vs best、人工抽样 |
| 数据层 | Prompt 分布、chat template、token length、去重、train/test mismatch |
| 算法层 | Reward、advantage、loss、entropy、KL、top-K overlap |
| 系统层 | GPU 显存、CPU 内存、KV cache、Ray、I/O、worker 异常 |
| 环境层 | Python、CUDA、vLLM、transformers、flash-attn、git diff |

---

### Q3.2: 模型上线后能力突然下降，怎么排查？

**回答模板**：

> 我会先确认下降是真实能力下降还是评估/流量变化。
>
> 第一步看线上请求分布是否变了：prompt 长度、任务类型、语言。
>
> 第二步看输出是否被截断、format 是否变化、拒答率是否上升。
>
> 第三步抽样人工看 case，区分 reasoning 错、格式错、知识错还是安全策略误伤。
>
> 如果是模型版本变更后下降，对比上线前后的 checkpoint、tokenizer、chat template、generation config。OPD 项目里我特别会检查 `enable_thinking`、max tokens、temperature、top_p。
>
> 如果是训练后模型下降，回看训练日志：KL reward 是否异常、top-K overlap 是否下降、entropy 是否塌缩。最后用上一个稳定版本灰度回滚。

**项目中的实际案例**：
> 我复现里遇到过类似"评估能力看起来很差"的情况，AIME 在 2048 max tokens 下非常低。排查后发现不是模型完全不会，而是长推理被截断导致大量 format error。这个经验让我不会第一时间改训练，而是先检查评估和输出完整性。

---

### Q3.3: 系统上线后突然变慢，怎么排查？

**回答模板**：

> 我会先把耗时拆到 generation、teacher forward、log_prob、actor update、I/O 和调度。
>
> 如果是推理慢，先看 vLLM KV cache 是否不足、batching 是否下降、max_model_len 是否过大。
>
> 如果是训练慢，看 FSDP all-gather/reduce-scatter、CPU offload 是否造成 PCIe 瓶颈、Ray 临时目录是否 I/O 压力。
>
> 我项目里为了 3090 跑通打开了 param/optimizer offload，这会省显存但增加 CPU/GPU 数据搬运。所以如果系统变慢，不能只看算法，要看 offload、KV cache、Ray tmp 这类工程配置。

**项目中的具体配置**：
- vLLM `gpu_memory_utilization=0.4`：显存稳定但可能牺牲吞吐
- `save_freq=200`：减少 checkpoint 磁盘压力
- `RAY_TMPDIR=/mnt/sdb2/...`：避免根分区空间不足
- `CUDA_LAUNCH_BLOCKING=1`：debug 用，不适合生产

---

### Q3.4: 训练 loss 长期为 0，怎么判断问题？

**区分算法**：

**GRPO 早期 loss 为 0 可能是正常的**：
- Outcome reward 稀疏
- 如果一组 responses 全错，advantage 没有有效信号
- 实测：GRPO step 1-131 大部分 `actor/pg_loss=0`

**OPD 不应该长期为 0**：
- Token-level KL reward 是 dense 的
- 如果 OPD 的 `actor/pg_loss` 长期为 0，检查：
  1. Teacher reward model 是否启用
  2. `LOG_PROB_TOP_K` 是否大于 0
  3. Student_top_k_ids 是否传到 teacher
  4. Reward 是否被 mask 全部清掉
  5. Response_mask 是否全 0
  6. Top-K strategy 是否产生空 valid token

**TCOD 中的额外检查**：
- Checkpoint strategy 是否正确设置
- Training step 是否正确传递给 workflow
- Teacher 成功轨迹是否正确加载

---

### Q3.5: 评估结果突然变好，你会相信吗？

**回答**：

> 不会立刻相信。我会先查：
> 1. 是否数据泄漏
> 2. 评估 max tokens 是否变了
> 3. N rollouts 是否变了
> 4. Best_score 是否被当成 mean_score
> 5. 是否启用了 model verifier
> 6. 是否保存了不完整 JSONL

**项目中的案例**：
> 2048 到 16384 max tokens 导致结果大幅变化，这是合理的，因为 format error 大幅下降。但如果没有记录评估配置，别人可能误以为模型训练本身大幅提升。

---

### Q3.6: 同样代码换机器跑不起来，怎么定位？

**回答模板**：

> 先固定环境版本：Python、CUDA、torch、transformers、vLLM、flash-attn、verl commit。
>
> 我的项目里裸 `python3` 是 3.5.4，而 `.venv/bin/python` 是 3.12.3，如果脚本直接写 `python3`，换 shell 就会失败。
>
> 然后查 GPU 显存和 CUDA capability。比如 flash-attn wheel 和 CUDA 13.1 不兼容，原始 A800 80GB 配置放到 3090 24GB 肯定 OOM。
>
> 最后查路径硬编码，例如 `RAY_TMPDIR=/mnt/sdb2/...`、模型路径、数据路径。

**项目中的具体问题**：
- Flash-attn fallback API 不完整（只返回 3 个值，标准接口返回 5 个）
- 裸 `python3` 是 3.5.4，项目需要 3.12
- CUDA_HOME 需要设置为 `/usr/local/cuda-13.1`

---

## 能力四：工程落地能力

### 面试官想看什么

面试官想看你是否知道：
- 理论上可行的方案，生产中为什么可能不可行
- 如何部署和服务模型
- 如何保证系统稳定
- 如何做灰度、回滚、监控

---

### Q4.1: 从 A800 80GB 适配到 3090 24GB，你做了什么？

**问题背景**：

原项目默认是 8xA800 80GB，3090 只有 24GB。最大的压力是：
- Rollout 阶段 vLLM 需要模型权重和 KV cache
- Training 阶段 FSDP 需要权重、梯度、优化器、激活值
- OPD 比 GRPO 多一个 teacher reward model forward

**改动清单**（OPD 项目）：

| 配置项 | 原始值 | 3090 值 | 原因 |
|--------|--------|---------|------|
| MODEL_DTYPE | fp32 | bfloat16 | 权重/激活减半 |
| MAX_RESP_LENGTH | 7168 | 2048 | 减少 KV cache |
| N_RESPONSES | 4 | 2 | 减少 rollout 显存 |
| MINI_BATCH_SIZE | 64 | 16 | 匹配更短序列 |
| gpu_memory_utilization | 0.8 | 0.4 | 留出 FSDP 空间 |
| param_offload | False | True | 参数卸载到 CPU |
| optimizer_offload | False | True | 优化器卸载到 CPU |
| save_freq | 20 | 200 | 减少 checkpoint 磁盘占用 |
| flash_attn | flash_attention_2 | sdpa | flash-attn 未安装 |

**实测结果**：
```
每步时间：~47.7 秒
总时长：~13.7 小时（1119 步）
显存使用：21.1 GB allocated / 26.5 GB reserved
CPU 内存：~159-166 GB（用于 offload）
吞吐量：~197 tokens/s
```

---

### Q4.2: vLLM `gpu_memory_utilization` 为什么要降到 0.4？FSDP offload 的 trade-off？

**问题背景**：

这个参数控制 vLLM KV cache 预算，不是模型权重。

**核心机制**：

原始 A800 80GB 可以给 vLLM 很大空间（0.8 = 64GB），但 3090 只有 24GB。OPD 同时还有 FSDP actor、ref、teacher reward model 和激活值压力。

**实测显存分配**：
```
Rollout 阶段：~15 GB（vLLM 模型权重 + KV cache）
Training 阶段：~18.6 GB（FSDP 模型 + 优化器 + 激活值）
Reserved：~27.6 GB（PyTorch 内存池）
```

所以 `gpu_memory_utilization=0.4`（约 9.6GB 给 KV cache），给 FSDP 和 reward model 留空间。

**FSDP offload 的 trade-off**：
- 优势：用 CPU 内存换 GPU 显存，机器 CPU 内存有 200GB+，可行
- 代价：PCIe 通信开销，训练速度可能降低 10-20%，CPU 内存使用 ~159-166 GB

---

### Q4.3: 如果要把 OPD 用到生产，你会怎么做？

**离线训练 pipeline**：
1. 收集真实业务 prompt，脱敏、去重、分桶
2. Student 当前线上模型生成多样 response
3. Teacher 对相同 prompt 给 token-level feedback
4. 混合业务 outcome reward、安全规则、格式规则
5. 训练候选 student

**评估**：
- 离线 benchmark 只是第一层
- 业务集评测、人工抽样、安全红队、延迟和成本评估

**上线**：
- 灰度发布，小流量 A/B
- 监控成功率、拒答率、投诉率、延迟、成本、format compliance
- 保留旧模型和配置 bundle，异常时回滚

**TCOD 生产落地的额外考虑**：
- 多轮 Agent 的环境部署（ALFWorld/WebShop/ScienceWorld）
- Teacher 模型的实时推理成本
- 成功轨迹的收集和更新策略
- Curriculum 策略的在线调整

---

### Q4.4: 模型怎么部署？上线后怎么保证系统稳定？

**部署流程**：
> 训练完成后需要先把 FSDP checkpoint merge 成 HuggingFace 格式，确认 tokenizer、generation_config 和 chat template 一致。
>
> 部署时可以用 vLLM/TGI/SGLang 之类的推理框架，根据业务吞吐和 latency 选择 tensor parallel、batching、KV cache 预算。
>
> 不能只部署权重，不记录 generation config，因为 temperature、top_p、max_tokens、stop tokens 都会影响行为。

**四类保障**：

1. **版本稳定**
   - 模型权重版本、tokenizer 版本
   - Prompt template 版本、decoding config 版本

2. **服务稳定**
   - GPU 利用率、显存、KV cache hit/eviction
   - QPS、P50/P95/P99 latency
   - Error rate、timeout、OOM

3. **效果稳定**
   - 在线业务指标、人工抽检
   - 安全/合规触发率、prompt drift 监控

4. **回滚稳定**
   - 保留上一稳定模型
   - 配置可回滚
   - 数据和训练版本可追溯

---

### Q4.5: flash-attn fallback 问题是什么？评测脚本有什么静默失败风险？

**flash-attn 问题**：

远程环境 CUDA 13.1 和 flash-attn 预编译 wheel 不兼容。我加了一个纯 PyTorch fallback，让 FSDP + sdpa 路径跑通。

审查发现的问题：
- `unpad_input` 只返回 3 个值，标准接口返回 5 个值
- 当前 FSDP 路径靠 `*_` 能跑，但 Megatron 路径会失败
- 这是临时 workaround，不是完整修复

**面试要点**：
> 我不会说"我解决了 flash-attn 问题"，而会说"我做了一个让当前 FSDP+sdpa 路径跑通的 workaround，但审查发现 fallback API 不完整，正式修复应对齐 transformers/flash-attn 的返回值"。

**评测脚本静默失败风险**：

问题位置：`gen_vllm.py:185-187`
- Worker 捕获异常后只 print，然后返回已有结果
- 主进程只要 `all_results` 非空就保存 JSONL
- 没有校验行数是否等于 `num_samples * N`

修复建议：
```python
# 保存前断言
assert len(all_results) == len(samples) * N
for example_id, count in rollout_counts.items():
    assert count == N, f"Example {example_id} has {count} rollouts, expected {N}"
```

---

## 能力五：业务与实际场景理解

### 面试官想看什么

面试官会问：
- 这个方案适合什么样的**场景**？
- 用户更关心的是什么？
- 上线成本有多高？
- 如果资源有限，我们应该首先优化哪些部分？

---

### Q5.1: OPD/TCOD 适合什么场景？不适合什么场景？

**OPD 适合**：

1. **有强 teacher、需要部署小 student**
   - 大模型 API 很贵，线上要小模型低成本服务
   - OPD 可以把 teacher 在真实 student 状态上的行为迁移给 student

2. **长推理任务**
   - 数学、代码、复杂工具调用
   - Outcome reward 稀疏，token-level teacher signal 有价值

3. **有明确业务 prompt 分布**
   - 客服、教育、代码助手、企业知识库问答
   - 可以让 student 在真实业务 prompt 上 rollout

**TCOD 额外适合**：

4. **多轮交互式 Agent 任务**
   - Web 导航、家居控制、科学实验
   - 需要长 horizon 决策，复合误差是核心挑战

**不适合**：

1. Teacher 不可靠或成本极高
2. 业务目标不是 teacher likelihood 能表达的
3. 安全合规要求非常强但没有额外 safety evaluator
4. 线上延迟极其敏感，不能承受长推理
5. 简单任务，SFT 就能解决

---

### Q5.2: 用户更关心什么？上线成本多高？

**用户关心的**：

> 用户通常不关心模型是否和 teacher 分布接近，而关心：
> - 任务是否完成
> - 回答是否可靠
> - 速度是否快
> - 成本是否低
> - 安全是否合规
>
> OPD 优化的是 distribution alignment，它和用户目标相关但不完全一致。所以生产里我不会只看 KL reward，而会用业务指标闭环：问题解决率、正确率、延迟、成本。

**成本分析**：

**训练成本**：
- Student rollout + teacher forward + FSDP 训练
- OPD 比 GRPO 多 teacher token-level forward
- 比 SFT 多 on-policy rollout
- TCOD 比标准 OPD 多 teacher 导航（B2F）或环境重置（F2B）

**服务成本**：
- 取决于最终部署的 student
- 如果 OPD 能把强 teacher 的能力迁移到小 student，线上服务成本可能下降
- 但如果训练成本太高、业务量太小，ROI 不一定划算

---

### Q5.3: 如果资源有限，先优化哪里？

**按 ROI 排**：

1. **先优化评估和数据**
   - 找高频失败 case
   - 修复 format/template/max_tokens 这种低成本问题
   - 避免训练方向错

2. **再优化 decoding 和服务配置**
   - Max_tokens、temperature、stop tokens、batching
   - 很多线上问题不需要重训

3. **再做 targeted fine-tuning**
   - 只针对高价值业务分布做 SFT/OPD
   - 不全量大规模训练

4. **最后扩大模型或训练规模**
   - 更大 teacher、更长 response、更多 rollouts
   - 这是最贵的

**项目例子**：
> 我评估中把 max tokens 从 2048 调到 16384 后，指标变化很大。这说明资源有限时，先排查评估和生成配置可能比盲目加训练更有价值。

---

### Q5.4: 具体业务场景怎么用 OPD/TCOD？

**客服场景**：
> 我会先收集脱敏后的真实客服 query，按问题类型分桶。让当前小模型生成回答，再让强 teacher 或人工规则对回答进行指导。
>
> OPD 可以在小模型真实会犯错的状态上做蒸馏，同时我会加入业务规则 reward：是否回答完整、是否引用正确知识库、是否触碰合规风险。
>
> 评估不只看 benchmark，而看人工转接率、用户满意度、合规违规率和延迟成本。上线时灰度发布，并保留旧模型回滚。

**代码助手场景**：
> 代码任务里 teacher 可以是更强的代码模型，student rollout 生成代码，teacher 提供 token-level 分布指导。但代码 correctness 不能只靠 teacher likelihood，所以还要结合编译结果、单元测试、静态检查作为 outcome reward。
>
> OPD 提供 dense signal，单测提供最终正确性约束。生产指标可以是 test pass rate、开发者接受率、生成延迟。

**多轮 Agent 场景**（TCOD）：
> 在 Web 导航或家居控制场景中，student 需要学习长 horizon 的决策序列。TCOD 通过 temporal curriculum 让 student 先学习短序列，逐步扩展到完整任务。
>
> 关键是平衡 teacher 指导和 student 自主探索：F2B 适合初期建立基础能力，B2F 适合后期突破困难任务。

---

### Q5.5: 怎么证明方案真的给公司带来价值？

**回答模板**：

> 我会设定业务指标，而不是只看 benchmark。
>
> - 客服场景：人工转接率下降、首次解决率提升、用户满意度提升
> - 代码助手：编译通过率、单测通过率、开发者接受率
> - 教育数学：解题正确率、讲解可读性
> - Web 导航：任务完成率、平均步数、成功率
>
> 然后用 A/B test 或灰度实验比较旧模型和 OPD student。如果收益覆盖训练和推理成本，才算有业务价值。

---

## 综合模拟问答

### 场景 1：面试官问"你这个方法为什么这样设计？"

**OPD 回答**：
> 设计来自两个痛点：普通蒸馏有 distribution mismatch，GRPO 早期 reward 稀疏。OPD 让 student on-policy rollout，再用 teacher 在 student-visited states 上提供 token-level KL reward。top-K 是为了降低 full-vocab KL 成本，PPO clipping 是为了稳定 policy update。
>
> 局限是 teacher 成本高、格式要兼容、KL 不等于最终业务目标，所以可改进方向是 OPD+outcome reward、teacher selection、top-K strategy ablation。

**TCOD 回答**：
> 设计来自多轮 Agent 中 OPD 的 Trajectory-Level KL Instability。复合误差导致 KL 随 trajectory 深度累积，student 进入 teacher 不熟悉的状态。
>
> TCOD 用 temporal curriculum 控制 trajectory 深度，从短到长逐步扩展。F2B 从头开始建立基础，B2F 利用 teacher scaffolding 在困难任务上获得好初始化。
>
> 局限是需要 teacher 预收集成功轨迹（B2F）、线性 pacing 可能不是最优、只在文本环境验证。改进方向是自适应 pacing、多模态环境扩展、与 RL 方法结合。

---

### 场景 2：面试官问"怎么证明有效？"

> 我会看训练和评估两层。训练上，OPD 的 `actor/pg_loss` early steps 非零，`critic/score/mean` 作为负 KL 从约 -0.26 到 -0.204，`topk/overlap_ratio` 从约 0.72 到 0.76。
>
> 评估上，AIME/AMC N=16 规则评分显示长生成下指标提升。但我也会控制 confounder，比如 max tokens、format error、JSONL 完整性、mean_score vs best_score。更严谨还要做同口径 baseline 和 ablation。
>
> TCOD 方面，ALFWorld 上 TCOD-F2B 比标准 OPD 提升 15.71 点，KL 曲线更稳定，训练时间减少 32%。

---

### 场景 3：面试官问"模型突然变差怎么办？"

> 我会先判断是真能力下降还是评估/线上分布变化。检查 prompt 分布、max_tokens、chat template、format error、generation config；再看训练信号如 reward、KL、entropy、top-K overlap；再看系统环境和版本。
>
> 项目里我就遇到过 2048 评估让 AIME 看起来很差，排查发现是截断和 format error，而不是先去改模型。

---

### 场景 4：面试官问"生产怎么落地？"

> 我会离线收集真实业务 prompt，让 student rollout，用 teacher 和业务 evaluator 提供反馈，训练候选 student；再用业务指标、安全指标、人工抽检和延迟成本做离线评估。
>
> 上线采用灰度和 A/B test，监控成功率、拒答率、合规、P95 延迟和成本，保留旧模型和配置 bundle 回滚。
>
> 对于 TCOD 多轮 Agent，还需要考虑环境部署、teacher 实时推理成本、成功轨迹收集策略。

---

### 场景 5：面试官问"业务价值是什么？"

> 业务价值不是 OPD/TCOD 本身，而是用强 teacher 提升小 student，降低线上成本或提升任务成功率。它适合高频、长推理、teacher 调用贵、又有明确业务 prompt 分布的场景。
>
> 用户关心的是正确、快、安全、便宜；所以 OPD 必须和业务 KPI 对齐，不能只看 teacher KL。

---

## 面试中要避免的回答

### ❌ 只讲概念

弱：> OPD 就是 on-policy distillation，用 KL 蒸馏 teacher。

强：> OPD 解决 off-policy KD 的 distribution mismatch 和 GRPO outcome reward 稀疏问题；它在 student rollout 的状态上用 teacher token distribution 做 dense reward，但依赖 teacher 质量和格式兼容，成本也更高。

### ❌ 只讲结果

弱：> 我最后 AIME24 到 28.1%。

强：> AIME24 28.1% 是在 N=16、MAX_TOKENS=16384、规则评分下的 mean_score；2048 时因为截断 format error 很高，只有 1.9%。所以这个结果要和评估长度一起解释。

### ❌ 把问题说成已经彻底解决

弱：> 我解决了 flash-attn 问题。

强：> 我做了一个让当前 FSDP+sdpa 路径跑通的 workaround，但审查发现 fallback API 不完整，Megatron 路径会失败，所以正式修复应对齐 transformers/flash-attn 的返回值。

### ❌ 把实验说成生产价值

弱：> 实验有效，所以上线一定有收益。

强：> 实验有效只说明在 benchmark 上有潜力。生产要看业务 KPI、延迟、成本、安全和用户满意度，还需要灰度和 A/B test。

---

## 五类能力速查表

| 能力 | 面试官问题 | 你的核心抓手 |
|------|----------|-------------|
| 底层原理 | "为什么这么设计？" | OPD 解决 distribution mismatch + sparse reward；TCOD 解决 trajectory-level KL instability；top-K 降成本；负 KL 做 dense reward |
| 实验验证 | "怎么证明有效？" | 训练信号、top-K overlap、评估结果、baseline、ablation、confounder 控制 |
| 问题定位 | "结果不符合预期怎么办？" | 指标层→数据层→算法层→系统层→环境层分层排查 |
| 工程落地 | "怎么上线？" | 训练 pipeline、部署服务、监控、灰度、回滚、成本控制 |
| 业务理解 | "有什么价值？" | 强 teacher → 小 student 降本提质；业务 KPI 而非 KL 才是最终目标 |

---

## 最后背诵版（5 分钟）

**OPD 项目**：
> 我这个项目复现的是 Rethinking OPD。它解决的核心问题是：传统蒸馏在固定 teacher 数据上训练，和 student 部署时自己的分布不一致；而 GRPO 这类 outcome-level RL 在数学推理早期 reward 很稀疏。
>
> OPD 让 student 自己 rollout，然后在 student 访问到的 token 状态上，用 teacher 分布计算 top-K 近似 KL，把负 KL 当作 dense token reward，再用 PPO-style update 训练 student。
>
> 我在 8x3090 24GB 上把原 A800 配置缩小，使用 bf16、FSDP offload、gradient checkpointing，降低 response length、rollout 数和 vLLM KV cache 显存，完成 1119 step 1.5B OPD 训练。日志里 OPD early steps `actor/pg_loss` 非零，`topk/overlap_ratio` 约 0.72 到 0.76，说明蒸馏信号确实存在。
>
> 评估上我发现 max tokens 是关键 confounder，2048 会严重截断数学推理导致 format error，16384 离线评测下 AIME24 mean 28.1%、AIME25 24.0%、AMC23 72.2%。

**TCOD 项目**：
> TCOD 是将 OPD 扩展到多轮 Agent 任务的工作。核心发现是标准 OPD 在多轮交互中会出现 Trajectory-Level KL Instability，复合误差导致 KL 随 trajectory 深度累积。
>
> 解决方案是 Temporal Curriculum，控制暴露给学生的 trajectory 深度。F2B 从头开始逐步扩展，B2F 从末尾开始 teacher scaffolding。
>
> 在 ALFWorld、WebShop、ScienceWorld 三个 benchmark 上，TCOD 比标准 OPD 提升最高 18 点，训练时间减少 32%，甚至能超越 teacher 自身表现。

**工程与业务**：
> 从工程角度，我处理了 3090 显存限制、flash-attn 兼容性、评估 confounder 等实际问题。从业务角度，OPD/TCOD 的价值是用强 teacher 指导小 student，在真实业务 prompt 分布上降低部署成本或提升任务成功率；但生产里还要考虑 teacher 成本、监控、灰度、回滚、安全和业务 KPI，不能只看 KL 或 benchmark。

---

*文档版本：v1.0*  
*最后更新：2026-06-26*  
*基于 TCOD 与 OPD 两个项目的深度实践*
