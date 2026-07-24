# MegaPersona-Evolve 周报 2026-07-14  

## 1. 本周工作概述

本周主要围绕 Genome v3 的搜索机制展开，重点不是继续单纯扩大实验规模，而是判断当前进化机制是否真正带来稳定收益。

本周完成了三类工作：

| 工作 | 目的 | 当前结论 |
|---|---|---|
| 引入 hybrid MCTS 搜索 | 让算子选择从随机采样变为基于历史 reward 的搜索 | MCTS 对 validation search 有帮助，但泛化仍不稳定 |
| 增加 strict metrics | 避免旧 alignment / consistency 指标初始过高、难以解释 | strict error、axis target MAE、shadow MAE 更适合作论文解释 |
| 分析多轮实验曲线 | 判断进化是否能持续提升 | 多数实验在若干代后进入平台期，需要重新设计逃离平台期机制 |

整体判断是：Genome v3 进化不是无效，但当前系统更像是在强 baseline 上做局部搜索。它能较快找到一个较好候选，但后续容易平台化，并且在强化一致性后出现行为多样性下降的风险。

## 2. MCTS 搜索对进化的影响

本周对随机 v3 算子池和 hybrid MCTS 做了对比。
Hybrid MCTS 用来改变进化算子的选择策略。

原本的进化算子池依赖随机采样，虽然可以探索不同算子，但很难利用历史上“哪些算子组合更有效”的经验。加入 MCTS 的主要目的，是把算子选择从随机采样改成带历史反馈的搜索：让系统根据过去候选的 reward，动态决定下一步更应该尝试哪些算子或算子路径。

因此，MCTS 在当前阶段的定位不是替代 fitness，也不是保证最终结果一定更好，而是提供一种更可解释的算子搜索机制：

```text
random operator sampling -> history-aware operator search
```

### 实验结果
MCTS 的正面信号是：它确实能改变搜索路径，不是简单复刻随机搜索。例如 seed17 中，MCTS 找到的 best candidate 来自 `op16_v3_blueprint_binding`，而随机算子池中更明显的信号来自 `op18`、`op17` 和 `op19`。

但实验结果并没有形成明显、稳定的整体提升。不同 seed 和不同 reward 设计下，最优算子并不稳定；当 reward 更偏 strict consistency 时，搜索会倾向于更规整、更一致的候选，但行为覆盖和行为多样性可能下降。

因此，后续关键问题不是简单加大 MCTS 搜索深度，而是重新设计 MCTS reward：

> Q: MCTS reward 是否应该从单一 fitness 改成多目标结构？

| 层级 | 作用 |
|---|---|
| 硬约束 | schema validity 和 generation rate 不过关时，直接给低 reward |
| 软保护项 | coverage / diversity 明显下降时降低搜索优先级，但不直接否决候选 |
| 优化项 | 在满足前两者后，再奖励 fitness、shadow MAE、axis target MAE、strict consistency 的改善 |

## 3. 平台期现象

几次实验都观察到类似现象：进化曲线出现一次提升，然后长期不再刷新 global best。

| 实验 | 平台期表现 | 说明 |
|---|---|---|
| n12 长程实验 | best 在 gen5 出现，之后到 gen29 未刷新 | 说明继续增加 generation 不一定带来持续收益 |
| strict-aware MCTS n8/g15 | gen1 后短期平台，gen9 再提升，之后到 gen15 未刷新 | MCTS 能延后一次突破，但仍会再次平台化 |
| diversity-guard MCTS n8/g15 | gen9 出现 best，gen10-15 未刷新 | guard 没有解决长期平台化 |

这和最初预期不同。原本预期是：

```text
baseline 较差 -> gen10 明显提升 -> gen20 继续提升 -> gen30 开始放缓
```

平台期的可能原因包括：

1. baseline 已经由 schema、blueprint、slot sampling 和 validator 强力托底；
2. 当前算子多数是在局部修补 genome，而不是创造新的搜索方向；
3. 虽然系统已经记录 strict-best、coverage-best、diversity-best 等 multi-objective elites，但主曲线和最终 best 仍主要由 global fitness 决定；
4. MCTS reward 如果过于惩罚，会导致所有算子的平均 reward 偏负，搜索缺少明确正反馈。

## 4. 指标细化与优化

本周对指标进行了细化，主要原因是旧的 alignment 和 consistency 指标在 baseline 阶段已经偏高，容易出现“看起来已经很好，但不知道进化到底改进了哪里”的问题。

因此，指标优化的目标不是增加复杂度，而是把总 fitness 拆成更可解释的几个诊断维度：

```text
是否有效生成 -> schema / generation rate
是否贴近目标 -> axis target MAE
是否行为对齐 -> shadow MAE
是否跨字段自洽 -> strict consistency error
是否保持多样 -> coverage / diversity
```

strict-aware MCTS 的结果显示：

```text
strict consistency error: 0.0666 -> 0.0496  明显改善
axis target MAE:          0.0933 -> 0.0709  明显改善
validation coverage:      0.3990 -> 0.3910  略降
validation diversity:     0.3399 -> 0.3078  下降
```
| 指标 | 回答的问题 | 方向 |
|---|---|---|
| strict consistency error | persona 是否跨字段自洽，且没有明显机制矛盾 | 越低越好 |
| axis target MAE | 生成结果是否真的贴近目标 slot 的人格轴设定 | 越低越好 |
| validation coverage | shadow survey 行为响应是否覆盖足够多的行为区域 | 越高越好 |
| validation diversity | 不同 persona 的行为模式是否足够分散，而不是趋同 | 越高越好 |

如果只看总 fitness，容易把不同目标的变化混在一起。进化并不要求每个细分指标都同步变好；不同指标代表不同优化方向，彼此之间可能存在张力。
例如本次 strict-aware MCTS 的总分提高，主要来自一致性和轴贴合改善，但行为覆盖和多样性下降。细分指标的作用不是证明所有指标都应该提升，而是帮助判断当前搜索收益来自哪里，以及搜索方向是否出现偏置。

## 5. 下一步需要设计的问题

下一阶段不应只是继续堆 generation，而应该重新设计平台期后的搜索机制。

这里不是重新增加 elite 存储；相关 multi-objective elite archive 已经完成。下一步更关键的是：当 global best 多代不刷新时，如何使用这些已经记录的 elite 来切换父代来源、调整算子组合，并避免搜索继续围绕单一高分模板收缩。

建议重点设计：

| 方向 | 说明 |
|---|---|
| 平台期触发规则 | 当 global best 连续多代不刷新时，启动逃逸策略 |
| elite 使用策略 | 从 strict-best、coverage-best、diversity-best 中选择父代，而不是只沿 global-best 局部变异 |
| 算子组合策略 | 平台期后优先组合 coverage/diversity 算子与 consistency 算子 |
| 软选择保护 | 当 strict consistency 提升伴随 coverage/diversity 明显下降时，降低该路径后续采样优先级，但不直接丢弃候选 |

后续实验应重点验证：新的 reward 和平台期逃逸策略，能否在保持 strict consistency 与 shadow MAE 改善的同时，避免 persona 行为模式趋同。
