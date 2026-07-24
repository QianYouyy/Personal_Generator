# MegaPersona-Evolve 周报 2026-07-20

## 1. 本周工作

本周主要围绕 MCTS reward 机制继续推进。上周实验显示，MCTS 能改变算子搜索路径，但在平台期后容易出现两类问题：一是 reward 信号过弱或偏负，导致 MCTS 接近随机采样；二是 strict consistency 改善时，coverage / diversity 可能下降。

上周的主要现象是：多个 Genome v3 实验在前几代出现一次提升后，很快进入平台期。继续增加 generation 并没有稳定带来后续提升，说明问题不只是“跑得不够久”。进一步分析后，平台期更可能来自两类机制问题：

| 问题 | 表现 | 本周对应调整 |
|---|---|---|
| MCTS reward 信号弱 | global best 不动时，算子很难获得正反馈，搜索接近随机或整体 reward 偏负 | 增加 structured reward 和 reward 标准化 |
| 变异预算空转 | 部分 child 没有产生实际 genome 改动，或只是重复已有表型 | 增加 mutation audit、phenotype 去重和 no-op 处理 |

因此，本周的调整思路是：

```text
1. 改善 MCTS reward，让平台期的局部指标进步也能被记录；
2. review变异过程，减少 no-op 和重复表型对搜索信号的污染。
```

## 2. MCTS Reward

结构化的 Reward 的设计分三层：

| 层级 | 内容 |
|---|---|
| 硬门槛 | 评估失败或 `global_best <= 0` 时给低 reward |
| 软保护 | coverage / diversity 相对下降超过 2% 才有界惩罚，不把小幅波动当作失败 |
| 优化项 | 对 `global_best`、`shadow_mae_elite`、`axis_target_elite`、`strict_consistency_elite`、`coverage_elite`、`diversity_elite` 等 elite 指标做相对改进奖励 |

这样做的意义是：平台期不一定代表所有变异都无效，也可能只是综合 fitness 没有立刻刷新。structured reward 让 MCTS 能看见这些非 global-best 的局部改进，从而避免在平台期退化成近随机采样。

此外，policy 会跟踪各 elite 指标的跨岛历史最优。即使 global best 没有刷新，只要 child 刷新了其他长期停滞的 elite 指标，也会获得正向进度奖励。

## 3. Structured MCTS 实验结果

baseline 到 best 的主要变化：

| 指标 | baseline | best | 变化 |
|---|---:|---:|---:|
| validation fitness | 0.2277 | 0.2802 | +23.0% |
| research score v2 | 0.2093 | 0.2610 | +24.7% |
| validation coverage | 0.3240 | 0.3910 | +20.7% |
| validation diversity | 0.2552 | 0.3281 | +28.6% |
| schema fitness | 0.3972 | 0.4544 | +14.4% |
| strict consistency score | 0.9339 | 0.9498 | +1.7% |

best candidate：

```text
candidate = openevolve_000039_d69fb0fb025e
generation = gen7
operator = op17_v3_axis_coverage_grid
validation fitness = 0.280176
```

但它没有完全解决平台期。曲线在 gen7 刷新 best 后，gen8 到 gen15 仍未继续刷新 global best。

## 4. Deficit 动态权重实验结果
动态权重：structured reward 内部的指标权重调度方式。
structured reward 负责定义“奖励由哪些部分组成”，
deficit 负责决定优化项里“每个 elite 指标当前占多少权重”。

具体做法是：每个指标的权重由两部分决定：

```text
动态权重 = 基础权重 × (该指标历史最优值 - 当前 parent 值 + 0.02)
```

然后再归一化，使所有指标权重之和为 1。这样，已经接近历史最优的方向会自动降权，仍然落后的方向会自动升权。

结果显示，deficit 确实延后了平台期：

| 实验 | best generation | validation fitness |
|---|---:|---:|
| structured fixed | gen7 | 0.280176 |
| structured deficit | gen12 | 0.283855 |

deficit 的 global best 曲线：

```text
gen1  = 0.258560
gen5  = 0.261846
gen7  = 0.265129
gen10 = 0.274561
gen12 = 0.283855
gen13-15 = 平台
```

## 5. Deficit 的问题

虽然 deficit 提高了 validation fitness，并把 best 从 gen7 推迟到 gen12，但封存测试集表现不如 fixed structured。

| 指标 | fixed structured | deficit structured | 变化 |
|---|---:|---:|---:|
| validation fitness | 0.2802 | 0.2839 | +0.0037 |
| test coverage | 0.3940 | 0.3450 | -0.0490 |
| test balanced diversity | 0.3396 | 0.2883 | -0.0513 |
| test behavior avg dist | 0.4671 | 0.3842 | -0.0829 |
| test shadow MAE | 0.2081 | 0.2145 | +0.0064 |
| test strict error | 0.0502 | 0.0622 | +0.0121 |

这说明 deficit 并不是全面更好。它解决了一部分平台期信用分配问题，但没有稳定改善封存测试集上的多样性和 strict consistency。

一个重要观察是：候选池中已经出现了 coverage / diversity 更好的候选，但 global fitness 没有选择它们：

| candidate | coverage | diversity | fitness |
|---|---:|---:|---:|
| `openevolve_000072_cd48a35232ee` | 0.437 | 0.371 | 0.264 |
| `openevolve_000064_c2393b5f689c` | 0.444 | 0.348 | 0.264 |
| `openevolve_000088_4549a1c3cb1c` | 0.412 | 0.335 | 0.276 |
| final global best `openevolve_000070_268b57d6928b` | 0.387 | 0.306 | 0.284 |

因此，当前问题已经不只是 MCTS reward。MCTS 能产生不同目标上的候选，但最终 selection 仍然偏向 global fitness。

## 6. 变异审计与 no-op 诊断

在 structured reward 和 deficit 动态权重之后，进一步加入了变异审计，用来检查平台期是否还受到“无效变异 / 重复表型”的影响。

主要发现：

| 项目 | 结果 | 说明 |
|---|---:|---|
| candidate 总数 | 65 | g8 × 8 islands，加上初始候选 |
| 真实评估数 | 53 | 实际调用 persona / shadow evaluation 的候选 |
| phenotype cache hit | 12 | 约 18.5% 候选为重复表型 |
| undeclared edits | 0 | LLM mutator 基本按要求声明修改字段 |
| actual empty | 13 | 一部分变异没有产生实际 genome 字段变化 |

这说明当前平台期不仅是 reward 设计问题，还存在一部分“变异预算空转”：有些 child 只改变了谱系 metadata，或声明了修改但 normalize 后没有形成新的有效表型。

