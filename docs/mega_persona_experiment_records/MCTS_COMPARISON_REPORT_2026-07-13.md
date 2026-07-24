日期：2026-07-13

## 1. 报告目的

本报告比较原始随机 v3 算子池与新增的 hybrid MCTS 算子选择策略。

本次对比的目标比较明确：在不改变生成器、评估器、fitness 公式和 sealed-test 协议的前提下，检查 MCTS 是否能改善 Genome v3 算子池中的搜索行为。

## 2. 对比实验

| 标记 | 实验目录 | 状态 |
|---|---|---|
| 随机 v3 算子池 | `data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627` | 完整跑到 gen10；之后继续到 gen18 checkpoint，后续受网络异常影响 |
| MCTS seed17 | `data/results/mega_persona_v3_mcts_single_call_deepseek_n8_g10_20260712` | 完整跑到 gen20 |
| MCTS seed23 | `data/results/mega_persona_v3_mcts_single_call_deepseek_n8_g20_seed23_20260713` | 因 API / 网络速率问题在 gen11 checkpoint 中断 |

三组实验的共同配置：

```text
provider = deepseek
persona_pipeline = single_call
n = 8
operator_family = v3
shadow_surveys = 4
validation_shadow_surveys = 2
test_shadow_surveys = 2
items_per_shadow_survey = 6
```

本报告中的算子归因使用 `genome.openevolve_mutation.operator_id`。该字段记录 OpenEvolve mutator 为子代实际选择的算子。部分旧的随机算子池候选仍保留过时的 `genome.last_evolution_operator` 字段，下面的算子统计不使用该字段。

## 3. 主要结果

MCTS seed17 gen20 在 validation fitness 上超过了之前随机 v3 算子池的最佳 checkpoint。

| 结果 | 最佳候选 | 候选代数 | 最佳算子 | Fitness |
|---|---|---:|---|---:|
| 随机 gen10 final | `openevolve_000063_8098c1dac245` | 8 | `op17_v3_axis_coverage_grid` | 0.272785 |
| 随机 gen18 checkpoint | `openevolve_000104_480fcc970873` | 13 | `op18_v3_behavior_alignment_probes` | 0.278084 |
| MCTS seed17 gen20 final | `openevolve_000126_5233f4945091` | 16 | `op16_v3_blueprint_binding` | 0.286166 |
| MCTS seed23 gen11 checkpoint | `openevolve_000073_9bb1698bc6a2` | 9 | `op20_v3_realistic_novelty` | 0.262185 |

与之前随机算子池的最佳 checkpoint 相比，MCTS seed17 的 validation fitness 提升为：

```text
0.286166 - 0.278084 = +0.008082
相对提升约 +2.9%
```

## 4. 指标对比

主要比较对象：MCTS seed17 gen20 与随机 v3 算子池 gen18 checkpoint。

| 指标 | 随机 gen18 | MCTS seed17 gen20 | 变化 |
|---|---:|---:|---:|
| fitness | 0.278084 | 0.286166 | +0.008082 |
| schema_fitness | 0.444982 | 0.457916 | +0.012934 |
| validity_rate | 1.000000 | 1.000000 | +0.000000 |
| internal_consistency | 0.936747 | 0.955161 | +0.018414 |
| internal_consistency_min | 0.858303 | 0.904341 | +0.046037 |
| axis_alignment | 0.911514 | 0.943980 | +0.032467 |
| validation_behavior_coverage | 0.401000 | 0.407000 | +0.006000 |
| validation_shadow_alignment | 0.842524 | 0.817379 | -0.025145 |
| validation_behavior_balanced_diversity | 0.328854 | 0.303466 | -0.025388 |
| validation_behavior_avg_dist | 0.484054 | 0.382886 | -0.101168 |
| slot_coverage | 0.473000 | 0.473000 | +0.000000 |
| near_duplicate_rate | 0.000000 | 0.000000 | +0.000000 |

解释：

- MCTS 改善了结构侧指标，包括 schema、一致性、最差一致性和轴对齐。
- MCTS 对 validation behavior coverage 有轻微提升。
- 相比随机 gen18 best，MCTS 的 validation shadow alignment 和行为分散度下降。
- 因此，这个提升不是纯粹的行为对齐提升，更像是 blueprint / 结构 / 轴绑定方向的收益。

## 5. Sealed Test 表现

旧的随机实验只记录了部分 sealed-test 行为指标，因此 sealed-test 对比是不完整的。

| 测试指标 | 随机 gen10 final | MCTS seed17 gen20 | 变化 |
|---|---:|---:|---:|
| test_behavior_coverage | 0.363000 | 0.358000 | -0.005000 |
| test_shadow_alignment | 0.821519 | 0.791679 | -0.029840 |
| test_behavior_avg_dist | 0.371456 | 0.464807 | +0.093350 |

解释：

- MCTS seed17 提高了 sealed-test 行为距离。
- MCTS 没有提高 sealed-test behavior coverage。
- MCTS 降低了 sealed-test shadow alignment。
- 因此，当前证据支持 MCTS 是更好的 validation search policy，但还不能证明它带来了更好的 sealed-test 泛化。

## 6. 算子层面发现

### 6.1 随机 v3 算子池

在原始随机 v3 算子池实验中，过滤掉 0 fitness 的失败候选后，平均表现最强的算子是 `op18`。

| 算子 | 有效样本数 | 平均 fitness | 相对 baseline 平均提升 | 最高 fitness |
|---|---:|---:|---:|---:|
| `op16_v3_blueprint_binding` | 23 | 0.229416 | +0.014370 | 0.275309 |
| `op17_v3_axis_coverage_grid` | 21 | 0.238098 | +0.023052 | 0.276501 |
| `op18_v3_behavior_alignment_probes` | 13 | 0.247097 | +0.032051 | 0.278084 |
| `op19_v3_cross_field_coherence` | 22 | 0.236292 | +0.021246 | 0.277614 |
| `op20_v3_realistic_novelty` | 29 | 0.226847 | +0.011801 | 0.271894 |
| `op21_v3_schema_precision` | 20 | 0.229410 | +0.014364 | 0.276416 |

这说明原始随机实验并没有显示 `op16` 是平均最强的算子。随机算子池中更明显的信号来自 `op18`、`op17` 和 `op19`。

### 6.2 MCTS seed17

在完整的 MCTS seed17 gen20 实验中，最终最佳候选来自 `op16_v3_blueprint_binding`。

MCTS 根节点算子统计：

| 算子 | 访问次数 | 平均 reward |
|---|---:|---:|
| `op16_v3_blueprint_binding` | 17 | +0.014052 |
| `op20_v3_realistic_novelty` | 29 | -0.007493 |
| `op21_v3_schema_precision` | 27 | -0.008889 |
| `op17_v3_axis_coverage_grid` | 26 | -0.012697 |
| `op19_v3_cross_field_coherence` | 37 | -0.016382 |
| `op18_v3_behavior_alignment_probes` | 24 | -0.031875 |

这说明在 seed17 中，MCTS 学到 `op16` 的 blueprint binding 方向对提高 validation fitness 最有价值。

### 6.3 MCTS seed23 partial

中断的 seed23 实验达到 gen11 checkpoint，共评估 95 个候选。它不是最终结果，但可以作为稳定性检查。

当前 best：

```text
candidate = openevolve_000073_9bb1698bc6a2
generation = 9
operator = op20_v3_realistic_novelty
fitness = 0.262185
```

gen11 时的 MCTS 根节点 reward：

| 算子 | 访问次数 | 平均 reward |
|---|---:|---:|
| `op17_v3_axis_coverage_grid` | 11 | +0.008888 |
| `op19_v3_cross_field_coherence` | 14 | -0.111036 |
| `op16_v3_blueprint_binding` | 15 | -0.114420 |
| `op21_v3_schema_precision` | 21 | -0.114477 |
| `op20_v3_realistic_novelty` | 19 | -0.125708 |
| `op18_v3_behavior_alignment_probes` | 8 | -0.304546 |

seed23 没有复现 seed17 对 `op16` 的偏好。它当前最优候选来自 `op20`，根节点 MCTS reward 当前更偏向 `op17`。

这一点很重要：`op16` 是有潜力的信号，但还没有证明跨 seed 稳定。

## 7. MCTS 是否有效？

当前回答：

```text
是，MCTS 看起来有助于 validation search。
但否，它还没有证明 sealed-test 泛化一定更好。
```

支持 MCTS 有效的证据：

- MCTS seed17 gen20 超过了之前随机 v3 算子池的最佳 checkpoint。
- 它改善了 schema、一致性、最差一致性、轴对齐和 validation coverage。
- 它通过 `op16` 找到高质量候选，而随机池更偏 `op17` / `op18`，说明 MCTS 不是简单复刻随机搜索，而是产生了非平凡的算子选择。

需要谨慎的证据：

- 完整 seed17 run 的 sealed-test shadow alignment 下降。
- 相比随机 gen18 best，validation behavior diversity 和平均距离下降。
- seed23 partial run 没有复现 `op16` 偏好。
- seed23 受到 API / 网络失败严重影响，不能作为最终结论。

## 8. MCTS 是否损失多样性？

部分是，但不是单向的整体坍缩。

与随机 gen18 相比，MCTS seed17 gen20 表现为：

```text
validation_behavior_coverage: 0.401000 -> 0.407000  略升
validation_behavior_balanced_diversity: 0.328854 -> 0.303466  下降
validation_behavior_avg_dist: 0.484054 -> 0.382886  下降
```

这说明：

- MCTS 没有降低 validation coverage。
- MCTS 确实降低了 validation 行为分散度。
- 搜索方向从更宽的行为距离，转向更强的 blueprint / axis 结构。

在 sealed test 上：

```text
test_behavior_avg_dist: 0.371456 -> 0.464807
```

因此目前不能说出现了明确的全局多样性坍缩。更准确地说，多样性的形态发生了变化：validation 行为距离下降，而 sealed-test 行为距离上升。

## 9. 建议

暂时不要修改主 fitness 公式。

下一步建议低并发 resume 或重跑 seed23：

```text
candidate_max_workers = 1 或 2
persona_max_workers = 2
shadow_max_workers = 8 至 12
```

原因：

- 当前 seed23 run 被 APIConnectionError 失败污染较多。
- 需要一个干净的第二个 seed，才能判断 MCTS 是否稳定有效。
- 如果 seed23 仍然不偏向 `op16`，说明 MCTS 可能是在适应 seed-specific landscape，而不是发现了一个通用最佳算子。

拿到干净的 seed23 结果后，需要评估：

1. MCTS 是否在 validation fitness 上超过随机 v3 算子池；
2. sealed-test alignment 是否恢复；
3. 行为多样性下降是否重复出现；
4. `op16` 是否仍然重要，还是只在 seed17 中出现。

只有在完成这些验证之后，才适合考虑修改 MCTS reward。例如可以加入轻量的多样性 / 对齐保护：

```text
+ validation_behavior_balanced_diversity delta 的 reward
+ validation_shadow_alignment delta 的 reward
- 大幅行为多样性下降的 penalty
```

## 10. 总结

MCTS 值得作为一个实验条件保留。它相比之前的随机 v3 算子池 checkpoint 找到了更好的 validation best candidate，主要收益来自更强的 schema、一致性和轴绑定。

但它还不是完全胜利：sealed-test shadow alignment 下降，第二个 seed 也还不完整，并且显示出不同的算子偏好。

当前结论：

> Hybrid MCTS 可以改善 Genome v3 的 validation search，但其泛化能力和算子偏好稳定性仍需要至少一个干净的额外 seed 来验证。

## 11. 后续补充：gen5 平台化实验与代码修正

在完成上述 MCTS 对比后，又补充分析了一个更大的 Genome v3 pool 长程实验：

```text
data/results/mega_persona_v3_pool_single_call_deepseek_n12_g20_survey4_seed17_23_20260714_1
```

该实验实际保存到 gen29 checkpoint。虽然因为后期 DeepSeek API `503 Server Overloaded` 没有形成完整 final summary，但 checkpoint 足够用于分析进化曲线。

### 11.1 关键观察：best 在 gen5 出现后长期平台化

该实验最重要的现象是：global best 在 gen5 出现后，直到 gen29 都没有再刷新。

| 代数 | global best | 说明 |
|---:|---:|---|
| gen1 | 0.320750 | 初始阶段已有提升 |
| gen3 | 0.321328 | 小幅提升 |
| gen5 | 0.337249 | 当前 best 出现 |
| gen6 至 gen29 | 0.337249 | 长期平台化 |

本次 best candidate：

| 项目 | 值 |
|---|---|
| candidate | `openevolve_000026_8b06cda55ed9` |
| generation | 5 |
| operator | `op19_v3_cross_field_coherence` |
| fitness | 0.337249 |
| validation shadow MAE | 0.150876 |
| strict consistency error | 0.067341 |

从 baseline 到 best，主要指标变化如下：

| 指标 | baseline | best | 变化 |
|---|---:|---:|---:|
| fitness | 0.307988 | 0.337249 | +9.50% |
| schema fitness | 0.484670 | 0.508738 | +4.97% |
| validation coverage | 0.451500 | 0.484000 | +7.20% |
| validation shadow MAE | 0.189200 | 0.150876 | -20.26% |
| balanced diversity | 0.339805 | 0.375658 | +10.55% |
| strict consistency error | 0.066871 | 0.067341 | +0.70% |

解释：

- 进化不是无效的，行为误差、coverage、diversity 和 schema fitness 都有改善。
- 但是改善几乎集中在前 5 代，后续没有继续产生新的 global best。
- strict consistency 没有改善，说明旧 fitness 选出的 best 并不一定是 strict consistency 最好的候选。
- 因此，问题不只是“generation 不够多”，而是搜索机制和选择信号可能过早收敛。

### 11.2 由平台化引出的第一轮代码修正

针对 gen5 后长期平台化，后续代码中加入了 strict-aware 搜索机制，目标是在不直接替换历史 fitness 的前提下，让搜索能看到旧 fitness 捕捉不到的候选。

主要改动：

| 改动 | 目的 |
|---|---|
| 新增 multi-objective elite archive | 除 `global_best` 外，额外保留 `research_score_v2`、`shadow_mae_elite`、`axis_target_elite`、`issue_rate_elite`、`strict_consistency_elite` 等候选 |
| 新增 plateau-aware parent selection | 当 global best 连续 4 代不刷新时，从 strict / shadow / coverage / diversity elites 中选择父代，避免只围绕早期 best 局部扰动 |
| 新增 `research_score_v2` | 在旧 fitness 基础上温和引入 strict consistency 和 shadow MAE 信号，用作并行诊断指标 |
| 更新 MCTS reward | MCTS reward 不再只看旧 global fitness，也奖励 shadow MAE、strict consistency、axis target 和 issue rate 改进 |
| 调整 v3 算子 parent 偏好 | `op16` / `op19` 更偏 strict consistency elite，`op18` 更偏 shadow MAE elite |

这轮修正的核心思想是：

```text
不直接废弃旧 fitness，
但让搜索过程保留 strict-best / shadow-best / coverage-best 等不同方向的候选，
从而降低早期 best 锁死搜索空间的风险。
```

### 11.3 strict-aware 15 代验证实验

完成第一轮修正后，补充跑了一个完整的 n8/g15 验证实验：

```text
data/results/mega_persona_v3_strict_mcts_single_call_deepseek_n8_g15_seed17_20260714
```

结果显示，新机制确实让 best 出现时间从 gen5 推迟到 gen9，并明显改善 strict consistency：

| 指标 | baseline | best | 变化 |
|---|---:|---:|---:|
| validation fitness | 0.252787 | 0.284172 | +12.42% |
| `research_score_v2` | 0.234166 | 0.264742 | +13.06% |
| axis target MAE | 0.093267 | 0.070902 | -23.98% |
| strict consistency error | 0.066600 | 0.049631 | -25.48% |
| validation coverage | 0.399000 | 0.391000 | -2.01% |
| validation balanced diversity | 0.339919 | 0.307771 | -9.46% |

曲线变化：

| 代数 | global best | 说明 |
|---:|---:|---|
| gen0 | 0.252787 | baseline |
| gen1 | 0.265564 | 第一轮提升 |
| gen2 至 gen8 | 0.265564 | 短期平台 |
| gen9 | 0.284172 | 当前 best 出现 |
| gen10 至 gen15 | 0.284172 | 再次平台 |

解释：

- strict-aware 机制缓解了“gen5 后完全不动”的问题，至少在 gen9 出现了第二次跳升。
- 但 gen9 后仍然平台化，说明长期搜索问题还没有完全解决。
- 更重要的是，sealed-test behavior coverage 和 balanced diversity 明显下降，说明 strict reward 可能把候选推向更规整但更窄的空间。

### 11.4 第二轮代码修正：加入多样性保护

基于 n8/g15 strict-aware 实验的副作用，后续又进行了第二轮代码修正，重点是防止 strict consistency 改善以牺牲 coverage / diversity 为代价。

主要改动：

| 改动 | 目的 |
|---|---|
| MCTS reward 加入 coverage / diversity 保护 | 如果 strict consistency 或 shadow MAE 变好，但 coverage / diversity 下降，会额外扣分 |
| 平台期 parent selection 优先 coverage / diversity elite | global best 卡住时，先从行为空间更宽的候选继续探索 |
| `op21_v3_schema_precision` 改从 `coverage_elite` 选父代 | 避免 schema precision 算子持续从最规整、最窄的 schema-best 候选继续收缩 |
| 新增 reward 回归测试 | 确保“strict 变好但 coverage/diversity 坍缩”的候选不会获得高 reward |

第二轮修正后的目标是：

```text
不是单纯追求 strict consistency 更低，
而是在 strict consistency 保持改善的同时，
让 test behavior coverage 和 balanced diversity 回升。
```

### 11.5 当前结论更新

结合后续实验，MCTS 的结论需要比 7 月 13 日初版报告更谨慎：

1. MCTS 对 validation search 是有帮助的；
2. 但它容易把搜索推向结构更强、行为更窄的候选；
3. 旧 fitness 容易在早期找到局部 best 后平台化；
4. strict-aware 搜索可以改善 strict consistency，但需要 coverage / diversity guard；
5. 下一步实验应验证“多样性保护版 MCTS”能否在不牺牲 strict consistency 的情况下恢复 test coverage 和 diversity。
