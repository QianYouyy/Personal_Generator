# MegaPersona OpenEvolve Medium Run Analysis

日期：2026-06-16  
批次目录：`data/results/mega_persona_openevolve_medium_20260615_n24_i6_g3`

## 1. 实验状态

本次 OpenEvolve 中等规模实验已完整跑完。实验从 2026-06-15 19:01:25 开始，到 2026-06-16 13:55:39 结束，总运行时间约 `18.9` 小时。

- engine：`src.open_evolve.engine.OpenEvolve`
- 人格规模：每个 candidate 每个 seed 生成 `24` 个人格
- seeds：`17, 23`
- island 数量：`6`
- generations：`3`
- children per island：`1`
- 评估 candidate 数：`19`
- shadow surveys：train 6 份，validation 3 份，test 3 份
- 每份 shadow survey item 数：`8`
- shadow simulation 并发：`4`
- best candidate：`openevolve_000022_84f5f9724380`
- best validation fitness：`0.373588`

本次日志中没有致命 `ERROR`。存在若干 LLM timeout / connection transient warning，但均通过 retry 机制继续推进，最终 sealed test 和 summary 均成功写出。

## 2. 主要结论

这次实验提供了比前几轮更明确的正向进化信号：最终 best 来自第 3 代，而不是 baseline。

baseline candidate：

| Candidate | Fitness | Schema | Val Coverage | Val Alignment | Slot Coverage |
|---|---:|---:|---:|---:|---:|
| `openevolve_000001_22d221eb496f` | `0.342958` | `0.578757` | `0.334000` | `0.776317` | `0.896500` |

best candidate：

| Candidate | Generation | Fitness | Schema | Val Coverage | Val Alignment | Slot Coverage |
|---|---:|---:|---:|---:|---:|---:|
| `openevolve_000022_84f5f9724380` | `3` | `0.373588` | `0.600260` | `0.382500` | `0.801594` | `0.864500` |

相对 baseline：

- fitness：`+0.030629`
- 相对提升：约 `+8.9%`
- schema fitness：`+0.021503`
- validation behavior coverage：`+0.048500`
- validation shadow alignment：`+0.025278`
- slot coverage：`-0.032000`

这说明进化主要提升了人格的 schema/行为质量和行为覆盖，而不是单纯提高 slot coverage。

## 3. 代际变化

OpenEvolve history：

| Generation | Evaluations | Improvements | Best Fitness | Mean Fitness | Min Fitness | Std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 6 | `0.361572` | `0.353517` | `0.346249` | `0.005417` |
| 2 | 6 | 5 | `0.361572` | `0.359832` | `0.356882` | `0.001952` |
| 3 | 6 | 3 | `0.373588` | `0.362128` | `0.356882` | `0.005484` |

按实际 mutation generation 聚合 candidate：

| Generation | N | Mean Fitness | Best Fitness | Min Fitness | Std |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | `0.342958` | `0.342958` | `0.342958` | `0.000000` |
| 1 | 6 | `0.348790` | `0.357580` | `0.337351` | `0.007255` |
| 2 | 6 | `0.347318` | `0.359813` | `0.332547` | `0.010546` |
| 3 | 6 | `0.346445` | `0.373588` | `0.316363` | `0.017787` |

解释：

- 第 1 代已经超过 baseline，说明进化机制不是无效扰动。
- 第 2 代 best 未继续提高，但平均值较好，说明搜索进入局部稳定区。
- 第 3 代找到明显更好的 candidate，但方差也变大，说明后期探索风险增加。

## 4. 最优 Genome

最终 best 使用：

- mutation mode：`operator_only`
- operator：`op06_low_axis_fidelity`
- mutation scale：`0.17112`
- OpenEvolve generation：`3`
- stagnation：`1`

operator 内容：

```text
When a target axis is low, show a concrete cost in behavior and do not rescue it with generic competence.
Low values should remain plausible but measurably visible.
```

best prompt profile：

| Category | Choice |
|---|---|
| anti_stereotype | `explicit` |
| axis_binding | `mechanistic` |
| behavioral_signal | `mixed_evidence` |
| coverage_strategy | `edge_cases` |
| mechanism_focus | `balanced` |
| specificity | `concrete` |
| tension_level | `moderate` |

best quota weights：

| Quota Bucket | Weight |
|---|---:|
| `belonging_oriented_collaborator` | `0.221613` |
| `self_directed_builder` | `0.194794` |
| `curious_low_structure` | `0.185186` |
| `anxious_high_effort` | `0.137036` |
| `reserved_resilient_observer` | `0.136872` |
| `externally_driven_performer` | `0.124499` |

## 5. Mutation Mode 结果

按 mutation mode 聚合：

| Mode | N | Mean Fitness | Best Fitness | 判断 |
|---|---:|---:|---:|---|
| `operator_only` | 3 | `0.362683` | `0.373588` | 当前最强，稳定性最好 |
| `prompt_only` | 6 | `0.350276` | `0.359813` | 有收益，但上限低于 operator_only |
| `numeric_only` | 4 | `0.346783` | `0.354045` | 略高于 baseline，但作用有限 |
| `mixed` | 5 | `0.335696` | `0.348951` | 本轮不稳定，平均低于 baseline |
| None / baseline | 1 | `0.342958` | `0.342958` | 初始默认 genome |

本轮结论与前一轮 operator ablation 有一个重要变化：之前 `mixed` 的上限较高，但在本次正式 OpenEvolve 里，`mixed` 平均表现偏弱；`operator_only` 更适合当前主流程。

## 6. Operator 结果

按 operator 聚合：

| Operator | N | Mean Fitness | Best Fitness | 判断 |
|---|---:|---:|---:|---|
| `op06_low_axis_fidelity` | 2 | `0.364543` | `0.373588` | 本轮最强 |
| `op04_within_bucket_contrast` | 1 | `0.359813` | `0.359813` | 延续此前强信号 |
| `op07_high_axis_cost` | 1 | `0.357580` | `0.357580` | 正向，值得保留 |
| `op01_axis_decoupling` | 1 | `0.354877` | `0.354877` | 本轮单点正向，但历史风险较高 |
| `op02_behavioral_evidence` | 3 | `0.351010` | `0.356882` | 稳定中等 |
| None / numeric-only | 5 | `0.346018` | `0.354045` | 可作为对照，不宜主推 |
| `op05_failure_recovery_cycle` | 1 | `0.334036` | `0.334036` | 本轮偏弱 |
| `op03_shadow_survey_alignment` | 4 | `0.332383` | `0.345699` | 本轮偏弱 |
| `op08_validation_conservatism` | 1 | `0.350235` | `0.350235` | 中性偏正 |

当前最值得进入下一轮重点测试的 operator：

1. `op06_low_axis_fidelity`
2. `op04_within_bucket_contrast`
3. `op07_high_axis_cost`
4. `op02_behavioral_evidence`

需要降权或谨慎使用：

1. `op03_shadow_survey_alignment`
2. `op05_failure_recovery_cycle`
3. `mixed` mode 下的宽扰动

## 7. Top Candidates

| Rank | Candidate | Gen | Mode | Operator | Fitness | Schema | Val Coverage | Val Alignment |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | `openevolve_000022_84f5f9724380` | 3 | `operator_only` | `op06` | `0.373588` | `0.600260` | `0.382500` | `0.801594` |
| 2 | `openevolve_000016_21eabd597e3f` | 2 | `prompt_only` | `op04` | `0.359813` | `0.587306` | `0.371000` | `0.783040` |
| 3 | `openevolve_000012_41438d54392e` | 1 | `operator_only` | `op07` | `0.357580` | `0.596925` | `0.344500` | `0.781034` |
| 4 | `openevolve_000015_deeba110929b` | 2 | `operator_only` | `op02` | `0.356882` | `0.593106` | `0.350500` | `0.781034` |
| 5 | `openevolve_000020_b25444805a36` | 3 | `prompt_only` | `op06` | `0.355498` | `0.569261` | `0.383000` | `0.805803` |

Top 5 中 `operator_only` 占 3 个，进一步支持降低 `mixed`、提高 `operator_only` 比例。

## 8. Sealed Test

最终 best 在 sealed test 上的表现：

| Metric | Value |
|---|---:|
| `test_shadow_alignment.mean` | `0.795497` |
| `test_behavior_coverage.mean` | `0.359500` |
| `test_behavior_avg_dist.mean` | `0.352857` |
| `test_shadow_alignment.std` | `0.016135` |
| `test_behavior_coverage.std` | `0.009500` |
| `test_behavior_avg_dist.std` | `0.010603` |

对比 validation：

| Metric | Validation | Test | Gap |
|---|---:|---:|---:|
| behavior coverage | `0.382500` | `0.359500` | `-0.023000` |
| shadow alignment | `0.801594` | `0.795497` | `-0.006097` |

test 相比 validation 略低，但没有明显崩坏。当前更像是轻微 validation 优化，而不是严重过拟合。

## 9. 质量与风险

正向信号：

- best 来自第 3 代，而不是 baseline。
- best fitness 相比 baseline 提升约 `8.9%`。
- validation 和 sealed test 差距不大。
- validity rate 保持 `1.0`，near duplicate rate 保持 `0.0`。
- `op06`、`op04`、`op07`、`op02` 与之前消融结论基本兼容。

主要风险：

- 本次仍只有 `2` 个 seed，统计置信度有限。
- candidate 总数为 `19`，operator 覆盖不均衡。
- `mixed` 方差较大，可能引入不必要的 schema/behavior 扰动。
- best 的 slot coverage 低于 baseline，说明优化更偏向行为质量，不是全指标同步提升。
- 运行耗时接近 `19` 小时，正式实验前需要继续优化并发和缓存策略。

技术注意：

- candidate artifact 的顶层 `generation` 字段均为 `0`，但实际 OpenEvolve mutation generation 保存在 genome 的 `openevolve_mutation.generation` 中。本报告按 `openevolve_mutation.generation` 统计代际。

## 10. 后续建议

下一轮不建议继续均匀随机抽所有 operator。建议改成 focused evolution：

1. 提高 `op06_low_axis_fidelity` 的采样权重。
2. 保留高权重 `op04_within_bucket_contrast`。
3. 中等权重保留 `op07_high_axis_cost` 和 `op02_behavioral_evidence`。
4. 降低 `op03_shadow_survey_alignment`、`op05_failure_recovery_cycle` 的权重。
5. 将 mutation mode 主体改为 `operator_only`，少量保留 `prompt_only`，显著降低 `mixed`。
6. 增加 seed 数到 `17,23,31`，确认 `op06` 是否稳定泛化。
7. 在下一轮报告中同时记录 baseline、generation best、operator group mean、sealed test gap。

推荐下一轮命令方向：

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-model-key llm.simulator_model \
  --n 24 \
  --seeds 17,23,31 \
  --generations 3 \
  --num-islands 6 \
  --children-per-island 1 \
  --elite-count 3 \
  --shadow-surveys 6 \
  --validation-shadow-surveys 3 \
  --test-shadow-surveys 3 \
  --items-per-shadow-survey 8 \
  --shadow-max-workers 4 \
  --base-mutation-scale 0.12 \
  --output-dir data/results/mega_persona_openevolve_focused_20260616_n24_i6_g3
```

在跑这条前，建议先修改 operator/mutation mode 采样权重，否则仍然会有较多预算花在本轮表现较弱的 operator 与 `mixed` mode 上。

## 11. 一句话结论

本次中等规模 OpenEvolve 实验首次给出了比较清晰的正向进化证据：best candidate 在第 3 代达到 `0.373588`，相比 baseline 提升约 `8.9%`，并且 sealed test 没有明显崩坏。当前最强信号来自 `op06_low_axis_fidelity` 与 `operator_only` mode；下一步应从“均匀探索所有 operator”转为“围绕 op06/op04 的加权 focused evolution”。
