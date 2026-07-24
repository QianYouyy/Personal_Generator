# MegaPersona Student-Realistic OpenEvolve Stage 1

日期：2026-06-24  
Batch：`mega_persona_openevolve_student_realistic_medium_20260624`  
结果目录：`data/results/mega_persona_openevolve_student_realistic_medium_20260624`

## 1. 实验目的

本批实验是阶梯式放大路线中的 Stage 1，用于验证 `student-realistic` 作为主 evaluator 时，OpenEvolve 是否能推动人格生成器获得更好的行为覆盖与人格-行为对齐。

此前 simulator audit 已显示 `student-realistic` 在 offline evaluator 对比中综合表现最好。本批进一步验证它在真实进化闭环中是否可用。

## 2. 实验设置

运行配置：

- generator mode：`llm`
- simulator backend：`student-realistic`
- simulator model：`gpt-4o-mini`
- personas per candidate：`32`
- seeds：`17,23`
- generations：`3`
- islands：`8`
- children per island：`1`
- candidate max workers：`4`
- persona max workers：`4`
- shadow max workers：`8`
- train shadow surveys：`8`
- validation shadow surveys：`4`
- sealed test shadow surveys：`4`
- items per survey：`10`

总运行时间约 `56` 分钟。

日志检查：

- OpenEvolve 正常完成 `3` 代。
- 输出了 `checkpoint_gen_1.json`、`checkpoint_gen_2.json`、`checkpoint_gen_3.json`。
- 最终 sealed test 正常完成。
- 出现过 1 次 simulator 连接重试、1 次 persona LLM 连接重试，均自动恢复。
- 若干 persona validation revision 属于生成器自动修复流程，不是实验失败。

## 3. 最优结果

最优 candidate：

- candidate：`openevolve_000008_cbd9f1e2a6c6`
- OpenEvolve generation：`1`
- island：`6`
- validation fitness：`0.3979`
- operator：`op13_autonomy_pressure_test`
- mutation mode：`mixed`

最优 genome 特征：

- `axis_stretch.cognitive_abstraction = 1.0647`
- `axis_stretch.motivation_autonomy = 1.1758`
- `axis_stretch.self_regulation_resilience = 1.0580`
- `prompt_profile.axis_binding = mechanistic`
- `prompt_profile.behavioral_signal = action_predictive`
- `prompt_profile.mechanism_focus = motivational`

说明：本轮最优解来自 `op13_autonomy_pressure_test`，即通过“外部压力/义务/权威与个人兴趣冲突”的情境来暴露 autonomy 差异。这与 `student-realistic` evaluator 的机制设计是匹配的。

## 4. Validation 指标

| Metric | Value |
|---|---:|
| validation fitness | 0.3979 |
| validation behavior coverage | 0.4010 |
| validation shadow alignment | 0.8061 |
| schema fitness | 0.6290 |
| validity rate | 1.0000 |
| near duplicate rate | 0.0000 |
| train behavior coverage | 0.3840 |
| train shadow alignment | 0.8072 |
| slot coverage | 0.9595 |

解读：

- schema validity 保持 `1.0000`，说明进化没有以牺牲合法性换覆盖度。
- near duplicate rate 为 `0.0000`，说明当前样本规模下没有明显换皮重复。
- validation coverage 高于 train coverage，说明不是只在 train shadow surveys 上过拟合。
- train/validation alignment 非常接近，说明泛化稳定。

## 5. Sealed Test 结果

| Metric | Value |
|---|---:|
| test shadow alignment mean | 0.8006 |
| test behavior coverage mean | 0.3950 |
| test behavior avg dist mean | 0.3156 |
| test shadow alignment std | 0.0061 |
| test behavior coverage std | 0.0010 |
| test behavior avg dist std | 0.0020 |

sealed test 没有参与选择，且 test coverage `0.3950` 接近 validation coverage `0.4010`，说明本批没有明显 train/validation-only 的过拟合迹象。

## 6. 代际趋势

OpenEvolve checkpoint 记录的每代统计：

| Generation | Evaluations | Improvements | Best Fitness | Mean Fitness | Max Coverage | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 0.3979 | 0.3818 | 0.3979 | 0.3818 |
| 2 | 8 | 5 | 0.3979 | 0.3914 | 0.3979 | 0.3914 |
| 3 | 8 | 5 | 0.3979 | 0.3926 | 0.3979 | 0.3926 |

结论：

- 最优解在第 1 代出现，后续没有被超过。
- 但第 2/3 代的平均 fitness 上升，说明群体整体被推向更好的区域。
- 这不是完全无效进化；更准确地说，是“早期找到强解，后续群体追平但未突破”。

## 7. Candidate 与 Operator 观察

Top candidates：

| Candidate | Fitness | Val Coverage | Val Alignment | Operator | Mode |
|---|---:|---:|---:|---|---|
| `openevolve_000008_cbd9f1e2a6c6` | 0.3979 | 0.4010 | 0.8061 | `op13_autonomy_pressure_test` | `mixed` |
| `openevolve_000004_33ebeec90266` | 0.3919 | 0.4000 | 0.8149 | `op01_axis_decoupling` | `prompt_only` |
| `openevolve_000007_7683a553a778` | 0.3880 | 0.4035 | 0.8056 | `op13_autonomy_pressure_test` | `operator_only` |
| `openevolve_000018_a2de107dbcc2` | 0.3873 | 0.3935 | 0.8000 | `op13_autonomy_pressure_test` | `operator_only` |
| `openevolve_000016_db6b35ef810c` | 0.3834 | 0.3950 | 0.8068 | `op15_survey_discriminating_cues` | `mixed` |

按 operator 粗略平均：

| Operator | Count | Avg Fitness | Max Fitness |
|---|---:|---:|---:|
| `op01_axis_decoupling` | 2 | 0.3869 | 0.3919 |
| `op13_autonomy_pressure_test` | 4 | 0.3862 | 0.3979 |
| `op15_survey_discriminating_cues` | 1 | 0.3834 | 0.3834 |
| `op10_contextual_bucket_split` | 1 | 0.3800 | 0.3800 |
| `op12_support_network_asymmetry` | 2 | 0.3787 | 0.3821 |

观察：

- `op13` 是本批最强信号，最高分来自它，且多次进入 top。
- `op01` 平均表现高，但它之前被认为有风险，后续应继续观察而不是立即放大权重。
- `op02_behavioral_evidence` 本批表现靠后，可能在 `student-realistic` evaluator 下不如预期。
- numeric-only 变异整体偏弱。

## 8. 阶段性结论

Stage 1 结果支持继续放大实验规模。

理由：

1. `student-realistic` 能在 OpenEvolve 闭环中产生有效选择信号。
2. 最优 validation fitness 达到 `0.3979`，sealed test coverage 达到 `0.3950`。
3. train、validation、test 三者没有明显背离。
4. schema validity 与 near duplicate 控制正常。
5. 运行过程有少量瞬时网络失败，但重试机制正常工作。

需要注意：

- 最优解出现在第 1 代，后续代际未突破，说明 Stage 2 需要更大的搜索空间或更多 generation 才能观察持续改进。
- candidate 记录中的 `generation` 字段仍显示为 `0`，但 OpenEvolve checkpoint 中真实 best generation 是 `1`。后续可以修复 candidate JSON 的 generation 记录，以免报告时混淆。

## 9. 下一步建议

建议进入 Stage 2：`n=48, seeds=17/23/31, generations=4, islands=10`。

同时建议保留两个观察点：

- 重点观察 `op13` 是否继续稳定有效。
- 检查 Stage 2 是否能突破 `0.3979`，而不只是复制 Stage 1 的早期强解。

如果 Stage 2 中：

- validation fitness 超过 `0.41`
- sealed test coverage 稳定在 `0.40+`
- test alignment 维持 `0.79+`

则可以进入 Stage 3 formal run。

