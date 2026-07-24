# MegaPersona-Evolve 周报

日期：2026-06-28  
周期：2026-06-23 至 2026-06-28  
主题：Genome v3 重构、v3 算子池实验、指标体系梳理与下一阶段实验规划

## 1. 本周总体进展

本周项目重点从“能跑进化实验”推进到“明确进化对象、明确指标口径、验证 Genome v3 是否带来可测收益”的阶段。

本周最重要的变化是：进化对象从早期较分散的 prompt / operator 配置，升级为结构化的 Genome v3。现在 OpenEvolve 进化的不是项目源码，也不是一句固定提示词，而是一个“人格生成器的可配置蓝图”。这个蓝图负责把目标撒点、人格字段、行为问卷响应和一致性规则连接起来。

实验方面，完成了一次较关键的 v3 pool 小规模进化实验：

```text
data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627
```

该实验使用：

- `deepseek`
- `single_call` 人格生成管线
- `student-realistic-v2` 行为模拟器
- `v3` operator pool
- `n=8`
- `generations=10`
- `8` 个岛屿

从 `evolution_dashboard.html` 和 `metric_summary/analysis_summary.md` 看，v3 pool 相比 baseline 出现了明确正向信号：综合 fitness、schema、coverage、consistency、axis alignment、behavior diversity 均有提升。

之后又从 gen10 resume 到 gen18。有效 checkpoint 已保存到：

```text
open_evolve/checkpoint_gen_18.json
```

gen13 出现新的全局最佳。但 gen19 附近出现大量 `APIConnectionError`，后续候选大面积失败，因此提前暂停。该失败主要是网络/API 连接问题，不纳入进化效果判断。

## 2. 本周完成的工作

### 2.1 Genome v3 作为进化对象的重构与文档化

本周明确了 Genome v3 的定位：

> Genome v3 是人格生成器的可配置蓝图，用来描述“如何根据目标空间生成一个结构化、可评估、行为可解释的大人格”。

更具体地说，Genome v3 不是最终生成出来的人格文本，也不是某一句 prompt，而是介于“实验目标空间”和“LLM 生成人格”之间的中间表示。它把一次人格生成中真正需要被优化的内容拆成结构化字段，让 OpenEvolve 可以对这些字段进行变异、选择和保留。

一次候选生成器的实际链路可以概括为：

```text
目标槽位 / Monte Carlo 坐标
        ↓
Genome v3 解释目标和约束
        ↓
形成 persona blueprint 与生成策略
        ↓
LLM 生成完整大人格
        ↓
validator / consistency / shadow survey 评估
        ↓
fitness 反馈给 OpenEvolve
        ↓
operator 继续修改 Genome v3
```

它包含的主要模块包括：

| 模块 | 作用 |
|---|---|
| 目标绑定 | 把 target slot、primary axes、Monte Carlo 坐标绑定到 persona blueprint |
| 维度控制 | 控制认知、动机、自我调节、社交、心理韧性等维度的表达方式 |
| 结构约束 | 保证字段完整、长度合理、schema 合法 |
| 一致性规则 | 约束跨字段冲突，使多个维度描述同一个人 |
| 行为映射 | 将人格特征映射到 shadow survey 的回答倾向 |
| 多样性策略 | 扩大覆盖、避免模板化、保持真实差异 |

因此，Genome v3 的核心作用是把“我要覆盖什么样的人格空间”和“LLM 应该怎么写出这样的人格”连接起来。没有 Genome v3 时，进化往往只能改零散 prompt，难以稳定控制目标轴、字段一致性和问卷行为映射；有了 Genome v3 后，每个候选生成器都可以被表示为一个可比较、可保存、可复现的生成策略。

本周已经在实验记录中补充了 Genome v3 的详细解释和简化示例：

```text
docs/mega_persona_experiment_records/README.md
```

关键判断：

- 当前不直接进化 `generator.py` 或 `evolution.py`。
- 固定 evaluator、validator、shadow survey、checkpoint 和 dashboard。
- 只让 OpenEvolve 修改 Genome v3 中的生成策略。
- 这样可以避免候选生成器通过修改评分、绕过校验或破坏持久化来获得虚假 fitness。

这一设计使实验更可控，也更适合做科学比较。

### 2.2 v3 operator pool 实验完成

本周使用 v3 operator pool 进行了一次小规模实验。实验配置如下：

| 参数 | 值 |
|---|---|
| LLM provider | `deepseek` |
| persona pipeline | `single_call` |
| simulator backend | `student-realistic-v2` |
| operator family | `v3` |
| n | `8` |
| seed | `17` |
| generations | `10` |
| islands | `8` |
| children per island | `1` |
| shadow surveys | `4` |
| validation shadow surveys | `2` |
| test shadow surveys | `2` |
| items per shadow survey | `6` |

本次 v3 operator pool 包括：

| operator | 方向 |
|---|---|
| `op16_v3_blueprint_binding` | 目标坐标到人格蓝图的绑定 |
| `op17_v3_axis_coverage_grid` | 目标空间覆盖和轴向分辨率 |
| `op18_v3_behavior_alignment_probes` | 人格到行为问卷的可解释映射 |
| `op19_v3_cross_field_coherence` | 跨字段一致性 |
| `op20_v3_realistic_novelty` | 真实感和非模板化差异 |
| `op21_v3_schema_precision` | schema 精准度和字段边界 |

### 2.3 v3 pool gen10 实验结果

dashboard 对应的主要结果来自 gen10：

```text
data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627/evolution_dashboard.html
```

结果摘要：

| 项目 | 值 |
|---|---|
| candidate 数 | `81` |
| generations | `10` |
| baseline candidate | `openevolve_000001_bd01ee449da0` |
| best candidate | `openevolve_000063_8098c1dac245` |
| best generation | `8` |
| best operator | `op17_v3_axis_coverage_grid` |
| baseline fitness | `0.215046` |
| best fitness | `0.272785` |

baseline 到 best 的主要指标变化：

| 指标 | baseline | best | 变化量 | 变化百分比 |
|---|---:|---:|---:|---:|
| fitness | 0.215046 | 0.272785 | +0.057739 | +26.85% |
| schema | 0.384090 | 0.452756 | +0.068666 | +17.88% |
| validity | 1.000000 | 1.000000 | +0.000000 | +0.00% |
| consistency | 0.917959 | 0.948271 | +0.030312 | +3.30% |
| consistency_min | 0.852293 | 0.899674 | +0.047381 | +5.56% |
| coverage | 0.290000 | 0.364000 | +0.074000 | +25.52% |
| alignment | 0.810335 | 0.813770 | +0.003435 | +0.42% |
| diversity | 0.230788 | 0.250239 | +0.019451 | +8.43% |
| avg_dist | 0.360622 | 0.391300 | +0.030678 | +8.51% |
| axis_alignment | 0.888959 | 0.934138 | +0.045178 | +5.08% |

主要判断：

1. v3 pool 相比 baseline 有明确提升。
2. 提升不是单一指标驱动，而是 coverage、schema、consistency、axis alignment 等多个指标同时变好。
3. alignment 提升较小，说明人格-问卷行为对齐仍是主要瓶颈。
4. 最佳候选来自 `op17_v3_axis_coverage_grid`，说明目标空间覆盖方向的算子对当前阶段有效。

### 2.4 gen10 到 gen18 的续跑结果

在 gen10 基础上继续 resume，目标跑到 gen20。实际有效 checkpoint 保存到 gen18：

```text
data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627/open_evolve/checkpoint_gen_18.json
```

gen13 出现新的全局最优：

| 项目 | 值 |
|---|---|
| best candidate | `openevolve_000104_480fcc970873` |
| best generation | `13` |
| best operator | `op18_v3_behavior_alignment_probes` |
| fitness | `0.278084` |

gen10 best 到 gen18 checkpoint best 的变化：

| 指标 | gen10 best | gen18 checkpoint best | 变化量 | 变化百分比 |
|---|---:|---:|---:|---:|
| global_best | 0.272785 | 0.278084 | +0.005300 | +1.94% |
| coverage_elite | 0.364000 | 0.401000 | +0.037000 | +10.16% |
| alignment_elite | 0.813770 | 0.842524 | +0.028754 | +3.53% |
| consistency_elite | 0.948271 | 0.936747 | -0.011524 | -1.22% |
| diversity_elite | 0.250239 | 0.328854 | +0.078615 | +31.42% |
| schema_elite | 0.452756 | 0.444982 | -0.007774 | -1.72% |

主要判断：

- 继续跑到 gen18 后仍找到更优解，说明 gen10 不是完全平台期。
- 新最优来自 `op18_v3_behavior_alignment_probes`，说明行为对齐方向的算子在后续阶段开始发挥作用。
- coverage、alignment、diversity 明显改善。
- schema 和 consistency 有小幅下降，需要在后续实验中观察是否属于正常 trade-off。
- gen19 之后出现大量 `APIConnectionError`，导致候选大面积为 `0.0`，不应纳入有效进化结论。

### 2.5 指标体系重新梳理

本周对进化指标做了代码级核对，并写入实验记录。

当前最终适应度不是线性加权，而是乘法门控公式：

```text
behavior_gate    = 0.5 + 0.5 * clip(validation_behavior_coverage, 0, 1)
alignment_gate   = 0.5 + 0.5 * clip(validation_shadow_alignment, 0, 1)
consistency_gate = 0.5 + 0.5 * clip(internal_consistency, 0, 1)

seed_score = schema_fitness
           * consistency_gate
           * behavior_gate
           * alignment_gate
           * generation_rate
```

其中：

- `schema_fitness` 来自 schema 合法性、近重复率和 primary-axis 多样性的合成。
- `internal_consistency` 衡量人格内部一致性。
- `validation_behavior_coverage` 衡量 validation shadow survey 行为轴空间覆盖。
- `validation_shadow_alignment` 衡量人格轴值和问卷行为轴值的一致程度。
- `generation_rate` 衡量成功生成的人格比例。

文档中已经补充：

- `schema_fitness` 公式。
- `near_duplicate_rate` 的 Jaccard 计算口径。
- `DiversityMetrics` 中 coverage、avg_dist、min_dist、KL 的计算方式。
- 内部一致性和轴对齐度的计算方式。
- shadow alignment 的计算方式。
- OpenEvolve elite 指标的来源。

这一部分的意义是：后续所有实验报告都可以明确说明“fitness 为什么涨了”，而不是只展示一个黑盒分数。

### 2.6 可视化与结果表达优化

本周围绕 `evolution_dashboard.html` 做了展示层面的改进和检查。当前 dashboard 已能展示：

- baseline vs best 的指标变化。
- 分代 best fitness / mean fitness 趋势。
- operator 级别表现。
- validation 到 sealed test 的行为指标迁移。

同时需要注意：

- 当前 dashboard 主要对应 gen10 实验结果。
- gen18 的续跑结果来自 checkpoint，尚未重新生成 dashboard。
- 旧版 sealed test 只记录了 `test_shadow_alignment`、`test_behavior_coverage`、`test_behavior_avg_dist` 三个行为指标。
- 后续实验应使用已经补全的完整 sealed test 指标，包括 test schema、test consistency、test axis alignment 等。

### 2.7 MCTS-style 算子选择策略初步加入

本周新增了一个可选的 hybrid MCTS-style operator policy。它的作用不是生成人格，也不是做目标空间撒点，而是在进化过程中选择下一步使用哪个 operator。

当前定位：

```text
随机选算子：验证 v3 算子池本身是否有效
MCTS 选算子：验证能否更高效地利用 v3 算子池
```

当前判断：

- MCTS-style 选择策略有实验价值。
- 但它会引入额外变量，不应立即作为默认主流程。
- 下一阶段可以将其作为消融实验，与随机 operator 选择进行对比。

## 3. 本周关键结论

### 3.1 Genome v3 是当前更合理的进化对象

直接进化源码会让候选之间不可比较，也可能引入绕过评分、破坏 validator、改变数据保存逻辑等问题。Genome v3 将可变部分限制在生成策略层，使进化结果更可解释。

本项目选择 Genome v3，而不是直接进化代码或只进化普通 prompt，主要有五个原因：

1. 可控性更强  
   代码层面的采样、评估、validator、持久化都保持不变，进化只发生在“人格生成策略”上。这样 fitness 的变化更能反映生成策略变化，而不是工程逻辑被改动。

2. 可比较性更强  
   所有候选都遵循同一套生成流程和同一套评估流程。不同候选之间的差异主要来自 Genome v3 字段，而不是来自各自改了不同 Python 代码。

3. 可复现性更强  
   Genome v3 可以作为 JSON-like 的结构保存下来。每一代候选、最优 genome、operator 来源都能持久化，后续可以复查“到底进化出了什么”。

4. 更适合大人格任务  
   大人格不是单一标签，而是由认知、动机、自我调节、价值观、社交、心理健康、行为倾向等多个字段共同组成。Genome v3 可以显式描述这些字段之间的绑定和一致性规则，普通 prompt 很难稳定做到这一点。

5. 更适合做科学实验  
   后续可以对 Genome v3 的不同模块做消融：只改目标绑定、只改行为映射、只改一致性规则、只改多样性策略。这样能够回答“哪个生成机制真的带来提升”，而不只是看到一个黑盒 prompt 变好了。

因此，当前实验路线应该继续坚持：

```text
固定主流程 + 固定评估器 + 固定 validator + 进化 Genome v3
```

### 3.2 v3 operator pool 已经出现有效进化信号

gen10 dashboard 显示：

- fitness 提升 `26.85%`
- coverage 提升 `25.52%`
- schema 提升 `17.88%`
- consistency 提升 `3.30%`
- axis alignment 提升 `5.08%`

这比之前 genome v2 中“各指标变化微乎其微”的情况更有说服力。

### 3.3 后续重点应转向复现和稳定性

当前结果仍属于探索性结论，原因是：

- 主要实验只有单 seed。
- n 只有 `8`。
- gen18 后半段受到 APIConnectionError 干扰。
- dashboard 尚未基于 gen18 重新生成。
- sealed test 完整指标需要在后续 run 中补齐。

因此，下一阶段不是盲目扩大规模，而是先做可复现实验。

## 4. 当前问题与风险

| 问题 | 影响 | 处理方式 |
|---|---|---|
| 单 seed 结果可能有偶然性 | 无法直接作为正式结论 | 做多 seed 小规模复现 |
| gen19 出现大量 APIConnectionError | 后几代候选为 0，不能纳入分析 | 只使用完整 checkpoint 到 gen18 的结果 |
| alignment 提升较小 | 行为对齐仍是瓶颈 | 重点观察 `op18` 和行为映射相关 genome 字段 |
| schema/consistency 在 gen13 best 有小幅下降 | 说明 coverage/alignment 提升可能有 trade-off | 后续看 sealed test 和多 seed 是否稳定 |
| MCTS 引入额外变量 | 会干扰 v3 pool 本身有效性判断 | 作为单独消融实验，不默认启用 |
| 旧 sealed test 指标不完整 | 泛化判断不充分 | 使用新代码重跑完整 sealed test |

## 5. 下周计划

### 5.1 任务三：多 seed 小规模复现

目标：排除单 seed 偶然性。

建议：

- `n=8`
- `generations=10`
- `seeds=17,23,31`
- `operator-family=v3`

重点观察：

- best over baseline 的平均提升。
- 不同 seed 下是否仍出现 `op17` / `op18` 优势。
- sealed test 是否稳定。

### 5.2 任务四：MCTS 算子选择消融实验

目标：验证 MCTS-style operator policy 是否能比随机算子选择更快找到高分候选。

对比组：

```text
--search-strategy openevolve
--search-strategy hybrid_mcts
```

控制变量：

- 同一 n。
- 同一 seed。
- 同一 generations。
- 同一 v3 operator pool。
- 同一 simulator。

判断标准：

- 是否更早找到高 fitness。
- 是否提高 gen10 / gen20 best。
- 是否降低无效候选比例。
- 是否过度偏向早期偶然高分算子。

### 5.3 任务五：中规模确认实验

在小规模复现稳定后，再进入中规模：

- `n=16`
- `generations=20`
- `seeds=17`
- `operator-family=v3`
- `shadow-surveys=6`
- `validation-shadow-surveys=3`
- `test-shadow-surveys=4`

目标是确认 v3 pool 在更大人格数量下仍然有效。

## 6. 本周总结

本周最重要的成果不是单次 fitness 提升，而是实验路线变得更清楚：

1. 进化对象明确为 Genome v3，而不是源码或单句 prompt。
2. v3 operator pool 在 gen10 实验中产生了可见提升。
3. gen18 续跑进一步发现 `op18` 方向有潜力。
4. 指标体系已经按代码真实口径写清楚，可以支撑后续严谨汇报。
5. 后续重点应从“继续堆大实验”转为“复现、消融、sealed test 完整化”。

当前可以形成阶段性判断：

> Genome v3 + v3 operator pool 已经比早期 genome v2 更有实验价值；下一步需要通过固定算子、多 seed 和完整 sealed test 验证其稳定性。
