# MegaPersona 实验规划与记录

更新时间：2026-06-27

本目录用于集中记录 MegaPersona 项目的实验目的、技术路线、关键实验结果、阶段性判断和后续计划。后续每次重要实验应在本目录下追加独立记录，避免实验结论散落在不同批次文件中。

## 1. 实验目的

本项目的核心目标不是简单生成更多人格，而是验证：

1. 能否通过 OpenEvolve 对人格生成器进行有效进化，使生成的人格在目标空间中覆盖更广。
2. 能否在提高覆盖度的同时，保持人格结构合法、内部一致、行为问卷响应合理。
3. 能否将大人格生成从固定 Prompt 工程推进到可迭代优化的“生成器进化”范式。

因此，实验关注的是一组联合指标，而不是单一的多样性指标。理想结果应同时满足：

- Schema 合法率稳定，不出现大量字段缺失、长度越界或类型错误。
- 人格内部一致性较高，不出现明显跨维度冲突。
- Primary axes / target slots 能被有效覆盖。
- Shadow survey 行为响应与人格描述保持一致。
- 验证集提升能够迁移到 sealed test，而不是只在验证问卷上过拟合。

## 2. 技术路线

### 2.1 目标空间设计

生成前先定义目标人格空间，使用配额槽和连续坐标共同约束人格初始化。

- 配额槽：保证群体结构可控，例如不同认知、动机、自我调节或社会维度的组合。
- 连续坐标：在 primary axes 上做 Monte Carlo / quasi-random 式撒点，避免人格集中在少数模板区域。
- Axis extractor：把生成文本和结构化字段映射回可计算指标，用于覆盖度、轴对齐和后续评估。

### 2.2 人格生成器

当前探索阶段优先使用 `single_call` 管线，即一次 LLM 调用生成完整大人格。这样做的目的：

- 降低早期进化实验的 LLM 调用成本。
- 减少 5-agent 管线中多次调用带来的累计噪声。
- 更容易观察进化算子对生成器整体行为的影响。

后续如果需要更高文本质量或更强分模块可控性，可以再与 multi-agent / five-agent 管线比较。

### 2.3 进化对象

当前进化对象已经升级到 Genome v3。这里的 genome 不是生物学意义上的基因，而是“人格生成器的可配置蓝图”。它把一次人格生成中真正需要被优化的部分显式拆出来，让 OpenEvolve 改的是生成策略，而不是任意改项目源码。

Genome v3 可以理解为一个结构化生成器配置，主要包含以下几类内容：

| 模块 | 进化内容 | 作用 |
|---|---|---|
| 目标绑定 | 如何把 target slot、primary axes、撒点坐标绑定到 persona blueprint | 决定生成器是否真的执行蒙特卡洛目标，而不是忽略坐标自由发挥 |
| 维度控制 | 每个维度的强度、极性、描述密度、边界条件 | 决定人格在认知、动机、自我调节、社交、心理韧性等维度上的可控变化 |
| 结构约束 | 字段完整性、长度约束、schema 约束、字段间依赖 | 保证生成结果能通过 validator，不出现格式合法但语义崩坏的人格 |
| 一致性规则 | 跨字段冲突处理、互相支撑的证据链、例外解释 | 防止“大人格”内部不同模块互相打架 |
| 行为映射 | 人格特征如何转化为 shadow survey 回答倾向 | 让问卷行为不是随机答案，而是能被人格描述解释 |
| 多样性策略 | 如何扩大覆盖、避免模板重复、保留真实差异 | 提高覆盖度和差异性，但不把人格推向不真实的极端 |

因此，Genome v3 进化的不是某一段固定 prompt，而是“生成器如何理解目标、如何组织人格、如何平衡覆盖度和一致性”的一组高层策略。它比早期 v2 更适合观察“进化是否真的改变生成器行为”。

#### Genome v3 的工作方式

一次候选生成器评估时，Genome v3 的作用链条是：

```text
target slot / Monte Carlo 坐标
        ↓
Genome v3 解释目标坐标和约束
        ↓
生成 persona blueprint 与字段写作策略
        ↓
LLM 根据 genome 生成完整大人格
        ↓
validator / consistency / shadow survey 评估
        ↓
fitness 反馈给 OpenEvolve
        ↓
operator 继续修改 Genome v3
```

所以，Genome v3 不是最终人格文本本身，而是“如何生成这类人格”的策略描述。相同的 target slot，如果使用不同 genome，生成器会以不同方式解释目标轴、组织心理机制、处理字段冲突，并最终产生不同质量的人格。

Genome v3 的关键点有三个：

1. 它把目标空间和文本生成连接起来  
   例如 `cognitive_abstraction=high` 不能只让 LLM 写“这个学生很聪明”，而要要求它写出可观察触发场景、优势、边界条件和失败情境。

2. 它把多字段人格组织成同一个人  
   认知、动机、自我调节、社交、健康不能像五份独立小作文，而要围绕同一套核心张力和行为证据互相呼应。

3. 它把人格描述和问卷行为连接起来  
   如果人格里写“遇到模糊任务会先找样例再动手”，shadow survey 里面对学习策略、求助、拖延、压力恢复等题目时，就应该能推出相对一致的回答。

#### Genome v3 简化示例

下面是一个简化后的 Genome v3 示例。真实实验中的 genome 字段更多，但核心结构与此类似：

```json
{
  "blueprint_policy": {
    "axis_evidence_rule": "For each primary axis, name one observable school-life trigger and one boundary condition.",
    "core_tension_rule": "Express the target axis combination as a concrete trade-off in a specific scenario.",
    "persona_coherence_rule": "All sections must describe the same student and reuse the same trigger-behavior evidence."
  },
  "axis_expression_policy": {
    "high": "Include a clear advantage and a boundary where the high trait becomes a liability.",
    "mid": "Describe two contexts where the trait appears differently because of a named trigger.",
    "low": "Include a visible cost and a limited workaround that does not fully compensate."
  },
  "behavior_anchors": {
    "learning": "Trigger: ambiguous assignment. Interpretation: what evidence feels useful. Response: planning, asking, or experimenting.",
    "motivation": "Trigger: external expectation conflicts with interest. Response: compromise, persistence, or avoidance.",
    "stress_recovery": "Trigger: deadline or failed attempt. Response: latency, coping routine, and later adjustment.",
    "belonging": "Trigger: group task or teacher feedback. Response: help-seeking, withdrawal, or contribution."
  },
  "consistency_rules": [
    "Cognition, motivation, self-regulation, social behavior, and health must describe the same student.",
    "A limitation in one field must have a plausible echo in at least one other field.",
    "Contrasts are allowed only when a context boundary explains why behavior changes."
  ],
  "openevolve_mutation": {
    "operator_id": "op18_v3_behavior_alignment_probes",
    "backend": "llm",
    "mutation_mode": "mixed"
  }
}
```

这个例子说明了 Genome v3 进化的不是一句 prompt，而是一组生成机制：

- `blueprint_policy` 决定人格先怎么搭骨架。
- `axis_expression_policy` 决定高、中、低轴值如何写得有区分度。
- `behavior_anchors` 决定人格如何映射到问卷行为。
- `consistency_rules` 决定跨字段如何保持同一个人。
- `openevolve_mutation` 记录这个 genome 是由哪个算子、以什么方式变异出来的。

例如，当目标 slot 是“高认知抽象 + 中等自主动机 + 低自我调节”时，这个 genome 不会只生成一个笼统的“聪明但拖延”的学生，而会推动生成器写出：

- 他在开放题中能快速抽象规律。
- 但在截止日期临近时容易过度构思，执行延迟。
- 他不是完全没有动机，而是在外部任务和个人兴趣冲突时会摇摆。
- 这种摇摆会同时出现在学习策略、压力恢复和社交协作字段里。
- 问卷中面对拖延、求助、任务规划、失败反馈时，应出现可解释的回答模式。

这就是 Genome v3 比普通 prompt 更适合作为进化对象的原因：它能把目标轴、人格结构、行为预测和一致性约束放进同一个可变但可评估的生成框架中。

#### 为什么不直接进化代码

现阶段不直接让 OpenEvolve 改 `generator.py`、`evolution.py` 或 validator 源码，主要有几个原因：

1. 科学可控性更强  
   实验要验证的是“生成策略是否能被进化优化”，而不是验证 LLM 能不能写出一段侥幸跑通的 Python。固定主流程、只进化 genome，可以把变量控制在生成逻辑本身。

2. 可复现性更好  
   如果每代都改源码，候选之间可能出现不可比较的问题：有的改了采样，有的改了评分，有的绕过校验，有的改变数据保存。这样 fitness 提升不一定代表人格质量提升。Genome v3 保持评估器、validator、数据持久化和 OpenEvolve 框架不变，结果更容易复现和解释。

3. 安全边界更清楚  
   直接进化代码容易生成不可执行代码、无限循环、异常吞掉、绕过 schema、篡改指标等问题。Genome v3 把可变范围限制在生成器的策略层，降低实验被“作弊解”污染的风险。

4. 便于做消融实验  
   Genome v3 的字段是结构化的，可以单独比较目标绑定、行为映射、一致性规则、多样性策略等部分的贡献。如果直接进化源码，很难判断到底是哪一处代码变化带来了效果。

5. 更符合当前研究问题  
   当前问题不是“让系统自动重写一个项目”，而是“如何构建一个可进化的大人格生成器”。因此更合理的进化对象是生成器基因组，而不是整个代码库。

#### 固定代码和可进化内容的边界

当前实验中固定不变的部分包括：

- OpenEvolve 调度机制。
- candidate 评估流程。
- schema validator。
- shadow survey 划分与 sealed test 机制。
- 指标计算逻辑。
- 数据持久化和 dashboard 生成逻辑。

允许进化的部分包括：

- persona blueprint 的组织方式。
- target coordinates 到人格字段的映射方式。
- 每个心理维度的表达策略。
- 字段之间的一致性维护策略。
- 行为问卷回答倾向的描述策略。
- 覆盖度、多样性和真实感之间的平衡策略。

这种边界设计的核心思想是：让进化作用在“人格生成的科学假设和表达策略”上，而不是作用在“评估系统和工程基础设施”上。

#### Genome v3 与进化算子的关系

v3 operator pool 的作用不是直接生成最终人格，而是修改 Genome v3 的不同部分。例如：

- `op17_v3_axis_coverage_grid` 主要修改目标覆盖和轴向分辨率相关配置。
- `op18_v3_behavior_alignment_probes` 主要修改人格到行为问卷的映射策略。
- `op19_v3_cross_field_coherence` 主要修改跨字段一致性规则。
- `op21_v3_schema_precision` 主要修改 schema 和字段边界表达。

也就是说，operator 是“怎么变异”的规则，Genome v3 是“被变异的对象”。最终生成的人格质量由二者共同决定。

### 2.4 进化机制

项目使用 OpenEvolve 机制进行搜索：

- 多岛屿并行进化。
- 每代每个岛屿产生候选生成器。
- 使用 v3 operator pool 对 genome 做定向变异。
- 评估结果持久化保存，支持断点续跑。
- 每代保存候选、elite、checkpoint 和可视化所需指标。

当前 v3 operator pool：

| operator | 方向 |
|---|---|
| `op16_v3_blueprint_binding` | 强化目标坐标到人格蓝图的绑定 |
| `op17_v3_axis_coverage_grid` | 强化目标空间覆盖和轴向分辨率 |
| `op18_v3_behavior_alignment_probes` | 强化人格描述到问卷行为的可解释映射 |
| `op19_v3_cross_field_coherence` | 强化跨字段一致性 |
| `op20_v3_realistic_novelty` | 强化真实感与非模板化差异 |
| `op21_v3_schema_precision` | 强化 schema 精准度和字段边界 |

v2 operators 保留为历史对照，但后续主要使用 v3 pool 继续验证。

### 2.5 评估指标

本节只描述当前代码中的真实计算口径。核心实现位置包括：

- `src/mega_persona/evolution.py`：候选生成器评估、最终 `fitness`。
- `src/mega_persona/evaluation.py`：schema / validity / near duplicate / primary-axis diversity。
- `src/mega_persona/consistency.py`：内部一致性与 axis alignment。
- `src/mega_persona/shadow_simulator.py`：人格-问卷行为对齐。
- `src/evaluator/metrics.py`：coverage、avg_dist、min_dist 等多样性指标。

#### 2.5.1 单个 candidate 的评估流程

对每个候选 genome，在每个 seed 上执行：

```text
1. 根据 genome 和 seed 生成 target slots
2. 生成 personas
3. 对 personas 做 schema / validator 评估
4. 对 personas 做内部一致性评估
5. 用 shadow simulator 回答 train + validation shadow surveys
6. 聚合 validation shadow behavior
7. 计算 validation behavior diversity
8. 计算 seed_score
```

如果一个 candidate 有多个 seed，candidate 的最终 `fitness` 是所有 seed 的 `score` 平均值：

```text
candidate_fitness = mean(seed_score for each seed)
```

当前常用实验里只有一个 seed 时，`candidate_fitness` 就等于该 seed 的 `seed_score`。

#### 2.5.2 schema_fitness

`schema_fitness` 来自 `evaluate_mega_personas()`，它不是 LLM 主观打分，而是由 validator、近重复率和 primary-axis 多样性合成。

计算步骤：

```text
validity_rate = valid_count / sample_size
near_duplicate_rate = duplicate_pairs / total_pairs
axis_matrix = primary_axes(valid_personas)
diversity_metrics = DiversityMetrics(coverage_radius).fitness(axis_matrix)
```

其中 `valid_personas` 必须通过：

- Pydantic schema 校验。
- `validate_mega_persona()` 中的硬规则检查。
- 如果出现 severity 为 `error` 的 issue，则该 persona 不计入 valid。

近重复率使用文本 token set 的 Jaccard 相似度：

```text
near_duplicate_rate = proportion(Jaccard(tokens_i, tokens_j) >= duplicate_threshold)
```

`schema_fitness` 的最终公式是：

```text
max_distance = sqrt(dim)
avg_dist_norm = clip(avg_dist / max_distance, 0, 1)
min_dist_norm = clip(min_dist / max_distance, 0, 1)
coverage = clip(coverage, 0, 1)

diversity_score = 0.65 * coverage
                + 0.25 * avg_dist_norm
                + 0.10 * min_dist_norm

schema_fitness = validity_rate
               * (1 - near_duplicate_rate)
               * diversity_score
```

因此，`schema_fitness` 实际上同时惩罚三类问题：

- 生成不合法：`validity_rate` 下降。
- 模板化重复：`near_duplicate_rate` 上升。
- primary axes 覆盖不足：`coverage / avg_dist / min_dist` 较低。

#### 2.5.3 DiversityMetrics 的 coverage / avg_dist / min_dist

`DiversityMetrics` 对输入矩阵 `Z` 计算多样性。`Z` 可以是：

- valid personas 的 primary-axis matrix。
- shadow survey 行为轴 matrix。
- target slots 的 axis matrix。

主要指标口径：

| 指标 | 当前计算方式 |
|---|---|
| `coverage` | 在 `[0,1]^d` 中采样 `1000` 个固定随机参考点，计算这些参考点是否落在任一生成点半径 `coverage_radius` 的球内，返回被覆盖比例 |
| `avg_dist` | 所有样本两两欧氏距离的平均值 |
| `min_dist` | 所有样本两两欧氏距离的最小值 |
| `convex_hull` | 样本点凸包体积；点数不足或计算失败时为 `0` |
| `dispersion` | 固定参考点到最近样本点距离的最大值；越小越好，返回时取负号 |
| `kl_divergence` | 样本经验分布相对单位空间均匀分布的 KL 散度；越小越好，返回时取负号 |

当前进化的 `fitness` 直接使用的是 `validation_behavior_coverage`，不是直接使用全部 6 个 diversity 指标。

#### 2.5.4 内部一致性（internal_consistency）和轴对齐度（axis_alignment）

内部一致性（`internal_consistency`）由 `evaluate_population_consistency()` 计算。它先对每一个人格计算一份人格一致性报告（`PersonaConsistencyReport`），再把所有人格的分数聚合为群体均值和最低值。

单个人格的计算分两步。

第一步，计算轴对齐度（`axis_alignment`）。它衡量生成出的人格轴值是否贴近该人格原本对应的目标槽位（`target slot`）：

```text
axis_alignment = 1 - mean(abs(persona_axis - target_slot_axis))
```

其中：

- 人格轴值（`persona_axis`）：从生成后的人格字段中重新提取出的主轴数值。
- 目标槽位轴值（`target_slot_axis`）：生成前撒点或配额槽给定的目标轴数值。
- 两者越接近，轴对齐度越高。
- 如果没有对应的目标槽位，则轴对齐度记为 `1.0`。

第二步，计算规则一致性分数（`rule_score`）。代码会检查一组跨字段一致性规则，例如：

- 自我调节数值和计划风格（`planning_style`）是否冲突。
- 动机驱动（`motivation drive`）和内在动机（`intrinsic_motivation`）是否冲突。
- 应对方式（`coping_style`）和自我调节能力（`self_regulation`）是否冲突。
- 社交能量（`social_energy`）和协作风格（`collaboration_style`）是否冲突。
- 学业表现区间（`performance_band`）是否有自我调节或动机支撑。
- 压力负荷（`stress_load`）、心理韧性（`resilience`）和恢复模式（`recovery_pattern`）是否合理。

规则惩罚：

```text
error issue   -> penalty 0.08
warning issue -> penalty 0.035
penalty_total = sum(issue penalties)
rule_score = clip(1 - min(0.65, penalty_total), 0, 1)
```

单个人格的一致性分数（`persona_consistency`）由轴对齐度和规则一致性共同决定：

```text
persona_consistency = clip(0.70 * axis_alignment
                         + 0.30 * rule_score,
                         0, 1)
```

也就是说，当前代码中轴对齐度权重为 `70%`，规则一致性权重为 `30%`。这意味着该指标首先关注“生成人格是否贴近目标撒点”，其次关注“人格内部字段是否自洽”。

最后，对整批人格聚合得到群体指标：

```text
internal_consistency = mean(persona_consistency)
internal_consistency_min = min(persona_consistency)
axis_alignment = mean(axis_alignment)
```

其中：

- 内部一致性均值（`internal_consistency`）：所有人格一致性分数的平均值，会进入最终适应度（`fitness`）。
- 内部一致性最低值（`internal_consistency_min`）：整批人格中最差个体的分数，主要用于诊断是否存在少数严重崩坏的人格。
- 轴对齐度均值（`axis_alignment`）：所有人格轴对齐度的平均值，用于观察生成器是否执行了目标撒点。

进入最终适应度（`fitness`）的是内部一致性均值（`internal_consistency`），不是内部一致性最低值（`internal_consistency_min`）。

#### 2.5.5 shadow_alignment

`shadow_alignment` 来自 `aggregate_shadow_behavior()`。

流程：

```text
1. 每个人格回答多个 shadow surveys
2. 每份问卷回答会被转成 axis_scores
3. 对同一 persona 的多个 survey axis_scores 求均值
4. 将 behavior axis mean 与 persona.primary_axes 对比
```

每个轴先计算平均绝对误差：

```text
persona_behavior_mae[axis]
  = mean(abs(persona.primary_axes[axis] - behavior_axis_score[axis]))
```

总体对齐度：

```text
overall_mae = mean(persona_behavior_mae across axes)
shadow_alignment = clip(1 - overall_mae, 0, 1)
```

进化使用的是 `validation_shadow_alignment`，也就是 validation shadow surveys 上的人格-行为对齐度。

#### 2.5.6 behavior_coverage 和 behavior diversity

`behavior_coverage` 来自 validation shadow survey 的行为轴矩阵：

```text
behavior_matrix = shadow_behavior_axis_matrix(personas, validation_simulations)
validation_behavior_diversity = DiversityMetrics(coverage_radius).fitness(behavior_matrix)
validation_behavior_coverage = validation_behavior_diversity["coverage"]
```

也就是说，它衡量的是“问卷行为响应形成的行为轴空间”覆盖了多少，而不是人格文本 primary axes 本身覆盖了多少。

`validation_behavior_avg_dist` 直接来自 `validation_behavior_diversity["avg_dist"]`。

`validation_behavior_balanced_diversity` 是额外的诊断 / elite 指标，公式是：

```text
coverage = clip(metrics["coverage"], 0, 1)
avg_dist = clip(metrics["avg_dist"] / sqrt(3), 0, 1)
min_dist = clip(metrics["min_dist"] / 0.25, 0, 1)
uniformity = clip(exp(metrics["kl_divergence"]), 0, 1)

balanced_diversity = 0.45 * coverage
                   + 0.25 * uniformity
                   + 0.20 * avg_dist
                   + 0.10 * min_dist
```

注意：这里的 `metrics["kl_divergence"]` 已经是 `DiversityMetrics.fitness()` 返回的负 KL，因此 `exp(metrics["kl_divergence"])` 可以理解为把“越接近均匀越好”的信息映射到 `[0,1]`。

#### 2.5.7 最终 evolution fitness

当前进化使用的 `fitness` 不是线性加权求和，而是乘法门控公式：

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

- `schema_fitness`：见 2.5.2。
- `validation_behavior_coverage`：见 2.5.6。
- `validation_shadow_alignment`：见 2.5.5。
- `internal_consistency`：见 2.5.4。
- `generation_rate = 成功生成的人格数量 / 目标 slot 数量`。

这个公式的设计目的，是避免生成器只在单一指标上冲高。例如只提高覆盖度但 schema 崩坏，或者只保持格式合法但行为空间坍缩，都会被乘法结构压低总分。三个 gate 的下限是 `0.5`，这样可以保留进化梯度，不会因为某个软指标偏低就直接变成 0；但 `schema_fitness` 和 `generation_rate` 不设 gate，下限仍然可以直接把无效生成器压到低分。

#### 2.5.8 OpenEvolve 中显示的 elite 指标

OpenEvolve 接收的是一组指标槽，不只有 `global_best`。这些指标来自同一次 candidate 评估：

| OpenEvolve 指标 | 来源 |
|---|---|
| `global_best` | `fitness` / `seed_score` 聚合结果 |
| `coverage_elite` | `validation_behavior_coverage.mean` |
| `alignment_elite` | `validation_shadow_alignment.mean` |
| `consistency_elite` | `internal_consistency.mean` |
| `diversity_elite` | `validation_behavior_balanced_diversity.mean` |
| `schema_elite` | `schema_fitness.mean` |

因此，OpenEvolve 中的 elite 指标和进化评估指标是一致来源，但用途不同：

- `global_best` 是综合选择信号。
- 其他 elite 维度用于保留不同优势方向的候选。
- `diversity_elite` 当前不直接进入最终 `fitness` 公式，但用于保留行为多样性较好的候选。

#### 2.5.9 Train / validation / sealed test 的作用

- `train_shadow_*`：记录训练问卷上的行为表现，主要用于诊断。
- `validation_shadow_*`：进入进化选择和 `fitness` 计算。
- `test_shadow_*` / sealed test：只在最终最佳候选上评估，用于检查泛化，不参与进化选择。

## 3. 当前关键实验记录

### 3.1 实验信息

实验目录：

`data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627`

实验设置：

| 参数 | 值 |
|---|---|
| LLM provider | `deepseek` |
| persona pipeline | `single_call` |
| simulator backend | `student-realistic-v2` |
| operator family | `v3` |
| n | `8` |
| seeds | `17` |
| generations | `10` |
| islands | `8` |
| children per island | `1` |
| shadow surveys | `4` |
| validation shadow surveys | `2` |
| test shadow surveys | `2` |
| items per shadow survey | `6` |

可视化文件：

`data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627/evolution_dashboard.html`

### 3.2 结果摘要

本次共评估 `81` 个候选生成器。最佳候选为：

| 字段 | 值 |
|---|---|
| candidate | `openevolve_000063_8098c1dac245` |
| generation | `8` |
| operator | `op17_v3_axis_coverage_grid` |
| fitness | `0.272785` |

### 3.3 Baseline vs Best

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
| slot_coverage | 0.473000 | 0.473000 | +0.000000 | +0.00% |
| axis_alignment | 0.888959 | 0.934138 | +0.045178 | +5.08% |
| near_duplicate_rate | 0.000000 | 0.000000 | +0.000000 | 0.00% |

### 3.4 分代趋势

| generation | best fitness | mean fitness | best operator |
|---:|---:|---:|---|
| 0 | 0.215046 | 0.215046 | baseline |
| 1 | 0.242925 | 0.229227 | `op21_v3_schema_precision` |
| 2 | 0.257611 | 0.230104 | `op20_v3_realistic_novelty` |
| 3 | 0.254048 | 0.234016 | `op18_v3_behavior_alignment_probes` |
| 4 | 0.262486 | 0.230921 | `op18_v3_behavior_alignment_probes` |
| 5 | 0.257003 | 0.230617 | `op20_v3_realistic_novelty` |
| 6 | 0.249833 | 0.222845 | `op17_v3_axis_coverage_grid` |
| 7 | 0.250371 | 0.228896 | `op21_v3_schema_precision` |
| 8 | 0.272785 | 0.239700 | `op17_v3_axis_coverage_grid` |
| 9 | 0.239495 | 0.227137 | `op16_v3_blueprint_binding` |
| 10 | 0.267455 | 0.235552 | `op20_v3_realistic_novelty` |

### 3.5 Operator 表现

| operator | n | mean fitness | max fitness | mean coverage | mean alignment | mean schema |
|---|---:|---:|---:|---:|---:|---:|
| `op17_v3_axis_coverage_grid` | 13 | 0.236383 | 0.272785 | 0.313000 | 0.821864 | 0.407942 |
| `op20_v3_realistic_novelty` | 20 | 0.221861 | 0.267455 | 0.315450 | 0.817344 | 0.392452 |
| `op18_v3_behavior_alignment_probes` | 10 | 0.243028 | 0.262486 | 0.343400 | 0.828104 | 0.413745 |
| `op19_v3_cross_field_coherence` | 13 | 0.230961 | 0.253413 | 0.318154 | 0.814583 | 0.399811 |
| `op21_v3_schema_precision` | 13 | 0.229613 | 0.250371 | 0.310692 | 0.819338 | 0.405093 |
| `op16_v3_blueprint_binding` | 11 | 0.231287 | 0.250039 | 0.340000 | 0.822595 | 0.395852 |

### 3.6 Sealed Test

当前这次实验是在“完整 sealed test 指标补全”代码修改前完成的，因此 sealed test 里只有行为相关指标：

| 指标 | 值 |
|---|---:|
| test_shadow_alignment.mean | 0.821519 |
| test_behavior_coverage.mean | 0.363000 |
| test_behavior_avg_dist.mean | 0.371456 |

这些指标没有显示明显崩坏。尤其是：

- validation coverage 为 `0.364`，test coverage 为 `0.363`，基本一致。
- validation alignment 为 `0.813770`，test alignment 为 `0.821519`，sealed test 略高。
- validation avg_dist 为 `0.391300`，test avg_dist 为 `0.371456`，test 略低但仍在可接受范围。

但由于旧 run 没有记录 test schema / test consistency / test axis alignment，不能把这次 sealed test 作为完整泛化结论。

## 4. 阶段性判断

这次 v3 pool 实验比之前 v2 / genome v2 的实验更有意义。

主要依据：

1. 多个指标同时上升，而不是只靠 fitness 权重变化制造提升。
2. 覆盖度、schema、consistency、axis alignment 都有可见改善。
3. 行为 alignment 没有明显牺牲，sealed test 行为指标也没有崩坏。
4. 最佳候选来自 `op17_v3_axis_coverage_grid`，说明 v3 算子开始对目标空间覆盖产生有效作用。
5. `op18_v3_behavior_alignment_probes` 的 mean fitness 和 mean alignment 较高，后续值得单独做固定算子实验。

但目前还不能说已经得到最终科学结论，原因是：

- 只有单 seed：`17`。
- n 只有 `8`，属于探索规模。
- sealed test 的完整指标是在本次实验后才补全的。
- alignment 提升很小，说明行为对齐仍是瓶颈。
- 多岛屿随机搜索仍可能产生偶然最优，需要复现实验。

当前合理结论是：

> Genome v3 + v3 operator pool 已经出现可测的进化信号，值得进入复现和放大实验；但还需要多 seed、完整 sealed test 和更大 n 来确认稳定性。

## 5. 后续实验规划

### Stage 0：Genome v3 使用有效性验证

目的：回应“如何确定 Genome v3 蓝图真的被人格生成器使用”这个基础问题。Genome v3 不能只作为理论假设存在，必须通过操作检查和消融实验验证其确实影响生成结果。

核心思路：

```text
不是假定 LLM 会服从 Genome v3
而是检查 Genome v3 的变化是否会稳定改变生成结果和评估指标
```

需要完成三类实验：

1. 操作检查实验  
   固定 target slot，只替换不同 Genome v3，观察生成的人格是否发生系统性变化。  
   如果不同 Genome v3 对同一目标槽位生成出可区分的人格，并且差异体现在 axis alignment、internal consistency、shadow alignment 或 behavior coverage 上，说明 Genome v3 确实被生成器使用。

2. 目标响应实验  
   固定同一个 Genome v3，只改变 target axes，观察生成结果中的 persona axes 是否跟随目标轴变化。  
   如果目标轴改变后，生成后重新提取的人格轴值也随之变化，说明 Genome v3 能把 Monte Carlo 撒点传递到人格生成过程。

3. Genome 模块消融实验  
   分别移除或弱化 Genome v3 中的关键模块，例如：
   - 去掉 `behavior_anchors`，观察 shadow alignment 是否下降。
   - 去掉 `consistency_rules`，观察 internal consistency 是否下降。
   - 去掉 `axis_expression_policy`，观察 axis alignment 和 coverage 是否下降。
   - 去掉 `blueprint_policy`，观察人格字段是否更容易变成松散拼接。

验收标准：

- Genome v3 改变后，生成结果和指标出现方向一致的变化。
- 去掉某个模块后，对应指标出现可解释下降。
- 固定 Genome v3 改变 target axes 时，axis alignment 仍能保持较高水平。
- 这些变化不能只靠单个样本判断，至少需要多个 target slots 或多个 seeds 支撑。

该阶段的作用不是追求最高 fitness，而是证明“Genome v3 是一个有效的、可执行的生成蓝图”。只有这一点成立，后续对 Genome v3 的进化实验才更有理论和实验依据。

### Stage A：复跑同规模，补全完整 sealed test

目的：验证本次好结果是否能在新代码的完整 sealed test 指标下复现。

建议命令：

```bash
python scripts/run_mega_persona_evolution.py \
  --llm-provider deepseek \
  --generator-mode llm \
  --persona-pipeline single_call \
  --n 8 \
  --seeds 17 \
  --generations 10 \
  --num-islands 8 \
  --children-per-island 1 \
  --operator-family v3 \
  --simulator-backend student-realistic-v2 \
  --shadow-surveys 4 \
  --validation-shadow-surveys 2 \
  --test-shadow-surveys 2 \
  --items-per-shadow-survey 6 \
  --extinction-interval 4 \
  --candidate-max-workers 3 \
  --persona-max-workers 4 \
  --shadow-max-workers 24 \
  --output-dir data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_fulltest_20260627
```

可视化命令：

```bash
python scripts/visualize_mega_persona_evolution_dashboard.py \
  --input data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_fulltest_20260627 \
  --output data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_fulltest_20260627/evolution_dashboard.html
```

验收标准：

- best fitness 相比 baseline 有正向提升。
- coverage / schema / consistency 至少两个指标稳定提升。
- sealed test 的 schema、consistency、alignment、coverage 不出现明显回落。

### Stage B：固定单算子实验

目的：验证某一个算子是否真的能稳定改变生成器，而不是算子池随机搜索带来的偶然提升。

优先测试：

1. `op17_v3_axis_coverage_grid`：当前最佳候选来源。
2. `op18_v3_behavior_alignment_probes`：当前 mean fitness 和 alignment 表现较好。
3. `op20_v3_realistic_novelty`：后期 generation 10 的 best 来源。

设计：

- n = 8
- generations = 10
- seeds = 17
- 每次只允许一个 operator
- 对比 baseline、固定 op17、固定 op18、固定 op20、v3 pool

若固定算子也能稳定提升，说明该算子的作用机制更可信。

### Stage C：多 seed 小规模复现

目的：排除单 seed 偶然性。

建议：

- n = 8
- generations = 10
- seeds = 17,23,31
- operator-family = v3
- simulator = student-realistic-v2

重点看：

- best over baseline 的均值提升。
- 不同 seed 下最优 operator 是否集中。
- sealed test 是否稳定。

### Stage D：中规模确认实验

目的：确认 v3 pool 在更大人格数量下仍然有效。

建议：

- n = 16
- generations = 20
- seeds = 17
- operator-family = v3
- shadow surveys 可适当增加到 6
- validation shadow surveys 可增加到 3
- test shadow surveys 可增加到 4

重点看：

- 提升是否仍超过随机波动。
- 行为 alignment 是否仍是瓶颈。
- schema 和 consistency 是否在更大 n 下保持稳定。

### Stage E：生成管线对比

目的：比较 `single_call` 与 multi-agent / five-agent 的取舍。

前提：v3 pool 进化效果已经在 Stage A-D 中稳定。

对比：

- `single_call`：低成本、快速探索。
- `five_agent`：更高文本质量、更强模块控制，但成本高。

不建议太早做这个对比，因为如果进化信号还没有稳定，管线差异会干扰结论。

## 6. 当前优先级

优先级从高到低：

1. 完成 Stage 0，验证 Genome v3 是否真的被生成器使用。
2. 复跑 Stage A，补全完整 sealed test。
3. 做固定单算子实验，确认 op17 / op18 的真实作用。
4. 做多 seed 小规模复现。
5. 再进入 n=16 / gen=20 的中规模实验。
6. 最后再考虑 single_call 和 five_agent 的正式比较。

## 7. 记录规范

后续每次实验建议在本目录新增文件：

```text
YYYY-MM-DD_<short_name>.md
```

每份记录至少包含：

- 实验目的。
- 完整命令。
- output dir。
- baseline vs best 指标表。
- generation trend。
- operator 表现。
- sealed test 结果。
- 是否支持当前假设。
- 下一步决策。

这样可以避免“跑了很多实验但不知道每次为什么跑、结论是什么”的问题。
