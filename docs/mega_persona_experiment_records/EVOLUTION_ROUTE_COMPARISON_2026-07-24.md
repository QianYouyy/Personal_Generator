# MegaPersona 进化路线对比

日期：2026-07-24

## 1. 对比目的

当前项目采用 Genome v4 + OpenEvolve 的路线：通过结构化 genome 控制 persona 的行为生成机制，再根据验证集指标选择 elite。该路线已经解决了 v3 自由文本维度过高、变异难归因的问题，但目前尚未证明能够产生超过评估噪声的稳定收益。

本文比较当前路线与其他可选进化对象，重点回答三个问题：

1. 哪种对象最容易产生可验证的进化效果？
2. 哪种路线最符合“生成高质量、多样化 persona population”的研究目标？
3. 后续应该继续优化 Genome v4，还是调整进化对象？

## 2. 当前路线：结构化 Genome v4

### 进化对象

Genome v4 是一个低维结构化行为生成程序，主要包括：

- `probe_assignment`：目标轴与行为场景的绑定；
- `axis_realization`：轴值的表达方式和信号强度；
- `interaction_mode`：强轴与弱轴之间的关系；
- `echo_graph`：跨字段行为证据传播；
- `context_modulation`：同一机制在不同情境下的变化；
- `repair_control`：证据密度和修复优先级。

`op22` 至 `op27` 每次只修改一个模块。OpenEvolve 维护岛屿与多指标 elite，当前可使用随机算子选择或 MCTS 算子选择。

### 优点

- 维度比 v3 更低，算子作用范围明确；
- genome 可以重复渲染为 generation blueprint；
- 变异记录可审计，便于分析算子与指标之间的关系；
- 可以继续复用现有 OpenEvolve、checkpoint、缓存和可视化体系。

### 当前问题

- genome 对最终 persona 的影响仍然经过 LLM，因果作用可能被模型自身能力覆盖；
- 每个候选需要重新生成完整 population，评估噪声较大；
- 当前最终选择仍以候选级 global fitness 为主，多样性主要通过独立 elite 保存；
- v4 smoke 中观察到的 fitness 提升没有通过重复评估验证。

噪声下限实验中：

| 对象 | 重复 fitness 均值 | 标准差 |
|---|---:|---:|
| v4 smoke selected best | 0.23497 | 0.00922 |
| v4 default seed | 0.23416 | 0.01202 |

两者均值只差 `0.00081`，远小于评估波动。因此，当前结果只能证明 v4 流程可运行，不能证明 v4 已经实现有效进化。

## 3. 不同进化路线对比

| 路线 | 进化对象 | 最终选择单位 | 因果杠杆 | 多样性能力 | 噪声敏感度 | 实现成本 | 论文解释性 |
|---|---|---|---|---|---|---|---|
| 当前 Genome v4 | 结构化生成策略 | 最优 genome | 中 | 中 | 高 | 低，已有实现 | 较强 |
| 目标槽位/课程进化 | axis × context 采样计划 | 最优任务分布 | 强 | 强 | 中 | 中 | 强 |
| Persona Archive / MAP-Elites | persona 或 persona population | 各行为格子的 elite | 最强 | 最强 | 中 | 中高 | 很强 |
| 生成流程进化 | generator/critic/repair 执行图 | 最优流程控制器 | 强 | 中 | 中高 | 高 | 强 |
| 示例库进化 | few-shot 示例集合 | 最优示例库 | 中强 | 中 | 高 | 中 | 中 |
| 生成器-挑战器共进化 | 生成策略与压力场景 | 鲁棒生成器 | 强 | 中强 | 高 | 很高 | 很强，但复杂 |
| 算子自身进化 | mutation operator/instruction | 最优变异策略 | 间接 | 取决于底层对象 | 很高 | 高 | 中 |

## 4. 备选路线分析

### 4.1 目标槽位/课程进化

固定 persona generator，进化“下一批应该生成什么”，例如：

- 目标 axis 组合；
- strongest/weakest tension；
- deadline、peer pressure、feedback 等行为情境；
- 各类组合的采样比例。

对应算子可以是：

- `fill_undercovered_cell`：填补低覆盖行为区域；
- `stress_axis_boundary`：增加边界轴值样本；
- `increase_context_contrast`：增强跨情境行为对比；
- `repair_target_mismatch`：针对目标轴偏差重新采样；
- `sample_rare_interaction`：增加稀有轴交互。

与 Genome v4 相比，它不需要依赖 LLM 理解一套间接 blueprint 才能发挥作用。采样计划会直接改变目标 slot，因此因果链更短，也更容易形成清晰的 coverage 和 axis error 改进曲线。

局限是它主要优化“生成什么”，未必能改善 persona 文本本身的结构质量和跨字段一致性。

### 4.2 Persona Archive / MAP-Elites

直接把 persona 或 persona population 作为搜索对象。行为特征空间划分为多个格子，每个格子保存该区域内质量最好的 persona。

可能的行为描述维度包括：

- strongest axis role；
- weakest axis role；
- context sensitivity；
- behavior coverage bucket；
- strict consistency bucket。

变异不再修改抽象 genome，而是执行：

- 为未覆盖格子定向生成 persona；
- 替换近重复 persona；
- 对单个 persona 做局部行为机制改写；
- 保留验证稳定且行为不同的 persona；
- 从多个格子的 elite 组合最终 population。

这条路线与项目目标最一致，因为最终研究对象本来就是 persona population，而不是一个生成参数。它也不要求所有目标压缩成单一 global fitness。MAP-Elites 的核心目标正是在预定义行为空间中保留多个高质量且不同的解：[Mouret and Clune, 2015](https://arxiv.org/abs/1504.04909)。

主要成本是需要新增行为网格、archive 更新和 population 组装逻辑。还需要明确 persona-level quality 与 archive-level coverage 的边界，避免把测试集用于 archive 更新。

### 4.3 生成流程进化

把候选定义为有限的生成流程图：

```text
生成若干草稿
→ axis critic
→ behavior critic
→ 候选选择
→ 定向 repair
```

可进化内容包括：

- 草稿数量；
- critic 类型和顺序；
- repair 触发条件；
- repair 字段范围；
- 候选选择规则；
- 质量与 API 成本的权衡。

与 Genome v4 相比，这类变化对生成结果有更直接的机制作用，并且容易通过流程消融解释。但每次评估可能使用不同数量的 API 调用，因此 fitness 必须同时报告质量和成本，不能只比较最终分数。

### 4.4 示例库进化

候选是固定大小的 few-shot persona、行为证据或失败反例集合。每个算子只增加、删除或替换一个示例。

它比自由文本 prompt genome 更容易影响模型输出，也比较容易实现。但示例可能造成风格模仿、训练场景过拟合，且不同示例之间的作用仍可能难以归因。Promptbreeder 展示了进化 task prompt 与 mutation prompt 的可行性，但本项目更适合限制为结构化行为示例，而不是重新开放无限文本字段：[Fernando et al., 2023](https://arxiv.org/abs/2309.16797)。

### 4.5 生成器-挑战器共进化

同时维护两个种群：

- 生成器：产生 persona 或生成策略；
- 挑战器：产生能暴露 persona 行为矛盾的训练场景。

挑战器寻找跨情境不一致、目标轴错位和行为预测失败；生成器学习在这些压力场景下保持一致。该路线类似自动课程和环境设计，可以产生逐渐增强的训练难度：[Dennis et al., 2020](https://arxiv.org/abs/2012.02096)。

这条路线适合后续形成更强的论文贡献，但不适合作为当前第一步。挑战场景必须限制在训练集，validation/test 继续冻结，否则会产生严重的数据泄漏和指标博弈。

## 5. 当前路线与推荐路线的关键差异

| 问题 | Genome v4 | 课程进化 | Persona Archive |
|---|---|---|---|
| 进化改变什么 | 生成机制参数 | 目标任务分布 | 最终 persona 集合 |
| 与最终产物距离 | 两层：genome → LLM → persona | 一层：slot → persona | 直接作用于 persona |
| 平台期可能来源 | LLM 忽略或吸收 blueprint 差异 | 目标空间已覆盖 | 行为网格已填满或局部质量饱和 |
| 多样性如何保留 | 独立 diversity elite | 持续采样低覆盖区域 | 每个行为格子独立保存 elite |
| 是否依赖单一 global best | 是，最终仍选一个 genome | 可以不依赖 | 不依赖 |
| 算子效果是否容易解释 | 中等 | 容易 | 容易解释为填格/替换/局部改写 |
| 最适合回答的问题 | 生成机制能否被进化优化 | 采样课程能否改善覆盖 | 能否产生多样且高质量的 persona population |

## 6. 推荐方案

### 短期：保留 Genome v4 作为对照

Genome v4 不应立即删除。它已经形成一个低维、可审计的机制级基线，适合作为后续路线的对照组。

但在继续多代搜索前，应先完成 `op22` 至 `op27` 的固定父代、单步、重复评估筛选。只有算子目标指标变化超过噪声，才有理由继续 Genome v4 的多代进化或 MCTS。

### 中期：优先实现目标槽位/课程进化

这是当前成本最低、因果链最清楚的替代路线。它可以继续复用现有 slot、generator、validator 和 shadow survey，只需要更换候选表示和变异算子。

建议先回答：动态采样课程能否在相同 API 预算下，显著提高 coverage、balanced diversity，并降低 axis target MAE？

### 主线：引入 Persona Archive / MAP-Elites

如果论文的核心目标是生成“高质量、多样化 persona population”，推荐最终采用 Persona Archive 作为主搜索结构：

```text
课程策略提出低覆盖目标
→ 固定 generator 生成 persona
→ validation 评估质量与行为位置
→ MAP-Elites 更新对应行为格子
→ 从多个格子组装最终 population
→ 只对最终冻结 population 运行 sealed test
```

这个组合中：

- 课程进化负责决定“接下来探索哪里”；
- MAP-Elites 负责决定“哪些不同 persona 值得保留”；
- Genome v4 可以作为可选 generator policy，而不再承担全部进化任务。

## 7. 建议实验顺序

| 阶段 | 实验 | 目的 | 继续条件 |
|---|---|---|---|
| E0 | v4 六个固定算子单步筛选，repeats ≥ 3 | 验证当前算子是否有因果作用 | 至少两个算子的目标指标超过噪声 |
| E1 | 固定采样 vs 课程进化 | 验证目标分布进化是否有效 | coverage/axis MAE 在相同预算下稳定改善 |
| E2 | global-best selection vs Persona Archive | 验证 archive 是否降低多样性损失 | archive coverage、QD-score 和 sealed test 不下降 |
| E3 | 课程进化 + Persona Archive | 验证两者交互 | 多 seed 下效果可重复 |
| E4 | 随机算子 vs MCTS | 只验证搜索策略 | 底层算子已被 E0/E1 证明有效 |
| E5 | 生成器-挑战器共进化 | 扩展鲁棒性研究 | 前述主线稳定后再进行 |

每个阶段只改变一个核心机制。不要同时修改 genome、archive、reward、parent selection 和 MCTS，否则即使结果提高，也无法判断提升来自哪里。

## 8. 结论

当前 Genome v4 路线在工程上自洽，但实验上仍缺少超过噪声的有效性证据。它适合保留为机制级对照，不适合在现阶段继续直接扩大到长代数实验。

推荐优先级为：

1. **目标槽位/课程进化**：最容易验证，适合作为下一项代码改动；
2. **Persona Archive / MAP-Elites**：最符合最终研究目标，适合作为论文主线；
3. **生成流程进化**：适合作为机制消融或后续增强；
4. **示例库进化与共进化**：作为扩展研究，不作为当前第一步；
5. **MCTS 和算子自身进化**：应建立在底层进化对象已经产生稳定信号之后。

综合来看，最值得推进的整体路线是：

> **课程进化负责探索行为空间，Persona Archive 负责保存不同区域的高质量 persona，Genome v4 退回为可选生成策略和对照条件。**
