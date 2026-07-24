# MegaPersona-Evolve：面向大人格生成的结构化基因组进化方法

论文草稿 v0.1  
日期：2026-06-29  
状态：内部讨论稿，部分引用与消融实验待补充

## 摘要

大语言模型可以生成丰富的人格文本，但在“大人格”场景中，生成结果往往同时面临三个问题：人格维度覆盖不足、跨字段一致性不稳定，以及文本人格与后续行为问卷响应之间缺少可验证联系。本文提出 MegaPersona-Evolve，一个面向大人格生成的结构化生成器进化框架。该框架不直接进化项目源码，也不只进化单句 prompt，而是引入 Genome v3 作为人格生成器的可配置蓝图，将目标空间绑定、轴向表达、行为锚点、一致性规则、结构约束和多样性策略显式组织为可变异对象。系统使用 OpenEvolve 多岛屿进化机制，在固定评估流程下对 Genome v3 进行搜索，并通过 schema 合法性、内部一致性、行为覆盖度、人格-问卷行为对齐度和多样性指标联合评估候选生成器。

在当前探索实验中，我们使用 DeepSeek 模型、single-call 人格生成管线、student-realistic-v2 行为模拟器，在 `n=8`、单 seed、v3 算子池、10 代进化设置下评估 81 个候选生成器。实验结果显示，最佳候选相对 baseline 的综合 fitness 从 0.215046 提升至 0.272785，提升 26.85%；验证集行为覆盖度从 0.290000 提升至 0.364000，提升 25.52%；schema fitness 从 0.384090 提升至 0.452756，提升 17.88%；内部一致性从 0.917959 提升至 0.948271，提升 3.30%。继续续跑至 checkpoint generation 18 后，当前最佳 fitness 达到 0.278084，较 generation 10 最佳结果进一步提升 1.94%，但后续代数受网络失败影响，尚不能作为完整收敛实验结论。本文的初步结果表明，Genome v3 比早期直接修改宽泛 prompt 的方式更容易产生可观测的进化收益；但要证明该方法的稳定有效性，还需要补充多 seed 复现、Genome 模块消融、固定算子对照、完整 sealed test 和人工质量评估。

关键词：大人格生成；大语言模型；进化计算；OpenEvolve；结构化 prompt；行为模拟；人格一致性

## 1. 引言

人格生成是大语言模型应用中的重要基础能力。相比简单的角色卡或短人格描述，“大人格”需要同时包含人口统计背景、思维方式、动机结构、自我调节、价值观、社交创造力、心理健康和行为倾向等多个层面。这类人格如果只依赖单次自由生成，容易出现两类问题：一是生成结果集中在少数常见模板上，难以覆盖目标人格空间；二是不同字段之间互相冲突，例如学习动机、压力反应和社交行为描述并不像同一个人。

现有人格生成方法通常强调 prompt 设计、schema 约束或多 agent 分工。它们可以提升单个人格的文本质量，却不一定能解决群体层面的覆盖问题。另一方面，进化计算和自动程序优化方法可以通过多代搜索优化候选生成器，但如果直接让模型修改源码，候选之间容易变得不可比较，甚至可能通过绕过评估器获得虚高分数。

本文关注的问题是：能否构造一种既可进化、又可解释、还便于科学评估的大人格生成器表示？为此，我们提出 Genome v3。Genome v3 不是最终人格文本，也不是任意源码补丁，而是介于目标空间和 LLM 生成之间的结构化生成蓝图。它规定生成器如何理解目标坐标、如何表达人格轴、如何组织跨字段因果一致性，以及如何让人格描述映射到行为问卷响应。

本文的主要贡献包括：

1. 提出一种用于大人格生成器进化的结构化表示 Genome v3，将人格生成中的目标绑定、轴向表达、行为锚点和一致性规则显式纳入可进化对象。
2. 构建一个基于 OpenEvolve 的多岛屿进化流程，在固定评估器和固定数据切分下搜索更优人格生成器。
3. 设计一组联合评估指标，同时考察 schema 合法性、内部一致性、行为覆盖度、人格-行为对齐度和多样性。
4. 基于当前探索实验给出初步证据：v3 算子池可以在小规模实验中带来可观测的 fitness 和覆盖度提升。

本文仍处于方法验证阶段。所有实验结论均应理解为“当前项目内的探索性证据”，而非最终充分验证结果。

## 2. 相关工作与理论基础

### 2.1 大语言模型人格生成

LLM 人格生成通常通过自然语言 prompt、结构化 schema 或多 agent 流程实现。结构化 schema 可以提升字段完整性，多 agent 流程可以将复杂人格拆分为多个子模块，但二者都需要解决跨模块一致性问题。本文延续结构化人格生成的思路，但将生成策略本身作为可进化对象，而不只是手工固定 prompt。

引用待补：

- LLM persona generation / simulated agents 相关工作。
- HACHIMI 或类似大人格、多 agent 人格生成相关文献。
- 结构化 persona 与行为模拟相关文献。

### 2.2 进化计算作为生成器优化框架

进化计算的核心思想是将候选解编码为 genotype，通过变异、选择和评估逐步搜索高 fitness 解。在本文中，Genome v3 扮演 genotype，实际生成的人格群体及其行为响应扮演 phenotype，schema、coverage、alignment 和 consistency 指标构成环境反馈。

这种设计的理论动机在于：如果直接优化最终文本，搜索空间会过于离散且难以复用；如果直接优化源码，搜索空间又过宽且容易破坏实验可控性。Genome v3 是一种间接编码，它不逐字指定输出，而是指定生成机制和约束。间接编码常用于复杂系统搜索，因为它可以在较小的表示空间中诱导较丰富的表型变化。

引用待补：

- 进化计算、遗传算法、间接编码相关文献。
- OpenEvolve / AlphaEvolve 类自动优化方法相关文献。
- MCTS 或 bandit-style operator selection 相关文献。

### 2.3 为什么需要验证 Genome v3 是否被使用

一个关键质疑是：即使 Genome v3 被写进 prompt 或配置中，如何保证人格生成器真的使用了它？这个问题不能只靠直觉回答，必须通过实验验证。本文目前采用两类证据：

1. 间接性能证据：使用 v3 算子池后，fitness、覆盖度、schema fitness 和内部一致性相对 baseline 出现提升。
2. 待补直接操控证据：固定 target slot，改变 Genome v3 中特定模块，检查输出人格和指标是否发生方向一致的变化。

因此，当前论文草稿中只能说 Genome v3 “在探索实验中显示出有效信号”，不能声称它已经被充分证明为因果来源。后续必须增加 Genome 模块消融和目标响应实验。

## 3. 方法

### 3.1 问题定义

给定一个目标人格空间 \(\mathcal{Z}\)，系统需要生成 \(N\) 个大人格：

\[
P = \{p_1, p_2, \ldots, p_N\}
\]

每个人格 \(p_i\) 包含结构化字段和长文本字段，并可被映射到一组主要人格轴：

\[
z_i = \psi(p_i)
\]

其中 \(\psi\) 是轴提取器。实验目标不是最大化单个人格文本的主观质量，而是在保证合法性和一致性的前提下，使生成群体覆盖目标空间，并在 shadow survey 中产生与人格描述一致的行为响应。

### 3.2 目标空间与初始化

当前系统使用配额槽和连续坐标共同定义目标空间。配额槽保证群体结构可控；连续坐标在 primary axes 上进行 Monte Carlo 或 quasi-random 式撒点，以减少模板集中。当前实验关注的主要轴包括认知抽象、动机自主性和自我调节韧性等学生人格相关维度。

目标空间初始化可表示为：

\[
s_i = (q_i, x_i)
\]

其中 \(q_i\) 是配额槽，\(x_i\) 是连续目标坐标。生成器需要根据 \(s_i\) 生成人格，而不是自由发挥。

### 3.3 Genome v3：可进化的人格生成器蓝图

Genome v3 是本文方法的核心。它不是最终人格文本，也不是项目源码，而是一组结构化生成策略。一个简化的 Genome v3 可表示为：

```json
{
  "blueprint_policy": {
    "axis_evidence_rule": "For each primary axis, name one observable school-life trigger and one boundary condition.",
    "core_tension_rule": "Express target axis combinations as a concrete trade-off in a specific scenario.",
    "persona_coherence_rule": "All sections must describe the same student and reuse the same trigger-behavior evidence."
  },
  "axis_expression_policy": {
    "high": "Include an advantage and a boundary where the high trait becomes a liability.",
    "mid": "Describe two contexts where the trait appears differently because of a named trigger.",
    "low": "Include a visible cost and a limited workaround that does not fully compensate."
  },
  "behavior_anchors": {
    "learning": "Trigger: ambiguous assignment; response: planning, asking, or experimenting.",
    "motivation": "Trigger: conflict between external expectation and interest; response: compromise, persistence, or avoidance.",
    "stress_recovery": "Trigger: deadline or failed attempt; response: latency, coping routine, and later adjustment.",
    "belonging": "Trigger: group task or teacher feedback; response: help-seeking, withdrawal, or contribution."
  },
  "consistency_rules": [
    "Cognition, motivation, self-regulation, social behavior, and health must describe the same student.",
    "A limitation in one field must have a plausible echo in at least one other field.",
    "Contrasts are allowed only when a context boundary explains why behavior changes."
  ]
}
```

Genome v3 的作用链条为：

```text
target slot / Monte Carlo 坐标
        ↓
Genome v3 解释目标坐标和约束
        ↓
生成 persona blueprint 与字段写作策略
        ↓
LLM 生成完整大人格
        ↓
validator / consistency / shadow survey 评估
        ↓
fitness 反馈给 OpenEvolve
        ↓
operator 继续修改 Genome v3
```

选择 Genome v3 而不是直接进化源码，主要基于三个考虑：

1. 可控性：固定评估流程、validator 和数据持久化逻辑，避免候选生成器通过改变评估系统获得虚高分数。
2. 可比性：不同候选之间只在生成策略层面变化，fitness 差异更容易解释。
3. 可消融性：可以单独移除或替换目标绑定、行为锚点、一致性规则等模块，评估各模块贡献。

### 3.4 进化算子

本文当前使用 v3 算子池，每个算子对应一种 Genome v3 变异方向：

| 算子 | 主要方向 |
|---|---|
| `op16_v3_blueprint_binding` | 强化目标坐标到人格蓝图的绑定 |
| `op17_v3_axis_coverage_grid` | 强化目标空间覆盖和轴向分辨率 |
| `op18_v3_behavior_alignment_probes` | 强化人格描述到问卷行为的映射 |
| `op19_v3_cross_field_coherence` | 强化跨字段一致性 |
| `op20_v3_realistic_novelty` | 强化真实感与非模板化差异 |
| `op21_v3_schema_precision` | 强化 schema 精准度和字段边界 |

早期 v2 算子保留为历史对照，但当前探索实验主要使用 v3 算子池。

### 3.5 OpenEvolve 搜索流程

系统使用 OpenEvolve 多岛屿搜索。每个岛屿在每代产生候选 Genome，候选经 LLM mutator 变异后进入完整评估流程。评估完成后，系统保存候选、fitness、elite、checkpoint 和可视化数据。

当前实验配置为：

| 参数 | 值 |
|---|---:|
| LLM provider | DeepSeek |
| persona pipeline | `single_call` |
| persona model | `deepseek-v4-flash` |
| mutator model | `deepseek-v4-pro` |
| simulator backend | `student-realistic-v2` |
| n | 8 |
| seed | 17 |
| generations | 20 目标，已完整分析前 10 代，续跑至 checkpoint gen18 |
| num islands | 8 |
| children per island | 1 |
| operator family | v3 |
| train shadow surveys | 4 |
| validation shadow surveys | 2 |
| test shadow surveys | 2 |
| items per shadow survey | 6 |

### 3.6 Shadow survey 行为评估

每个人格会在 shadow surveys 上被模拟作答。模拟器根据人格描述和问卷题目生成回答，再将回答映射为行为轴分数。系统用这些行为轴分数计算：

1. 人格-行为对齐度：人格轴与行为回答轴之间的一致性。
2. 行为覆盖度：生成群体在行为响应空间中的覆盖范围。
3. 行为多样性：行为向量之间的平均距离、最小距离和均匀性。

当前实验使用 `student-realistic-v2` 模拟器。该模拟器不是外部真实学生数据，而是当前项目内部设计的学生行为模拟器。因此，本文结果应解释为模拟环境下的算法信号，而非真实人类行为验证。

### 3.7 评估指标与 fitness

本文只使用当前代码中真实存在的指标口径。

#### 3.7.1 Schema fitness

Schema fitness 由合法率、重复率和 primary-axis 多样性共同决定：

\[
F_{schema} = validity\_rate \times (1 - near\_duplicate\_rate) \times diversity\_score
\]

其中：

\[
diversity\_score = 0.65 \times coverage + 0.25 \times avg\_dist\_norm + 0.10 \times min\_dist\_norm
\]

#### 3.7.2 内部一致性

内部一致性由轴对齐和规则得分组成：

\[
axis\_alignment = 1 - mean(|persona\_axis - target\_slot\_axis|)
\]

规则扣分中，`error` 扣 0.08，`warning` 扣 0.035。规则得分为：

\[
rule\_score = clip(1 - min(0.65, penalty\_total), 0, 1)
\]

单个人格一致性为：

\[
persona\_consistency = clip(0.70 \times axis\_alignment + 0.30 \times rule\_score, 0, 1)
\]

群体内部一致性取平均值，同时记录最小值和 axis alignment。

#### 3.7.3 行为覆盖度与行为多样性

行为覆盖度来自 validation shadow behavior axis matrix。系统在 \([0,1]^d\) 中采样固定参考点，并计算参考点是否被行为向量以设定半径覆盖。

平衡多样性定义为：

\[
balanced\_diversity =
0.45 \times coverage
+ 0.25 \times uniformity
+ 0.20 \times avg\_dist
+ 0.10 \times min\_dist
\]

其中各项经过归一化或裁剪。

#### 3.7.4 人格-行为对齐度

对每个人格，系统将 shadow survey 回答转换为行为轴分数，并与人格 primary axes 比较。平均绝对误差为：

\[
overall\_mae = mean(persona\_behavior\_mae)
\]

人格-行为对齐度为：

\[
shadow\_alignment = clip(1 - overall\_mae, 0, 1)
\]

#### 3.7.5 综合 fitness

单个 seed 的综合得分为：

\[
behavior\_gate = 0.5 + 0.5 \times clip(validation\_behavior\_coverage, 0, 1)
\]

\[
alignment\_gate = 0.5 + 0.5 \times clip(validation\_shadow\_alignment, 0, 1)
\]

\[
consistency\_gate = 0.5 + 0.5 \times clip(internal\_consistency, 0, 1)
\]

\[
seed\_score =
schema\_fitness \times consistency\_gate
\times behavior\_gate \times alignment\_gate
\times generation\_rate
\]

候选生成器的最终 fitness 是所有 seed 的平均值。当前实验只有一个 seed，因此候选 fitness 等于该 seed 的得分。

反馈给 OpenEvolve 的指标包括：

| OpenEvolve 指标 | 来源 |
|---|---|
| `global_best` | 综合 fitness |
| `coverage_elite` | validation behavior coverage |
| `alignment_elite` | validation shadow alignment |
| `consistency_elite` | internal consistency |
| `diversity_elite` | validation behavior balanced diversity |
| `schema_elite` | schema fitness |

其中 `global_best` 是主选择指标，其他指标用于分维度 elite 记录和分析。

## 4. 实验设计

### 4.1 实验目标

当前实验主要回答三个问题：

1. Genome v3 算子池是否比 baseline 产生更高综合 fitness？
2. fitness 提升是否来自多个指标的共同变化，而不是单一随机波动？
3. 哪些 v3 算子方向在当前任务中更有潜力？

### 4.2 实验设置

主实验路径：

```text
data/results/mega_persona_v3_pool_single_call_deepseek_n8_g10_survey4_20260627
```

关键设置：

- `--llm-provider deepseek`
- `--persona-pipeline single_call`
- `--n 8`
- `--seeds 17`
- `--num-islands 8`
- `--children-per-island 1`
- `--operator-family v3`
- `--simulator-backend student-realistic-v2`
- `--shadow-surveys 4`
- `--validation-shadow-surveys 2`
- `--test-shadow-surveys 2`
- `--items-per-shadow-survey 6`

实验一共完整分析前 10 代，共评估 81 个候选。后续曾在同一实验上继续运行至 checkpoint generation 18，但由于后续出现网络失败，本文将 gen18 作为补充观察，而不作为完整收敛结论。

### 4.3 数据切分

系统在实验初始化时冻结 train、validation 和 test shadow surveys，并记录哈希：

| split | hash |
|---|---|
| train | `316757ac86c1d3f1165d6e5f5f21f26276709ea559ae190869abf71bba18ffd5` |
| validation | `90324dfbecaf7fb25c312a6c4d897f846b5791cc4ee13da0c8cb25acea4febb6` |
| test | `ce2e2a7a799badcc96b4de0422a74c64fc8e88543596831a03787e9510dacce0` |

train 和 validation 用于进化评估，test 用于 sealed test。当前 sealed test 指标仍较少，后续需要补全 schema、consistency、balanced diversity 等完整测试指标。

## 5. 实验结果

### 5.1 最佳候选与 baseline 对比

前 10 代实验中，最佳候选为：

```text
openevolve_000063_8098c1dac245
generation = 8
operator = op17_v3_axis_coverage_grid
fitness = 0.272785
```

baseline 为：

```text
openevolve_000001_bd01ee449da0
fitness = 0.215046
```

主要指标变化如下：

| 指标 | baseline | best | 变化量 | 变化比例 |
|---|---:|---:|---:|---:|
| fitness | 0.215046 | 0.272785 | +0.057739 | +26.85% |
| schema fitness | 0.384090 | 0.452756 | +0.068666 | +17.88% |
| validity | 1.000000 | 1.000000 | +0.000000 | +0.00% |
| internal consistency | 0.917959 | 0.948271 | +0.030312 | +3.30% |
| consistency min | 0.852293 | 0.899674 | +0.047381 | +5.56% |
| behavior coverage | 0.290000 | 0.364000 | +0.074000 | +25.52% |
| shadow alignment | 0.810335 | 0.813770 | +0.003435 | +0.42% |
| balanced diversity | 0.230788 | 0.250239 | +0.019451 | +8.43% |
| avg distance | 0.360622 | 0.391300 | +0.030678 | +8.51% |
| slot coverage | 0.473000 | 0.473000 | +0.000000 | +0.00% |
| axis alignment | 0.888959 | 0.934138 | +0.045178 | +5.08% |
| near duplicate rate | 0.000000 | 0.000000 | +0.000000 | +0.00% |

结果显示，最佳候选的提升主要来自 schema fitness、行为覆盖度、axis alignment 和多样性指标，而 shadow alignment 只出现小幅变化。这说明当前 v3 算子更明显地改善了“覆盖与结构质量”，但对“人格-问卷行为对齐”的提升仍有限。

### 5.2 代际变化

各代最佳 fitness 如下：

| generation | candidates | best fitness | mean fitness | best operator |
|---:|---:|---:|---:|---|
| 0 | 1 | 0.215046 | 0.215046 | baseline |
| 1 | 8 | 0.242925 | 0.229227 | `op21_v3_schema_precision` |
| 2 | 8 | 0.257611 | 0.230104 | `op20_v3_realistic_novelty` |
| 3 | 8 | 0.254048 | 0.234016 | `op18_v3_behavior_alignment_probes` |
| 4 | 8 | 0.262486 | 0.230921 | `op18_v3_behavior_alignment_probes` |
| 5 | 8 | 0.257003 | 0.230617 | `op20_v3_realistic_novelty` |
| 6 | 8 | 0.249833 | 0.222845 | `op17_v3_axis_coverage_grid` |
| 7 | 8 | 0.250371 | 0.228896 | `op21_v3_schema_precision` |
| 8 | 8 | 0.272785 | 0.239700 | `op17_v3_axis_coverage_grid` |
| 9 | 8 | 0.239495 | 0.227137 | `op16_v3_blueprint_binding` |
| 10 | 8 | 0.267455 | 0.235552 | `op20_v3_realistic_novelty` |

从 generation 1 开始，几乎每代最佳候选都超过 baseline，说明 v3 算子池在当前设置下产生了稳定的高分候选。但 mean fitness 波动较小，说明进化并未让整个候选分布持续抬升，而是通过多岛搜索发现少数较优候选。这是后续需要优化的方向。

### 5.3 算子表现

不同算子的平均和最佳表现如下：

| operator | n | mean fitness | max fitness | mean coverage | mean alignment | mean schema |
|---|---:|---:|---:|---:|---:|---:|
| `op17_v3_axis_coverage_grid` | 13 | 0.236383 | 0.272785 | 0.313000 | 0.821864 | 0.407942 |
| `op20_v3_realistic_novelty` | 20 | 0.221861 | 0.267455 | 0.315450 | 0.817344 | 0.392452 |
| `op18_v3_behavior_alignment_probes` | 10 | 0.243028 | 0.262486 | 0.343400 | 0.828104 | 0.413745 |
| `op19_v3_cross_field_coherence` | 13 | 0.230961 | 0.253413 | 0.318154 | 0.814583 | 0.399811 |
| `op21_v3_schema_precision` | 13 | 0.229613 | 0.250371 | 0.310692 | 0.819338 | 0.405093 |
| `op16_v3_blueprint_binding` | 11 | 0.231287 | 0.250039 | 0.340000 | 0.822595 | 0.395852 |

从当前数据看，`op17_v3_axis_coverage_grid` 产生了全局最佳候选，说明目标覆盖和轴向分辨率是当前阶段的重要优化方向。`op18_v3_behavior_alignment_probes` 的平均 fitness 和平均 coverage 较高，说明行为映射类算子有潜力，但还需要更大样本验证。

### 5.4 Sealed test

当前 sealed test 结果如下：

| 指标 | 值 |
|---|---:|
| test shadow alignment mean | 0.821519 |
| test behavior coverage mean | 0.363000 |
| test behavior avg distance mean | 0.371456 |

由于当前 sealed test 只记录了部分指标，且标准差为 0.0 反映当前只有单 seed 或单汇总样本，因此 sealed test 不能作为强结论。后续需要补全 test schema fitness、test internal consistency、test balanced diversity，并采用多 seed 重复实验。

### 5.5 续跑观察

同一实验后续继续运行至 checkpoint generation 18。checkpoint 中的当前最佳候选来自 generation 13：

```text
candidate = openevolve_000104_480fcc970873
operator = op18_v3_behavior_alignment_probes
fitness = 0.278084
```

相对 generation 10 最佳候选，续跑最佳结果变化如下：

| 指标 | gen10 best | gen18 checkpoint best | 变化比例 |
|---|---:|---:|---:|
| global best | 0.272785 | 0.278084 | +1.94% |
| coverage elite | 0.364000 | 0.401000 | +10.16% |
| alignment elite | 0.813770 | 0.842524 | +3.53% |
| consistency elite | 0.948271 | 0.936747 | -1.22% |
| diversity elite | 0.250239 | 0.328854 | +31.42% |
| schema elite | 0.452756 | 0.444982 | -1.72% |

续跑显示 coverage、alignment 和 diversity 仍有继续提升空间，但 consistency 和 schema 出现小幅回落。这提示后续实验不应只追求单一 fitness 上升，还需要监控不同指标之间的 trade-off。

## 6. 讨论

### 6.1 Genome v3 是否有效

当前实验支持一个谨慎结论：Genome v3 在小规模探索实验中比 baseline 表现更好，并且提升并非只来自单一指标。fitness、schema fitness、coverage、axis alignment 和 balanced diversity 均有正向变化。这说明将生成器进化对象从宽泛 prompt 升级为结构化 Genome v3 是有意义的。

但是，当前证据还不足以证明 Genome v3 是提升的唯一原因。因为本实验仍存在单 seed、小样本、模拟器依赖和 sealed test 不完整等限制。要进一步确认，需要做以下消融：

1. 移除 Genome v3 中的 `behavior_anchors`，观察 shadow alignment 是否下降。
2. 移除 `consistency_rules`，观察 internal consistency 是否下降。
3. 弱化 `axis_expression_policy`，观察 coverage 和 axis alignment 是否下降。
4. 固定算子、固定 seed，多次复现实验，观察提升是否稳定。
5. 使用相同预算比较 v2 算子池、v3 算子池和无进化 baseline。

### 6.2 单次 LLM 生成是否合理

当前实验使用 `single_call` 管线，即一次 LLM 调用生成完整人格。这一选择适合探索阶段，因为它显著减少调用成本，也减少多 agent 管线的累计噪声。对于验证“进化算子是否能改变生成器行为”这一问题，single-call 更容易分析。

但 single-call 也可能限制文本质量和细粒度可控性。后续在确认 v3 进化有效后，可以比较：

1. single-call Genome v3。
2. two-stage blueprint + realization。
3. five-agent modular generation。

该对比应在相同目标 slot、相同 Genome、相同 evaluation split 下进行。

### 6.3 Fitness 是否足够

当前 OpenEvolve 主要根据 `global_best` 选择候选，同时记录 coverage、alignment、consistency、diversity 和 schema 等 elite 指标。这一设计适合早期搜索，因为需要一个明确主目标。但从续跑观察看，fitness 上升可能伴随 consistency 或 schema 小幅下降。因此，后续可以继续使用一个主 fitness，但必须在报告中同时展示分项指标，并设置最低质量门槛，例如 validity rate 和 internal consistency 不得低于阈值。

### 6.4 MCTS 算子选择是否必要

当前项目中新增的 Monte Carlo Tree Search 主要用于算子选择，而不是目标空间撒点。它与早期随机或轮转选择算子的区别在于，MCTS 会根据历史收益倾向于选择更有潜力的算子序列，同时保留探索。该模块目前应视为可选实验变量，不应与 Genome v3 主结果混在一起解释。后续需要专门设计 MCTS vs non-MCTS 的消融实验。

## 7. 局限性

本文当前结果存在以下局限：

1. 单 seed：当前主实验使用 seed 17，不能充分证明统计稳定性。
2. 样本较小：`n=8` 适合探索，但不足以代表大规模人格空间。
3. sealed test 不完整：当前 test 只包含部分行为指标，缺少完整 schema、consistency 和 balanced diversity。
4. 模拟器依赖：student-realistic-v2 是项目内部模拟器，尚未与真实学生数据或人工标注对齐。
5. 直接因果证据不足：尚未完成 Genome v3 模块消融，不能完全排除其他因素导致提升。
6. 引用待补：相关工作部分需要补充真实文献，不应在未核对前加入具体引用。

## 8. 结论

本文提出 MegaPersona-Evolve，一个基于 Genome v3 和 OpenEvolve 的大人格生成器进化框架。该方法将人格生成中的目标绑定、轴向表达、行为锚点和一致性规则组织为可进化蓝图，在固定评估流程下搜索更优生成策略。当前探索实验显示，v3 算子池在 `n=8`、10 代进化设置中使最佳候选相对 baseline 的综合 fitness 提升 26.85%，行为覆盖度提升 25.52%，schema fitness 提升 17.88%，内部一致性提升 3.30%。续跑结果进一步显示 coverage 和 diversity 仍有提升空间。

总体而言，Genome v3 是一个可行且值得继续验证的方向。它的价值不在于一次实验已经证明最终有效，而在于它把“大人格生成器应该如何被优化”从不可控的 prompt 调参转化为可记录、可变异、可消融、可复现的结构化搜索问题。后续工作的重点应是完成 Genome v3 使用有效性验证、多 seed 复现、模块消融和完整 sealed test，从而判断该框架是否能稳定扩展到更大规模的人格生成实验。

## 9. 待补实验

### 9.1 Genome v3 使用有效性验证

目标：回答“蓝图是否真的被人格生成器使用”。

实验设计：

1. 固定 target slot，替换不同 Genome v3，比较输出人格和指标变化。
2. 固定 Genome v3，改变目标坐标，检查 axis alignment 是否随目标变化。
3. 移除 `behavior_anchors`、`consistency_rules`、`axis_expression_policy`，分别观察对应指标是否下降。

通过标准：

- 输出人格的行为证据随 Genome 模块变化发生可解释变化。
- 移除某模块后，对应指标出现方向一致下降。
- 多 seed 下结果趋势稳定。

### 9.2 多 seed 复现

当前主实验只有 seed 17。后续至少使用 3 个 seed，例如 17、23、31，比较 baseline 与 best 的均值和方差。

### 9.3 完整 sealed test

补全以下 test 指标：

- test schema fitness
- test validity rate
- test internal consistency
- test internal consistency min
- test axis alignment
- test shadow alignment
- test behavior coverage
- test behavior balanced diversity

### 9.4 人工评估

抽样比较 baseline 人格与 evolved 人格，由人工评估：

- 是否像同一个真实学生。
- 是否字段之间互相支撑。
- 是否具有非模板化差异。
- 是否存在明显不现实或自相矛盾描述。

## 10. 证据-主张对应表

| 主张 | 当前证据 | 状态 |
|---|---|---|
| v3 算子池能提升最佳 fitness | gen10 best 相对 baseline +26.85% | 初步支持 |
| v3 能提升行为覆盖度 | validation behavior coverage +25.52% | 初步支持 |
| v3 能保持 schema 合法性 | validity 维持 1.0，schema fitness +17.88% | 初步支持 |
| v3 能提升内部一致性 | internal consistency +3.30%，axis alignment +5.08% | 初步支持 |
| v3 能显著提升人格-行为对齐 | shadow alignment +0.42% | 证据较弱 |
| v3 是提升的因果来源 | 尚未做模块消融 | 待验证 |
| 提升能泛化到 sealed test | 当前 sealed test 指标不完整 | 待验证 |
| 方法适用于大规模实验 | 当前 n=8 小规模探索 | 待验证 |

## 11. 引用待补清单

以下引用不得在未检索核对前写成正式参考文献：

1. 大语言模型人格生成与模拟智能体。
2. 结构化 persona / 大人格生成。
3. HACHIMI 或相关多 agent 人格生成研究。
4. DeepMind / OpenEvolve / AlphaEvolve 类进化式自动优化研究。
5. 进化计算中的 genotype-phenotype mapping 与 indirect encoding。
6. 行为问卷模拟、shadow survey 或 psychometric validation 相关研究。

