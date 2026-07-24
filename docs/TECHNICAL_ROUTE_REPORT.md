# MegaPersona-Evolve 技术路线汇报

## 1. 项目定位

本项目的目标不是完整复刻某一篇论文，而是在两个已有方向之间做融合创新：

- **HACHIMI** 解决的是“大人格如何结构化、如何不崩、如何用 schema 和 validator 保证一致性”。
- **DeepMind Persona Generator** 解决的是“如何用覆盖度和进化优化，让生成结果覆盖目标空间”。

本项目提出的路线是：

> 用 HACHIMI-style 的结构化大人格生成保证单个人格质量，用 DeepMind-style 的空间覆盖和 OpenEvolve 保证群体多样性，再用 CEPS/PISA-style shadow survey 做行为层验证。

核心研究问题：

1. 能否生成结构合法、语义丰富、非模板化的大人格？
2. 这些人格是否能覆盖预设的心理行为空间？
3. 人格文本/schema 中声明的特质，是否能在 shadow survey 行为中稳定体现？
4. 进化优化是否能在不破坏 schema 合法性的前提下提高覆盖度和行为一致性？

---

## 2. 总体技术路线

```text
目标人口结构 / quota buckets
  ↓
Primary Axes 撒点
  ↓
MegaPersona Schema
  ↓
多 Agent 生成流水线
  ↓
Symbolic Validator
  ↓
Scientific Shadow Survey
  ↓
LLM Shadow Simulator
  ↓
Schema Metrics + Behavior Metrics
  ↓
Durable Open-Evolve
  ↓
Validation Selection
  ↓
Sealed Test Report
```

系统分成三层：

| 层级 | 作用 | 当前实现 |
|---|---|---|
| 生成层 | 生成结构化 MegaPersona | `src/mega_persona/schema.py`, `generator.py`, `template_generator.py` |
| 评估层 | 计算合法率、内部一致性、覆盖度、行为对齐度 | `evaluation.py`, `consistency.py`, `shadow_survey.py`, `shadow_simulator.py` |
| 进化层 | 优化 schema-aware genome、operator、采样策略、prompt profile | `src.open_evolve.engine.OpenEvolve`, `openevolve_adapter.py`, `scripts/run_mega_persona_evolution.py` |

---

## 3. 大人格 Schema 设计

原 HACHIMI 中的“学业画像”不再作为主组件。本项目将其替换为更泛化的人格心理结构：

| 模块 | 内容 |
|---|---|
| Demographics | 年龄、阶段、地区、家庭背景 |
| Cognitive Motivation Profile | 思维方式、动机系统、学习取向、自我调节、压力反应、决策模式 |
| Values & Identity | 核心价值、自我定位、价值冲突、长期愿望 |
| Social & Creative Profile | 社交能量、协作方式、表达方式、创造模式、同伴影响 |
| Mental Health Context | 压力负荷、韧性、应对方式、保护因素、风险因素 |
| Derived Academic Tendency | 仅作为派生倾向，不作为主优化目标 |

设计原则：

- 主 schema 保持丰富，能承载长文本人格。
- 优化空间保持低维，避免“几十个字段直接做 coverage”导致维度灾难。
- 所有数值字段必须能被 validator 检查，防止全高、全低、字段冲突和不现实组合。

---

## 4. Primary Axes 与 Schema Binding 设计

当前使用 3 个 Primary Axes：

| Axis | 含义 | Schema 来源 |
|---|---|---|
| `cognitive_abstraction` | 具体到抽象的思维倾向 | thinking style |
| `motivation_autonomy` | 自主驱动 vs 外部压力驱动 | motivation system |
| `self_regulation_resilience` | 计划、坚持、情绪调节、恢复力 | self-regulation + mental health |

选择这 3 个轴的原因：

- 跨组件：不是单一字段，而能影响动机、行为、压力反应和社交表现。
- 可行为化：能通过问卷题项投影成 shadow behavior axes。
- 可解释：适合在汇报和实验分析中解释生成差异。

但是当前实现已经不再把这 3 个轴“写死”成唯一合法命名。项目内部新增了 `schema_binding`，显式保存：

- `axis_names`
- `axis_roles`
- `quota_buckets`

其中 `axis_roles` 负责把当前 schema 中的轴映射到三个核心角色：

- `cognitive_core`
- `motivation_core`
- `regulation_core`

这意味着：

1. 研究者可以重命名 primary axes，而不必手改整套 evaluator。
2. slot sampler、adaptive constraints、validator、shadow survey、simulator、visualization 会沿用同一套 axis binding。
3. 进化搜索的对象不再只是“当前三轴上的数值偏移”，而是“带有 schema 绑定信息的生成器控制空间”。

---

## 5. 生成机制

当前支持两种生成模式：

| 模式 | 作用 | 说明 |
|---|---|---|
| `mock` | 离线 baseline | 使用规则模板生成合法人格，不消耗 LLM |
| `llm` | 多 Agent 生成 | 使用 LLM 逐模块生成并合并白板 |

多 Agent 生成流程：

```text
Agent 1: demographics
Agent 2: cognition_motivation_profile
Agent 3: values_identity
Agent 4: social_creative_profile
Agent 5: mental_health_context
  ↓
merge whiteboard
  ↓
symbolic validation
```

生成结果不是自由文本，而是结构化 JSON + 长文本 narrative。这样既能计算指标，又能保留“大人格”的语义厚度。

---

## 6. Scientific Shadow Survey 设计

为了让行为评估更科学，当前 shadow survey 已经从“手写心理题”升级为 **HACHIMI-style CEPS/PISA 构念元数据**。

当前使用的非学业构念包括：

| 来源 | Scale IDs | 对应构念 |
|---|---|---|
| PISA 2022 | `CURIOAGR`, `GROSAGR` | 好奇心、成长型思维 |
| PISA 2022 | `CREATEFF`, `CREATOP` | 创造自我效能、开放性 |
| PISA 2022 | `RELATST`, `BELONG`, `BULLIED` | 关系、归属感、社会威胁 |
| PISA 2022 | `PSYCHSYM`, `LIFESAT`, `WORKHOME` | 心理压力、生活满意度、工作生活平衡 |
| CEPS | `CESD`, `TEACHREL`, `PEERREL`, `MISBEHAVIOR` | 抑郁症状、师生关系、同伴关系、行为自我调节 |

重要说明：

- 当前代码中的题项是 **construct proxy**，不是官方 CEPS/PISA 原题复刻。
- 每道题都带有 `instrument`, `scale_id`, `scale_name`, `axis_weights`。
- 后续如果接入官方公开题项和 coding，可以直接替换 item bank。

---

## 7. 行为计算逻辑

每个 persona 会通过 LLM Shadow Simulator 回答 shadow survey。

行为投影流程：

```text
Likert response: 1-5
  ↓
normalize to 0-1
  ↓
reverse item correction
  ↓
axis_weights 加权投影
  ↓
behavior_axis = [cognitive, motivation, regulation]
```

行为指标包括：

| 指标 | 公式/含义 |
|---|---|
| `behavior_coverage` | shadow behavior points 在 `[0,1]^3` 空间中的随机球覆盖率 |
| `shadow_alignment` | `1 - mean(abs(persona_axis - behavior_axis))` |
| `persona_behavior_mae` | 每个 axis 的平均绝对误差 |

为了避免循环验证，Shadow Simulator 不再看到 hidden numeric axis，例如 `abstraction_level=0.72`。它只看到叙事、类别、价值冲突、压力反应和行为习惯。

---

## 8. 严格验证与测试切分

当前实验已经采用冻结的三分法：

| Split | 用途 | 是否参与进化选优 |
|---|---|---|
| `train` | 运行内分析、后续 ablation | 否 |
| `validation` | candidate fitness 和 best selection | 是 |
| `test` | sealed final report | 否 |

严格性设计：

1. Run 开始时生成固定 survey splits。
2. 写入 `shadow_surveys/train.json`, `validation.json`, `test.json`。
3. 对每个 split 计算 SHA-256 hash。
4. Hash 写入 `manifest.json`, `checkpoint.json`, `shadow_surveys/hashes.json`。
5. Candidate evaluation 只跑 train/validation。
6. Best candidate 选出后才单独跑一次 sealed test。
7. Test 结果只写入 `final_test_report.json`，不进入任何 candidate fitness。

这样可以避免两个问题：

- 候选不能通过改变问卷 seed 来优化验证集。
- 实验过程中不会反复“偷看 test”。

---

## 9. 进化优化设计

当前进化主流程已经真正接入 OpenEvolve 引擎，但它优化的不是任意 Python 源码，而是一个 **schema-aware genome**。这样做的目的不是弱化进化，而是让实验保持可控、可复现、可持久化。

### 9.1 为什么不是直接改整份代码

如果让 LLM 每一代直接改整份生成器源码，会带来几个问题：

- 很容易改崩主流程
- 很难比较代际差异到底来自哪里
- 很难稳定做 checkpoint / resume
- 很容易把“代码质量变化”和“人格质量变化”混在一起

因此当前策略是：

```text
固定主流程
  + 进化可控 genome
  + LLM mutator 产生 child genome
```

### 9.2 当前 genome 结构

| Genome 字段 | 作用 |
|---|---|
| `schema_binding` | 定义当前 schema 下的 axis names / roles / quota buckets |
| `quota_weights` | 调整不同 quota bucket 的采样权重 |
| `axis_bias` | 对 primary axes 做整体偏移 |
| `axis_stretch` | 对 primary axes 做拉伸/压缩 |
| `prompt_profile` | LLM mode 下控制生成风格和机制表达 |
| `last_evolution_operator` | 记录当前子代由哪个 operator 驱动 |

### 9.3 当前 OpenEvolve 真实做了什么

当前真实链路是：

```text
parent genome
  -> OpenEvolve 选择 parent / island / elites
  -> LLM mutator 读取 parent genome + operator + generation context
  -> 输出 child genome JSON
  -> 结构修复 / 归一化 / fallback
  -> MegaPersona generator 执行 child genome
  -> simulator 评估 child persona population
  -> scientific fitness
  -> OpenEvolve 更新 elites
```

也就是说，当前系统里有三个功能明确分开的模型阶段：

- `mutator_model`：负责进化变异
- `persona_model`：负责人格生成
- `simulator_model`：负责行为模拟

这三者可以使用不同模型，也可以切换到 DeepSeek 这类 OpenAI-compatible provider。

### 9.4 为什么要锁住评分规则与问卷拆分

不让 genome 改评分公式、不让 genome 改 validation/test survey，这是为了防止“改尺子”或“改考卷”带来的伪提升。

Fitness 使用固定门控乘积：

```text
fitness =
  schema_fitness
  × (0.5 + 0.5 × validation_behavior_coverage)
  × (0.5 + 0.5 × validation_shadow_alignment)
  × generation_rate
```

其中：

- `schema_fitness` 防止非法人格获得高分。
- `behavior_coverage` 防止行为塌缩。
- `shadow_alignment` 防止人格文本和行为表现脱节。
- `generation_rate` 防止生成失败或 validator 大量拦截。

在当前代码里，还会同步记录：

- `internal_consistency.mean`
- `internal_consistency_min.mean`
- `axis_alignment.mean`

它们用于监控“覆盖度提升”是否靠不现实的人格冲突换来的。

---

## 10. 持久化与可恢复实验

每次 evolution run 会完整保存实验状态：

```text
data/results/mega_persona_evolution_run/
  manifest.json
  checkpoint.json
  final_summary.json
  final_summary.md
  final_test_report.json
  shadow_surveys/
    train.json
    validation.json
    test.json
    hashes.json
  candidates/
    candidate_*.json
  generations/
    generation_0000.json
    generation_0001.json
  evaluations/
    eval_000001_candidate_x/
      result.json
```

持久化策略：

- 每个 candidate 评估完成后立即写 `result.json`。
- 每次评估后更新 `checkpoint.json`。
- Resume 时校验 config 和 frozen survey hashes。
- 允许 `--generations` 在 resume 时增加；OpenEvolve 的 island 状态从 checkpoint 恢复。
- 断线、机器休眠或 API 中断后可以继续跑。

---

## 11. 可视化与报告

当前支持两类可视化：

| 类型 | 命令 | 输出 |
|---|---|---|
| PNG 静态图 | `--format png` | fitness 曲线、axis scatter、best genome、metrics |
| HTML 交互报告 | `--format html` | 可旋转 3D 图、指标面板、候选摘要 |

示例：

```bash
python scripts/visualize_mega_persona_results.py \
  --input data/results/mega_persona_evolution_run \
  --format html
```

---

## 12. 当前已完成

| 模块 | 状态 |
|---|---|
| MegaPersona Schema | 已完成 |
| Symbolic Validator | 已完成 |
| Slot Sampler | 已完成 |
| Rule-based baseline | 已完成 |
| LLM multi-agent generator | 已完成初版 |
| Scientific shadow survey metadata | 已完成初版 |
| LLM Shadow Simulator | 已接入 |
| Schema-aware genome v2 | 已完成 |
| LLM mutator for OpenEvolve | 已接入 |
| Train/validation/test frozen split | 已完成 |
| Sealed final test | 已完成 |
| Durable Open-Evolve | 已完成 |
| Resume/checkpoint | 已完成 |
| Artifact manifest/hash | 已完成 |
| PNG/HTML visualization | 已完成初版 |

---

## 13. 后续技术计划

| 优先级 | 工作 | 目的 |
|---|---|---|
| P0 | 接入官方 CEPS/PISA item wording/coding 或公开可用替代表 | 提高量表有效性 |
| P0 | 多模型 Shadow Simulator 对照 | 检查行为指标是否依赖单一 simulator |
| P1 | Test-retest 稳定性 | 同一 persona 多次答题，评估行为噪声 |
| P1 | Baseline vs Evolution 统计报告 | 多 seed 均值/方差、置信区间、显著性 |
| P1 | Ablation | 去掉 validation alignment、去掉 coverage、去掉 prompt_profile 等 |
| P1 | 新 schema 版本化实验 | 验证 schema_binding 重构后进化信号是否保持 |
| P2 | Agent-specific prompt genome | 从 coarse prompt profile 进化到模块级 prompt fragment |
| P2 | mutator prompt / repair 策略消融 | 分析 LLM mutation 质量瓶颈 |
| P2 | 更细的 construct-level analysis | 按量表族分析哪些构念更稳定 |
| P3 | 人工标注小样本校验 | 检查高分人格是否真的心理一致、非模板化 |

---

## 14. 汇报用一句话总结

MegaPersona-Evolve 的技术路线是：**用结构化 schema 和 validator 保证大人格质量，用低维 primary axes 和 coverage metrics 保证群体空间覆盖，用 CEPS/PISA-style shadow survey 验证行为可测性，再用可恢复、可审计的 Open-Evolve 在冻结 validation 上优化，并最终只对 best candidate 运行 sealed test。**
