# MegaPersona-Evolve 技术路线与阶段性实验汇报

- 汇报日期：2026-06-15
- 项目阶段：方案验证与进化提示词消融阶段
- 当前核心判断：生成与评估主流程已基本稳定，新版进化提示词出现更强早期信号，但“稳定进化增益”仍需完整复现实验确认

## 一、研究目标

本项目希望构建一个面向心理与教育行为模拟的“大人格生成器”。它不是单纯生成一段 persona 文本，而是希望生成：

1. 结构完整的人格画像
2. 可通过 schema 校验的人格 JSON
3. 在认知、动机、自我调节、社会行为、心理健康等维度上具有内部一致性
4. 能够在 shadow survey 中表现出可预测的行为差异
5. 能够通过进化机制不断提高覆盖度、多样性和行为对齐度

因此，本项目的核心问题是：

> 如何把“大人格的深度”和“行为/人格空间的广度”结合起来，并用可持久化的进化实验逐步优化生成策略？

## 二、整体技术路线

当前技术路线可以概括为三层：

```text
进化层
  优化对象：prompt profile、operator、采样策略、轴变换
  方法：接入 `src.open_evolve.engine.OpenEvolve` 的岛屿式进化优化

生成层
  方法：HACHIMI-style 多 Agent 流水线
  输出：结构化 MegaPersona JSON + 叙事性字段

评估层
  方法：schema validity + 行为空间覆盖 + shadow survey 行为对齐
  拆分：train / validation / sealed test
```

### 2.1 生成层

生成层采用多 Agent pipeline：

1. Demographics Agent：生成基本背景
2. Cognition & Motivation Agent：生成思维方式、动机系统、自我调节
3. Values & Identity Agent：生成价值观与身份叙事
4. Social & Creative Agent：生成社会互动和创造性表达
5. Mental Health Agent：生成压力、恢复、支持系统等心理健康背景

其中后 3 个 agent 已经并行执行，以减少单个人格的生成时间。

### 2.2 评估层

评估层主要包含三类指标：

1. **Schema 合法性**
   - 是否符合 MegaPersona schema
   - 字段是否完整
   - 人口统计、年龄、阶段、叙事字段是否一致
   - 是否存在近重复人格

2. **行为覆盖度**
   - 通过 shadow survey 模拟人格对问卷项目的回答
   - 将回答映射到行为轴空间
   - 计算 coverage、average distance 等多样性指标

3. **人格-行为对齐度**
   - 比较 persona 声明的 primary axes 与 shadow survey 中模拟出的行为轴
   - 用 overall alignment 衡量人格描述是否能预测行为表现

当前 fitness 的基本思想是乘法门控：

```text
fitness = schema_fitness
        × behavior_coverage_gate
        × alignment_gate
        × generation_rate
```

这样可以避免：某个候选只在单一维度上极端高分，但 schema 崩坏或生成率很低。

### 2.3 进化层

进化层当前优化的是“生成策略”，不是直接优化完整代码。主要包括：

- quota weights
- primary axis bias/stretch
- prompt profile
- evolution operator
- mutation mode

当前支持的 mutation mode：

| Mode | 含义 |
|---|---|
| `prompt_only` | 只改变 prompt/operator，不改采样参数 |
| `operator_only` | 只应用一个 operator，不额外随机扰动 prompt profile |
| `mixed` | 同时改变 operator、prompt profile 和采样参数 |
| `numeric_only` | 只改变 quota/axis sampling，不注入 operator |

这让我们可以区分：提升到底来自提示词、采样数值，还是两者组合。

### 2.4 OpenEvolve 接入方式

最新代码已经将 MegaPersona 主进化入口替换为仓库内 `src.open_evolve.engine.OpenEvolve` 引擎调度。具体做法是：

- MegaPersona 仍固定多 Agent schema 与评估逻辑，避免 LLM 直接改坏主流程。
- 将 MegaPersona genome 序列化为 OpenEvolve 的候选 `code` 字段。
- 使用 OpenEvolve 负责 island、metric elites、mutation、checkpoint 与 resume。
- 使用 MegaPersona evaluator 负责生成、schema 校验、shadow survey validation fitness 和 sealed test。
- 主入口为 `scripts/run_mega_persona_evolution.py`。

因此当前项目不是只“仿照 Open-Evolve”，而是由 OpenEvolve 引擎直接承担进化机制；MegaPersona 侧只保留 genome、评估和结果持久化。

## 三、实验方案设计

### 3.1 数据拆分

实验使用 shadow surveys，并固定拆分为：

- train shadow surveys
- validation shadow surveys
- sealed test shadow surveys

selection 只使用 validation；sealed test 只用于最终报告。这是为了避免过拟合测试集。

每次运行会保存：

- `manifest.json`
- `checkpoint.json`
- `shadow_surveys/train.json`
- `shadow_surveys/validation.json`
- `shadow_surveys/test.json`
- survey hash
- per-candidate evaluation artifacts

### 3.2 可恢复实验

所有进化实验都支持持久化：

- 每个 candidate 评估完成后立即写入磁盘
- 每次 evaluation 后更新 checkpoint
- 断线或电脑休眠后可以 resume
- resume 时检查关键配置是否一致

最近已新增：

- `run_mega_persona_operator_ablation.py --resume`
- `--shadow-max-workers`，用于 shadow survey 内部并发加速
- resume 时允许调整 shadow survey 内部并发数；主进化并发由 OpenEvolve 的 `children_per_island` 控制

### 3.3 Operator 消融

由于前几轮实验发现 generation-0 强初始化会掩盖 operator 效果，后续加入 fixed-parent ablation：

- 固定一个强 parent genome
- 对同一 parent 应用不同 operator
- 比较 parent replay、prompt_only、operator_only、mixed、numeric_only
- 通过 repeated parent replay 估计 baseline 波动

这是当前判断进化提示词是否有效的主要方法。

## 四、实验批次与结果

### 4.1 Batch 01：`pilot_batch02`

配置：

- `n=8`
- seed: `17`
- generations: `1`
- population size: `3`

关键结果：

| 指标 | 数值 |
|---|---:|
| best validation fitness | `0.2119` |
| validity rate | `1.0000` |
| validation behavior coverage | `0.2380` |
| validation alignment | `0.7843` |
| sealed test behavior coverage | `0.1710` |
| sealed test alignment | `0.7336` |

结论：

- 生成链路已经修复
- schema 合法性达到可用水平
- sealed test 路径可用
- 但行为覆盖度偏低
- best 与 baseline 差距太小，不能证明进化有效

后续优化：

- 修复 DiversityMetrics 中未固定随机参考点的问题
- coverage / dispersion 改为确定性计算
- 提高实验复现性

### 4.2 Batch 02：`medium_batch03`

状态：中等规模通过，是目前最强的 pipeline 证据

配置：

- `n=16`
- seeds: `17,23`
- generations: `3`
- population size: `5`
- shadow surveys: `8/3/3`

最佳候选：

- candidate: `candidate_0000_8a2d6494`
- generation: `0`
- validation fitness: `0.2985`

关键结果：

| 指标 | 数值 |
|---|---:|
| schema fitness | `0.5011` |
| validity rate | `1.0000` |
| validation behavior coverage | `0.3320` |
| validation alignment | `0.7880` |
| sealed test behavior coverage | `0.3060` |
| sealed test alignment | `0.7717` |

结论：

- 行为覆盖度从 pilot 的 `0.2380` 提升到 `0.3320`
- sealed test coverage 达到 `0.3060`
- validation/test gap 较小
- pipeline 在中等规模下可用
- 但最优候选仍来自 generation-0，不是后续 child

这说明当前系统已经能生成高质量人格，但还没有证明多代进化能稳定改进。

后续优化：

- 增加 LLM retry/backoff
- 增加 prompt_profile 的维度
- 增加 operator 记录
- 降低后代 mutation scale，让后续生成更偏局部探索

## 五、旧版 20 Operator 实验

### 5.1 Operator Smoke Test

时间：2026-06-14  
状态：通过，但没有证明 operator 有效

结果：

| 候选 | fitness |
|---|---:|
| baseline | `0.1867` |
| best child: `op16_failure_modes` | `0.1849` |

结论：

- operator bank 可以被注入和持久化
- child 没有明显崩坏
- 但未超过 baseline

### 5.2 Mutation Diagnostic

时间：2026-06-14  
状态：小规模正向信号

结果：

| 候选 | generation | mutation | operator | fitness |
|---|---:|---|---|---:|
| `candidate_0001_d61f7cb3` | 1 | `mixed` | `op03_behavioral_prediction` | `0.2091` |
| `candidate_0000_ffe3d57c` | 0 | `mixed` | `op16_failure_modes` | `0.2061` |

这是首次看到 generation-1 child 超过 generation-0，提升约 `+0.0030`。

但这个实验只有一个 seed，因此只能视为 pilot signal。

### 5.3 Mutation Confirmation

时间：2026-06-14  
状态：两 seed 复现实验没有确认单 seed 正向信号

结果：

| 候选 | generation | mutation | operator | fitness |
|---|---:|---|---|---:|
| `candidate_0000_5f68df59` | 0 | `mixed` | `op16_failure_modes` | `0.2086` |
| `candidate_0001_2e7bf7fe` | 1 | `prompt_only` | `op20_validation_guardrail` | `0.2023` |
| `candidate_0001_c2ab8c7b` | 1 | `mixed` | `op03_behavioral_prediction` | `0.2021` |
| baseline | 0 | - | - | `0.1971` |

结论：

- generation-1 child 接近强 parent，但没有超过
- 单 seed 的正向信号不稳定
- strong initialization 对最终结果影响很大

## 六、旧版 Operator 消融

### 6.1 Fixed-parent ablation

为了排除 parent quality 影响，后续固定 `candidate_0000_5f68df59` 作为 parent。

第一次消融：

- parent replay fitness: `0.212641`
- best child: `op01_axis_orthogonality + mixed`
- best child fitness: `0.213068`
- 提升：`+0.000427`

这个提升非常小，低于 LLM 重跑噪声。

### 6.2 Fixed-parent ablation confirmation

第二次消融扩大到 40 个候选。

结果：

- parent replay fitness: `0.216988`
- best child: `op01_axis_orthogonality + mixed`
- best child fitness: `0.222005`

但按 group mean 看，所有 operator 组均低于 parent：

| group | n | mean fitness | 稳定超过 parent |
|---|---:|---:|---|
| parent replay | 1 | `0.216988` | baseline |
| `prompt_only + op16` | 3 | `0.209572` | 否 |
| `prompt_only + op03` | 3 | `0.209208` | 否 |
| `mixed + op20` | 3 | `0.208170` | 仅单点超过 |
| `mixed + op01` | 3 | `0.205497` | 仅单点超过 |
| `numeric_only` | 3 | `0.201444` | 否 |

结论：

- 旧版 operator 偶尔能产生高分 child
- 但 group mean 不稳定
- 旧 operator 更像“写作建议”，不够像可检验的实验干预

因此决定重构 operator bank。

## 七、新版 8 Operator 设计

旧版 20 operator 被压缩为 8 个更具体、更可测量的 operator。

| Operator | 设计目的 |
|---|---|
| `op01_axis_decoupling` | 强制三轴之间出现高/低反差，避免所有维度一起高 |
| `op02_behavioral_evidence` | 要求 deadline、peer pressure、failure feedback、ambiguous task 四类行为证据 |
| `op03_shadow_survey_alignment` | 增强可推断问卷回答的行为线索 |
| `op04_within_bucket_contrast` | 在同一 quota bucket 内增加资源、日程、支持网络和风险差异 |
| `op05_failure_recovery_cycle` | 明确失败后的触发、解释、应对、结果、调整 |
| `op06_low_axis_fidelity` | 低轴必须体现行为代价，避免被写成泛化优秀 |
| `op07_high_axis_cost` | 高轴必须体现代价，避免人格理想化 |
| `op08_validation_conservatism` | 保证字段长度、时间线、人口统计和跨 agent 一致性 |

核心变化：

- 从“让人格更丰富”改成“必须写出可观察行为证据”
- 从抽象写作指令改成实验处理条件
- 更容易通过 fixed-parent ablation 判断 operator 是否有效

## 八、新版 Operator V2 消融实验

时间：2026-06-14 至 2026-06-15  
状态：仍在运行，当前为 partial analysis

配置：

- fixed parent: `candidate_0000_5f68df59`
- operators: 新版 8 operator
- mutation modes: `parent_replay,prompt_only,operator_only,mixed,numeric_only`
- replicates: `2`
- seeds: `17,23`
- planned candidates: `52`

当前 checkpoint：

- completed: `27/52`
- pending: `25`
- current best: `ablation_0014_op04_within_bucket_contrast_mixed_r01`
- current best fitness: `0.2189`

Parent replay baseline：

| Parent replay | Fitness |
|---|---:|
| `ablation_0000_parent_replay_r01` | `0.2038` |
| `ablation_0001_parent_replay_r02` | `0.2072` |
| Mean | `0.2055` |
| Std | `0.0017` |

当前 Top candidates：

| Rank | Candidate | Mode | Operator | Fitness | Val Coverage | Val Alignment |
|---:|---|---|---|---:|---:|---:|
| 1 | `ablation_0014_op04_within_bucket_contrast_mixed_r01` | `mixed` | `op04_within_bucket_contrast` | `0.2189` | `0.3285` | `0.7546` |
| 2 | `ablation_0008_op02_behavioral_evidence_mixed_r01` | `mixed` | `op02_behavioral_evidence` | `0.2189` | `0.3385` | `0.7293` |
| 3 | `ablation_0013_op04_within_bucket_contrast_operator_only_r01` | `operator_only` | `op04_within_bucket_contrast` | `0.2126` | `0.2960` | `0.7638` |
| 4 | `ablation_0011_op03_shadow_survey_alignment_mixed_r01` | `mixed` | `op03_shadow_survey_alignment` | `0.2120` | `0.2795` | `0.7445` |
| 5 | `ablation_0006_op02_behavioral_evidence_prompt_only_r01` | `prompt_only` | `op02_behavioral_evidence` | `0.2114` | `0.2925` | `0.7647` |

初步结论：

- 新版 operator 的早期信号明显强于旧版 20 operator
- `op04_within_bucket_contrast + mixed` 比 parent mean 高约 `+0.0134`
- `op02_behavioral_evidence + mixed` 也比 parent mean 高约 `+0.0134`
- 当前最值得关注的是 `op04` 和 `op02`
- 但实验还未完成，第二个 replicate 仍需确认稳定性

可靠性说明：

- 已保存的 27 个候选中，seed-level status 均为 `ok`
- 个别候选如 `op08_validation_conservatism_operator_only` 被 slot failure 明显拖低，不应直接解读为 operator 差

## 九、近期代码优化总结

### 9.1 稳定性优化

已完成：

- JSON repair
- schema prompt contract
- per-slot exception handling
- per-seed isolation
- retry/backoff
- detailed logging
- checkpoint/resume

效果：

- 从 batch01 的全局失败，推进到 batch02/batch03 的稳定生成
- 当前多数候选能保存完整 per-seed 结果

### 9.2 科学实验结构优化

已完成：

- train/validation/test 分离
- sealed test 只用于最终评估
- shadow survey hash 持久化
- manifest 记录配置、git 状态和命令
- fixed-parent ablation
- parent replay repetition

效果：

- 可以区分 parent quality 与 operator quality
- 可以判断单个高分 child 是否只是偶然
- 可以避免直接在 test 上调参

### 9.3 进化提示词优化

已完成：

- 从旧版 20 operator 重构为新版 8 operator
- 将抽象写作建议改为行为证据型干预
- 保留 mutation mode 诊断能力

效果：

- 旧版 operator 的提升接近噪声
- 新版 operator 在 partial checkpoint 中出现更大幅度提升

### 9.4 速度优化

已完成：

- persona 内部后 3 个 agent 并发
- OpenEvolve island/child 级进化调度
- 新增 shadow survey 内部并发参数 `--shadow-max-workers`
- 新增 ablation resume 支持

建议使用：

```bash
--children-per-island 1
--shadow-max-workers 4
```

这样每个 island 每代只生成 1 个 child，便于观察进化轨迹；但每个 candidate 内部的 shadow survey 请求可以并行，速度更快。

## 十、当前阶段性判断

### 已经可以支持的结论

1. MegaPersona 多 Agent 生成链路已经基本稳定。
2. Schema 合法性已经不再是主要瓶颈。
3. 行为覆盖度可以提升到 `0.30+`。
4. train/validation/test 拆分和 sealed test 路径已经可用。
5. fixed-parent ablation 可以有效诊断 operator。
6. 新版 8 operator 比旧版 20 operator 更有潜力。


## 十一、下一步实验计划

### Step 1：完成 V2 operator ablation

### Step 2：筛选 operator

完成后重点看：

- group mean 是否超过 parent replay mean
- win-rate 是否稳定
- 是否因为 API failure 导致 generation_rate 降低
- schema fitness 是否下降

暂定保留标准：

- group mean > repeated parent replay mean
- 至少两个 replicate 中有稳定正向结果
- 不牺牲 validity

### Step 3：小规模 evolution 复测

如果 V2 operator 有稳定信号，再跑小规模 evolution：

- `n=8` 或 `n=12`
- seeds: `17,23`
- generations: `2`
- population size: `5`
- children per generation: `4`

目标：

- 看新版 operator 是否能让 generation-1 或 generation-2 稳定超过 generation-0

### Step 4：中等规模复现

如果小规模 evolution 通过，再回到类似 `medium_batch03` 的规模复现。

成功标准：

- best evolved child 超过 best generation-0
- validation behavior coverage 保持 `0.30+`
- sealed test coverage 不明显下降
- validation/test alignment gap 可控

## 十二、总结

本阶段最大的进展是：项目已经从“生成链路不稳定”进入“可运行、可恢复、可消融、可诊断”的实验阶段。早期失败主要是工程稳定性问题；经过 JSON repair、schema contract、retry、checkpoint 和日志增强后，系统已经能在中等规模下稳定生成有效人格。

当前最大科学问题不再是 schema 合法性，而是：进化提示词是否能稳定带来增益。旧版 20 operator 的实验说明，抽象写作型提示词不足以形成稳定进化效果。新版 8 operator 将提示词改造成行为证据型干预，当前 V2 消融实验在 27/52 checkpoint 已经显示出更强早期信号，尤其是 `op04_within_bucket_contrast` 和 `op02_behavioral_evidence`。

因此，下一步不应直接扩大正式实验，而应先完成 V2 operator ablation，确认哪些 operator 真正稳定有效，再进入小规模 evolution 复测。这个节奏更符合严格科学实验：先校准干预变量，再扩大样本和代数。
