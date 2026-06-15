# MegaPersona-Evolve 阶段性实验汇报

- 汇报日期：2026-06-14
- 覆盖批次：2026-06-13 至 2026-06-14
- 项目目标：构建一个能够生成高质量、多样化、可行为验证的大人格数据工厂

## 1. 当前研究问题

本项目不是简单复刻 HACHIMI 或 DeepMind/AlphaEvolve，而是在两者之间做融合：

- HACHIMI 方向解决的是“大人格如何写得结构完整、不崩坏”
- DeepMind/Open-Evolve 方向解决的是“如何覆盖行为/人格空间，并用进化优化生成策略”
- 本项目尝试把两者结合为：结构化大人格生成 + 行为空间覆盖 + 可持久化进化搜索

当前实验围绕三个核心问题展开：

1. 生成器能否稳定产生 schema 合法的大人格？
2. 生成的人格是否能覆盖足够多的行为空间？
3. 进化提示词和 mutation operator 是否能稳定提升生成质量？

## 2. 实验批次总览

| 日期 | 批次 | 规模 | 状态 | 关键结论 |
|---|---|---:|---|---|
| 2026-06-13 | `mega_persona_formal_run` | `n=25`, seeds `17,23,31`, 5 generations | 失败 | 暴露 JSON、schema、timeout 和 checkpoint 问题，不能作为科学结果 |
| 2026-06-13 | `pilot_batch02` | `n=8`, seed `17`, 1 generation | 通过 | 生成链路修复成功，所有候选生成有效人格 |
| 2026-06-13 | `medium_batch03` | `n=16`, seeds `17,23`, 3 generations | 通过 | 行为覆盖度提升，pipeline 达到中等规模稳定性 |
| 2026-06-14 | `operator_smoke2` | `n=6`, seed `17`, 1 generation | 通过 | 20 个 operator 可以运行，但未证明进化稳定增益 |
| 2026-06-14 | `mutation_diagnostic` | `n=6`, seed `17`, 1 generation | 通过 | 首次看到 generation-1 child 小幅超过 generation-0 |
| 2026-06-14 | `mutation_confirm` | `n=6`, seeds `17,23`, 1 generation | 通过 | 单 seed 正向信号未复现，最优仍来自 generation-0 |
| 2026-06-14 | `operator_ablation` | fixed parent, 14 candidates | 通过 | 单个 child 极小幅超过 parent，但接近噪声 |
| 2026-06-14 | `operator_ablation_confirm` | fixed parent, 40 candidates | 通过 | child 偶尔高分，但均值不稳定，旧 operator 不够科学 |

## 3. 关键实验结果

### 3.1 失败批次：`mega_persona_formal_run`

第一轮正式配置较大：

- `n=25`
- seeds: `17,23,31`
- generations: `5`
- population size: `8`
- shadow surveys: train/validation/test = `12/4/4`

结果显示该批次不能作为科学结果：

- best fitness = `0.0000`
- 主要错误包括 `JSONDecodeError`、`AgentOutputError`、`APITimeoutError`
- LLM 输出的 JSON 和 schema 约束不稳定
- 单个 seed 或 slot 的失败容易拖垮整个候选评估

该批次的价值主要在工程诊断。之后已经修复：

- 增加 JSON repair
- 增加 per-slot/per-seed 异常隔离
- 增加 schema prompt contract
- 增加 checkpoint/resume
- 增加日志和 generation diagnostics
- 增加 LLM retry/backoff

### 3.2 小规模通过批次：`pilot_batch02`

修复后，`pilot_batch02` 成功跑通。

| 指标 | 数值 |
|---|---:|
| best validation fitness | `0.2119` |
| validity rate | `1.0000` |
| validation behavior coverage | `0.2380` |
| validation alignment | `0.7843` |
| sealed test behavior coverage | `0.1710` |
| sealed test alignment | `0.7336` |

结论：

- schema 合法性问题基本解决
- sealed test 路径可用
- 但行为覆盖度仍偏低
- best 与 baseline 差距极小，不足以证明进化有效

### 3.3 中等规模批次：`medium_batch03`

这是目前最强的完整 pipeline 证据。

| 指标 | 数值 |
|---|---:|
| best candidate | `candidate_0000_8a2d6494` |
| generation | `0` |
| validation fitness | `0.2985` |
| schema fitness | `0.5011` |
| validity rate | `1.0000` |
| validation behavior coverage | `0.3320` |
| validation alignment | `0.7880` |
| sealed test behavior coverage | `0.3060` |
| sealed test alignment | `0.7717` |

重要结论：

- 行为覆盖度从 pilot 的 `0.2380` 提升到 `0.3320`
- sealed test coverage 达到 `0.3060`
- validation/test gap 较小，说明评估拆分机制有效
- 但最优候选仍然是 generation-0 初始突变，不是后续进化 child

这说明：**生成和评估 pipeline 已经基本可用，但 Open-Evolve 增益尚未被证明。**

## 4. Operator 与 Mutation 诊断

### 4.1 20 operator smoke test

`operator_smoke2` 测试了 20 个初始 evolution operator。

| 候选 | fitness |
|---|---:|
| baseline | `0.1867` |
| best child: `op16_failure_modes` | `0.1849` |

结论：

- operator bank 可以被注入和记录
- child 没有明显崩坏
- 但没有超过 baseline

### 4.2 mutation diagnostic

单 seed 诊断中首次看到 generation-1 child 小幅超过 generation-0：

| 候选 | generation | mutation | operator | fitness |
|---|---:|---|---|---:|
| `candidate_0001_d61f7cb3` | 1 | `mixed` | `op03_behavioral_prediction` | `0.2091` |
| `candidate_0000_ffe3d57c` | 0 | `mixed` | `op16_failure_modes` | `0.2061` |

提升为 `+0.0030`，但只有一个 seed，因此只能视为 pilot signal。

### 4.3 mutation confirmation

两 seed 复现实验没有确认上述正向信号：

| 候选 | generation | mutation | operator | fitness |
|---|---:|---|---|---:|
| `candidate_0000_5f68df59` | 0 | `mixed` | `op16_failure_modes` | `0.2086` |
| `candidate_0001_2e7bf7fe` | 1 | `prompt_only` | `op20_validation_guardrail` | `0.2023` |
| `candidate_0001_c2ab8c7b` | 1 | `mixed` | `op03_behavioral_prediction` | `0.2021` |
| baseline | 0 | - | - | `0.1971` |

结论：

- generation-1 child 接近但没有超过 strong generation-0 parent
- operator/mutation 不是完全无效，但还不稳定
- 强初始化样本对最终结果影响很大

### 4.4 fixed-parent operator ablation

为了排除 parent quality 混淆，后续加入 fixed-parent 消融脚本。

第一次消融：

- parent replay fitness: `0.212641`
- best child: `op01_axis_orthogonality + mixed`
- best child fitness: `0.213068`
- 提升：`+0.000427`

该提升低于重跑噪声，不能视为可靠增益。

确认消融：

- parent replay fitness: `0.216988`
- best child fitness: `0.222005`
- best child: `op01_axis_orthogonality + mixed`

但按 group mean 看，所有 operator 组均低于 parent：

| mutation/operator | n | mean fitness | 是否稳定超过 parent |
|---|---:|---:|---|
| parent replay | 1 | `0.216988` | baseline |
| `prompt_only + op16` | 3 | `0.209572` | 否 |
| `prompt_only + op03` | 3 | `0.209208` | 否 |
| `mixed + op20` | 3 | `0.208170` | 仅 1/3 单点超过 |
| `mixed + op01` | 3 | `0.205497` | 仅 1/3 单点超过 |
| `numeric_only` | 3 | `0.201444` | 否 |

结论：

- 个别 child 可以高分，但不是稳定 operator 效应
- 旧 20 operator 更像“写作建议”，不是足够可测量的实验干预
- parent replay 本身存在明显 LLM 重跑波动，后续必须重复 parent baseline

## 5. 目前已经完成的代码能力

当前系统已经具备以下实验能力：

1. 可持久化进化流程
   - `checkpoint.json`
   - `manifest.json`
   - per-candidate evaluation result
   - generation summary
   - final summary

2. resume 能力
   - 断线或中途失败后可以继续跑
   - resume 时会检查关键 config 是否匹配

3. train/validation/test 拆分
   - shadow surveys 被 frozen 并保存 hash
   - validation 用于 selection
   - sealed test 只用于最终报告

4. LLM 稳定性增强
   - persona generation retry
   - shadow simulation retry
   - JSON repair
   - per-slot/per-seed failure isolation

5. mutation diagnostics
   - `prompt_only`
   - `operator_only`
   - `mixed`
   - `numeric_only`

6. fixed-parent ablation
   - 同一个 parent genome 下比较不同 operator
   - parent replay 现在也支持按 replicate 重复

7. operator bank 重构
   - 旧 20 operator 已压缩为 8 个更可测量的 operator
   - 不改变主 selection/fitness 逻辑

## 6. 最新代码调整：8 个新 operator

根据旧 operator 的实验结果，已经将 20 个泛化 operator 改为 8 个更接近实验处理条件的 operator：

| Operator | 设计目的 |
|---|---|
| `op01_axis_decoupling` | 强制生成高/低轴反差，避免三轴一起高 |
| `op02_behavioral_evidence` | 要求 deadline、peer pressure、failure feedback、ambiguous task 四类行为证据 |
| `op03_shadow_survey_alignment` | 强化可推断问卷回答的行为线索 |
| `op04_within_bucket_contrast` | 在同一 quota bucket 内增加资源、日程、支持网络、风险差异 |
| `op05_failure_recovery_cycle` | 显式写出失败后的触发、解释、应对、结果、调整 |
| `op06_low_axis_fidelity` | 低轴必须体现行为代价，避免被写成泛化优秀 |
| `op07_high_axis_cost` | 高轴必须体现代价，避免人格理想化 |
| `op08_validation_conservatism` | 保证字段长度、时间线、人口统计和跨 agent 一致性 |

这次调整的原则是：operator 不再只是“让人格更丰富”，而是必须改变可观察行为证据。

## 7. 当前科学判断

已经可以支持的结论：

1. MegaPersona 多 agent 生成 pipeline 已经可以稳定生成 schema 合法人格。
2. 行为覆盖度可以通过更大样本和更稳定生成策略提升到 `0.30+`。
3. train/validation/test 拆分和 sealed test 路径已经具备基本科学实验形态。
4. evolution operator 的影响可以被记录、消融和复现实验检验。

还不能支持的结论：

1. 还不能声称 Open-Evolve 已经稳定提升生成器。
2. 还不能声称某个 operator 是稳定最优。
3. 还不能把单次 best child 超过 parent 当作可靠进化效果。
4. 当前 shadow survey 仍然是代理测量，还需要进一步对齐科学量表。

## 8. 主要问题

### 问题 1：进化增益不稳定

最优经常来自 generation-0 初始突变，而不是后代。这说明当前进化过程还更像“随机搜索 + 局部扰动”，还没有形成稳定爬坡。

### 问题 2：旧 operator 过于抽象

旧 operator 使用“提高行为可预测性”“增强轴独立性”等泛化语言，对 LLM 来说太像写作风格提示，不一定转化成可测量行为。

### 问题 3：LLM 评估噪声明显

同一 parent replay 在不同重跑中会有明显 fitness 变化。因此后续实验必须重复 parent baseline，否则容易误判。

### 问题 4：评估成本高

每个 candidate 需要多 agent persona generation 和 shadow survey simulation。即使 `n=6`、两个 seed，也需要较长运行时间。

## 9. 下一步实验计划

### Step 1：新版 8 operator 消融

使用固定 parent，测试 8 个新 operator。

建议配置：

- `n=6`
- seeds: `17,23`
- replicates: `2`
- mutation modes: `parent_replay,prompt_only,operator_only,mixed,numeric_only`

成功标准：

- 至少一个 operator group 的 mean fitness 超过 repeated parent replay mean
- 该 operator 至少 `2/2` 或 `3/4` 次超过 parent
- 不出现 schema validity 下降

### Step 2：保留有效 operator，删除无效 operator

如果某些 operator 持续低于 parent，应从主进化中移除，避免扩大搜索噪声。

### Step 3：再跑小规模 evolution

只有当新版 operator ablation 有稳定信号后，再跑：

- `n=8` 或 `n=12`
- seeds: `17,23`
- generations: `2`
- population size: `5`
- children per generation: `4`

### Step 4：进入中等规模复现

如果小规模 evolution 出现 generation-1 或 generation-2 稳定超过 generation-0，再回到 `medium_batch03` 规模复现。

## 10. 一句话总结

截至 2026-06-14，本项目已经从“生成链路不稳定”推进到“可稳定生成、可持久化进化、可消融 operator”的阶段。当前最大的科学瓶颈不是 schema 合法性，而是进化 operator 的稳定有效性。昨天和今天的实验说明：旧 operator 偶尔能采到高分 child，但不能稳定超过 parent。因此，已经将 operator 从 20 个抽象写作提示压缩为 8 个行为证据型干预，下一步应先验证新版 operator，而不是直接扩大正式实验规模。
