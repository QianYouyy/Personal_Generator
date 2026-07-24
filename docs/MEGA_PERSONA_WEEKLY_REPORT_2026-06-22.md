# MegaPersona-Evolve 周报

日期：2026-06-22  
周期：2026-06-16 至 2026-06-22  
项目方向：结构化大人格生成、空间覆盖优化、OpenEvolve 进化搜索、行为一致性评估

## 1. 本周总体进展

本周主要围绕 MegaPersona-Evolve 的“可运行、可续跑、可分析、可扩展”四个目标推进。核心工作包括：完成 OpenEvolve 主流程替换、扩大算子库、优化并发效率、完成中等和较大规模进化实验分析，并初步接入 Concordia-style shadow simulator，为后续更强行为模拟器对比实验做准备。

整体来看，本周项目已经从早期的“概念验证和单次实验”进入到“可复现实验系统”的阶段。当前主流程已经能够完成从人格生成、shadow survey 行为模拟、fitness 计算、OpenEvolve 岛屿进化、checkpoint 持久化、sealed test 到结果报告输出的完整闭环。

## 2. 本周完成的工作

### 2.1 OpenEvolve 主流程落地

本周完成了从自定义 Open-Evolve-style 逻辑到项目内 `src.open_evolve.engine.OpenEvolve` 的主流程切换。当前实验入口为：

```bash
python scripts/run_mega_persona_evolution.py
```

主要能力包括：

- 使用 OpenEvolve island engine 进行多岛进化。
- 支持 `--num-islands` 指定岛屿数量。
- 支持 `--children-per-island` 控制每岛每代生成候选数量。
- 支持 `--resume` 从 `open_evolve/checkpoint.json` 续跑。
- 每个候选 genome、评估结果、checkpoint、final summary 均持久化保存。
- sealed test 只在最终 best candidate 选择后运行，避免测试集泄露到进化选择中。

目前旧的 `MegaPersonaEvolver.run()` 已退役，`MegaPersonaEvolver` 只保留为评估和持久化后端，真正进化由 `MegaPersonaOpenEvolveRunner` 调用 OpenEvolve 引擎完成。

### 2.2 进化算子扩展到 15 个

本周将进化提示词 operator 扩展到 15 个，覆盖更多人格变异机制：

- 轴解耦、行为证据、shadow survey 对齐
- quota bucket 内部差异化
- 失败-恢复机制
- 低轴保真、高轴代价
- validation 保守性
- 高低轴 tradeoff
- 场景化 bucket split
- 决策轨迹证据
- 支持网络不对称
- 自主性压力测试
- 恢复延迟
- survey-discriminating cues

从实验结果看，当前比较有价值的 operator 包括：

| Operator | 当前判断 |
|---|---|
| `op06_low_axis_fidelity` | 本周中等规模实验中信号最强 |
| `op04_within_bucket_contrast` | 多轮实验中较稳定 |
| `op09_low_high_axis_tradeoff` | 较大规模实验中出现 best candidate |
| `op13_autonomy_pressure_test` | 多次出现正向候选 |
| `op14_recovery_latency` | 单点表现较好，仍需更多样本 |

当前不建议完全均匀抽样所有 operator。下一阶段更适合做 focused evolution：提高 `op06`、`op04`、`op09`、`op13` 的采样权重，同时继续保留少量探索。

### 2.3 并发与运行效率优化

本周对实验速度做了几处关键优化：

- `--candidate-max-workers`：控制 OpenEvolve candidate 级别并发。
- `--persona-max-workers`：控制单个 candidate 内部人格生成并发。
- `--shadow-max-workers`：控制 shadow survey 模拟并发。
- OpenEvolve evaluator 增加锁，避免并发写 checkpoint 和 candidate id 时出现竞争。
- shadow simulation 已支持多线程并发。

较大规模实验中，`max_workers=8`、`shadow_max_workers=16`、`persona_max_workers=8` 后，整体运行时间显著下降。说明当前主要瓶颈已经从程序串行逻辑转向 LLM 调用耗时与接口稳定性。

### 2.4 中等规模与较大规模实验结果

本周重点分析了两个 OpenEvolve 实验。

中等规模实验：

```text
data/results/mega_persona_openevolve_medium_20260616_ops15_fast
```

配置摘要：

- n：`24`
- seeds：`17,23`
- generations：`3`
- islands：`6`
- best candidate：`openevolve_000018_2d5f2c46d49e`
- best fitness：`0.370101`
- sealed test alignment：`0.766076`
- sealed test behavior coverage：`0.350000`

较大规模实验：

```text
data/results/mega_persona_openevolve_large_20260616_ops15
```

配置摘要：

- n：`32`
- seeds：`17,23,31`
- generations：`4`
- islands：`8`
- best candidate：`openevolve_000029_78b2a0a1e457`
- best fitness：`0.372785`
- sealed test alignment：`0.778965`
- sealed test behavior coverage：`0.415667`

较大规模实验的主要结论：

- best candidate 来自第 4 代，不是 baseline，说明进化搜索有实际收益。
- sealed test behavior coverage 达到 `0.415667`，高于中等规模实验。
- 进化提升主要体现在行为覆盖和行为差异，而不是只提高 schema 表面合法性。
- `operator_only` 和部分机制型 operator 仍是当前最稳的方向。

### 2.5 文档与项目说明整理

本周同步更新了项目说明文件，使新的对话或新的 agent 可以快速理解项目：

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/MEGA_PERSONA_BATCH_2026-06-16_OPENEVOLVE_MEDIUM.md`

文档中补充了：

- OpenEvolve 正式入口命令。
- `--num-islands` 替代旧的 `--population-size`。
- 结果目录结构说明。
- 当前 operator 证据和后续搜索策略。
- Concordia-style simulator 的运行命令。

### 2.6 Concordia-style 模拟器初步接入

本周初步接入了 Concordia-style shadow survey simulator。代码层面已经支持：

```bash
--simulator-backend llm
--simulator-backend concordia
```

新增能力：

- `ConcordiaShadowSimulator`
- `build_shadow_simulator`
- simulator backend 写入 config/manifest
- 支持更强或第三方 OpenAI-compatible simulator model：

```bash
--simulator-model
--simulator-api-base
--simulator-api-key-env
```

已完成一次 Concordia smoke run：

```text
data/results/mega_persona_concordia_smoke
```

结果判断：

- 流程层面：可以跑完，产物完整写出。
- 运行日志显示：本机检测到了 `concordia` package。
- 实验结果：best fitness 为 `0.0`，sealed test 被跳过。
- 主要原因：本次 smoke 中人格生成阶段出现大量 API timeout，多个 candidate 的有效人格数为 `0/10`，导致后续 simulator 没有人格可评估。

因此，当前只能说明 Concordia backend 已接入并能进入主流程，但还不能说明 Concordia-style simulator 已经优于 baseline LLM simulator。下周需要用更小并发、更低 n 或更稳定模型做一次干净复测。

## 3. 本周关键判断

### 3.1 进化机制是有效的

多轮实验都显示 best candidate 不再停留在 baseline，而是来自后续 generation。这说明 OpenEvolve 的岛屿搜索和 operator mutation 能够找到更优 genome。

### 3.2 当前优化重点应从“广泛探索”转为“有重点探索”

15 个 operator 全部均匀抽样会浪费一部分预算。结合本周结果，下阶段更适合：

- 提高强信号 operator 权重。
- 降低历史表现差或高风险 operator 权重。
- 以 `operator_only` 为主，少量保留 `prompt_only`。
- 暂时减少大幅 `mixed` mutation。

### 3.3 simulator 是后续科学性提升的关键

当前 shadow survey simulator 是评估体系的核心。如果 simulator 太弱，进化会优化到 simulator 偏差；如果 simulator 更强，可能更好地区分“文本人格表面多样”与“真实行为可区分”。

因此，接入 Concordia 或更强 LLM simulator 是合理方向，但需要通过 A/B 实验验证，而不能只凭框架名判断有效。

### 3.4 并发提速有效，但需要配套稳定性策略

提高并发后运行速度明显提升，但也更容易触发 timeout。后续需要在效率和稳定性之间做配置分层：

- smoke run：小 n、高可控、用于功能验证。
- medium run：中等并发，用于 operator 方向判断。
- formal run：较大规模，但需要更稳的 timeout/retry/日志监控配置。

## 4. 当前问题与风险

| 问题 | 影响 | 下周处理方式 |
|---|---|---|
| Concordia smoke best fitness 为 0 | 无法判断 Concordia simulator 效果 | 用低并发重跑干净 smoke |
| operator 样本仍不均衡 | 不能严格比较所有 operator | 做 focused run 或 operator ablation |
| simulator 可能带来评估偏差 | 进化可能过拟合 simulator | 做 baseline LLM vs Concordia vs stronger LLM A/B |
| 高并发下 timeout 增多 | 有效人格数下降，fitness 失真 | 增加重试或降低 persona 并发 |
| 结果目录较多 | 项目可读性下降 | 后续保留关键批次，清理临时 run |

## 5. 下周计划

### 5.1 任务一：Concordia 原生集成设计与最小实现

目标：在当前 Concordia-style prompt adapter 的基础上，进一步推进 google-deepmind/concordia 的原生集成，明确 MegaPersona 如何映射为 Concordia agent、component、memory 和 game-master 场景。

当前版本已经支持 `--simulator-backend concordia`，但它仍然主要是 Concordia-style 的问卷模拟 adapter。下周需要做的是把“Concordia 风格模拟”推进到“Concordia 原生组件可运行”的最小版本。

具体工作：

- 阅读本地 `concordia` package 的 agent、component、memory、language_model、prefab 结构。
- 设计 `MegaPersona -> Concordia Agent` 的字段映射：
  - identity / values
  - motivation / goals
  - thinking style / decision pattern
  - social context
  - mental health context
  - autobiographical memory snippets
- 增加一个最小 adapter，例如 `ConcordiaNativeShadowSimulator` 或同等命名模块。
- 保持现有 shadow survey 接口不变，即输入仍为 `MegaPersona + ShadowSurvey`，输出仍为 `ShadowSurveySimulation`。
- 先支持单 persona、单 survey 的最小闭环，再考虑并发和批量评估。

验收标准：

- 能用一个 MegaPersona 构造 Concordia agent。
- 能通过 Concordia 的 agent/component 机制完成一次 survey 场景模拟。
- 输出仍能被当前 `score_shadow_survey()` 正常评分。
- 不破坏现有 `llm` backend 和 Concordia-style backend。
- 文档中明确区分：
  - `llm` simulator
  - `concordia-style` simulator
  - `concordia-native` simulator

预期产出：

```text
src/mega_persona/concordia_adapter.py
docs/MEGA_PERSONA_CONCORDIA_INTEGRATION_NOTES.md
```

### 5.2 任务二：重跑 Concordia 干净 smoke

目标：用低并发、小规模配置复测 Concordia-style simulator，确认它是否可以稳定完成“人格生成 -> shadow survey 模拟 -> fitness 计算 -> sealed test”的闭环。

下周第一步先不扩大实验规模，而是用更保守的运行配置完成一次干净 smoke，为后续 simulator A/B 对比建立可靠基础。

建议命令：

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-backend concordia \
  --simulator-model-key llm.simulator_model \
  --n 6 \
  --seeds 17 \
  --generations 1 \
  --num-islands 2 \
  --children-per-island 1 \
  --shadow-surveys 2 \
  --validation-shadow-surveys 1 \
  --test-shadow-surveys 1 \
  --items-per-shadow-survey 6 \
  --candidate-max-workers 1 \
  --persona-max-workers 1 \
  --shadow-max-workers 2 \
  --output-dir data/results/mega_persona_concordia_clean_smoke_20260623
```

判断标准：

- 每个 candidate 至少生成 `4/6` 个有效 persona。
- best fitness 必须大于 `0`。
- `final_test_report.json` 不能是 skipped。
- `run.log` 中没有影响主流程完成的连续失败。
- manifest 中需要记录 `shadow_simulator_backend = concordia`。

预期产出：

- `data/results/mega_persona_concordia_clean_smoke_20260623/final_summary.json`
- `data/results/mega_persona_concordia_clean_smoke_20260623/final_test_report.json`
- 一段简短结论：Concordia backend 是“可进入后续 A/B”，还是“需要继续修复后再比较”。

### 5.3 任务三：同规模 LLM baseline 对照

目标：给 Concordia clean smoke 建立同规模 baseline，避免只看单次 Concordia 指标无法判断好坏。

建议命令与任务二完全同规模，只把 backend 改成 `llm`：

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-backend llm \
  --simulator-model-key llm.simulator_model \
  --n 6 \
  --seeds 17 \
  --generations 1 \
  --num-islands 2 \
  --children-per-island 1 \
  --shadow-surveys 2 \
  --validation-shadow-surveys 1 \
  --test-shadow-surveys 1 \
  --items-per-shadow-survey 6 \
  --candidate-max-workers 1 \
  --persona-max-workers 1 \
  --shadow-max-workers 2 \
  --output-dir data/results/mega_persona_llm_clean_smoke_20260623
```

对比表至少记录：

| 指标 | LLM baseline | Concordia-style | Concordia-native（若已集成） | 判断 |
|---|---:|---:|---:|---|
| best validation fitness | 待填 | 待填 | 待填 | 哪个更高 |
| sealed test alignment | 待填 | 待填 | 待填 | 是否更稳 |
| sealed test behavior coverage | 待填 | 待填 | 待填 | 是否更分散 |
| valid persona rate | 待填 | 待填 | 待填 | 是否生成稳定 |
| timeout/error count | 待填 | 待填 | 待填 | 是否运行可靠 |

验收标准：

- 各组实验使用相同 `n`、`seed`、survey 数量、items 数量和 generation 配置。
- 只改变 simulator backend 或 simulator adapter。
- 如果 Concordia-style 的 coverage 更高但 alignment 大幅下降，需要标注为“探索性更强但一致性风险更高”。
- 如果 Concordia-style 与 LLM baseline 指标接近，但 timeout 更少或行为分布更合理，可以进入中等规模 A/B。

### 5.4 任务四：小型 simulator A/B 与 Concordia 集成对比

目标：在 smoke 通过后，扩大一点规模，同时比较 baseline LLM simulator、Concordia-style simulator 和 Concordia-native simulator，观察不同模拟器是否改变 operator 排名、行为覆盖度和 validation-test gap。

建议规模：

- n：`12`
- seeds：`17,23`
- generations：`2`
- islands：`4`
- children per island：`1`
- train/validation/test shadow surveys：`3/2/2`
- items per survey：`8`
- candidate 并发：`2`
- persona 并发：`2`
- shadow 并发：`4`

计划对比三组，第四组视时间和模型可用性决定：

| 组别 | Backend | 目的 |
|---|---|---|
| A | `llm` | 当前 baseline simulator |
| B | `concordia-style` | 测试基于 Concordia 思路的 prompt adapter |
| C | `concordia-native` | 测试原生 Concordia agent/component 集成 |
| D | `concordia-native + stronger simulator model` | 可选，用更强模型测试 Concordia-native 上限 |

对比指标：

- validation fitness
- sealed test alignment
- sealed test behavior coverage
- operator ranking 是否一致
- validation-test gap
- valid persona rate
- timeout/error rate
- simulator runtime

验收标准：

- 如果 Concordia-native 在 sealed test behavior coverage 上明显提升，并且 alignment 没有明显崩坏，则将 Concordia-native 保留为后续正式实验分支。
- 如果 Concordia-style 和 Concordia-native 指标接近，优先选择实现更稳定、运行成本更低的一支。
- 如果 Concordia-native 指标更好但 runtime 显著增加，需要单独评估是否只用于 final evaluation，而不用于每代进化选择。
- 如果 Concordia 相关 backend 指标不稳定或错误率高，则先回到 LLM simulator，Concordia 暂列为扩展项。
- 产出一份 batch 分析文档，建议命名为：

```text
docs/MEGA_PERSONA_BATCH_2026-06-24_SIMULATOR_AB.md
```

### 5.5 任务五：focused OpenEvolve 实验

目标：基于本周 operator 证据，提高搜索效率。

本周不再建议平均抽样所有 15 个 operator。下周 focused run 重点围绕已经出现正向信号的机制型 operator：

- `op06_low_axis_fidelity`
- `op04_within_bucket_contrast`
- `op09_low_high_axis_tradeoff`
- `op13_autonomy_pressure_test`
- `op14_recovery_latency`

建议策略：

- 提高上述 operator 的采样概率。
- 主体 mutation mode 使用 `operator_only`。
- 少量保留 `prompt_only` 做表达方式探索。
- 降低 `mixed` 和 `numeric_only` 的比例。
- 保留少量随机探索，避免过早收敛。

建议规模：

- n：`24`
- seeds：`17,23,31`
- generations：`3`
- islands：`6`
- children per island：`1`
- shadow surveys：`6/3/3`
- items per survey：`8`

验收标准：

- best candidate 不是 baseline。
- best validation fitness 相比 baseline 提升至少 `5%`。
- sealed test alignment 不低于 baseline 明显过多。
- sealed test behavior coverage 不低于上一轮同规模结果。
- operator 排名能支持下一次更大规模正式实验。

预期产出：

```text
data/results/mega_persona_openevolve_focused_20260624
docs/MEGA_PERSONA_BATCH_2026-06-24_FOCUSED_EVOLUTION.md
```

### 5.6 任务六：失败诊断与自动报告脚本

目标：减少每次实验后手工翻 `run.log`、`final_summary.json`、`result.json` 的成本，让实验分析更稳定。

计划增加一个分析脚本，例如：

```bash
python scripts/analyze_mega_persona_run.py data/results/<run_dir>
```

脚本至少输出：

- 每次 run 自动统计 error/timeout 数量。
- 自动统计每个 candidate 的有效人格生成率。
- 自动输出 operator/mutation mode 排名。
- 自动计算 baseline 到 best 的提升。
- 自动生成中文 batch report 草稿。

验收标准：

- 输入一个 run 目录即可生成 summary。
- 能识别 best candidate、baseline candidate、generation 数、candidate 数。
- 能统计 timeout/error 关键词。
- 能输出 validation/test gap。
- 能生成 Markdown 草稿，后续人工只需要补解释。

### 5.7 任务七：项目结果目录清理

目标：保留关键批次，清掉临时和失败产物，避免项目越来越难读。

计划保留：

- 中等规模 OpenEvolve 成功批次
- 较大规模 OpenEvolve 成功批次
- Concordia smoke 批次
- operator ablation 代表批次

临时 `default_*`、失败 smoke、重复中间结果可移到 archive 或删除，保持 `data/results` 清爽。

清理前要求：

- 先列出将要删除或归档的目录。
- 不删除已写入报告引用的关键结果。
- 清理后保留 `.gitkeep`。
- README/AGENTS/CLAUDE 中引用的结果目录要么存在，要么改成 `<run_name>` 占位。
