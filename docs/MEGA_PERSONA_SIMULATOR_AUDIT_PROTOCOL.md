# MegaPersona 多模拟器离线审计实验

本文档对应“实验 1”：固定同一批人格与同一批 shadow surveys，比较 `llm`、`concordia`、`concordia-native`、`student-realistic`、`student-realistic-v2` 等模拟器的离线行为模拟质量。

## 实验目的

本实验只回答一个问题：

> 在相同人格、相同问卷、相同模型配置下，哪个 shadow-survey simulator 产生的行为数据更合法、更有区分度、更稳定，并且更符合人格结构预期？

该实验不包含 OpenEvolve 进化，因此不会混入“进化搜索是否偶然找到更好 candidate”的影响。

## 严格控制变量

- 固定 persona set：一次生成后写入 `frozen_personas.json`。
- 固定 survey set：一次生成后写入 `frozen_surveys.json`。
- 固定输入哈希：报告中记录 `persona_sha256` 与 `survey_sha256`。
- 所有 backend 使用同一批 persona-survey 矩阵。
- 若 `--repeats > 1`，同一 backend 重复回答同一矩阵，用于估计稳定性。

## 指标

核心指标：

- `complete_response_rate`：是否回答了所有题目。
- `valid_value_rate`：回答是否均为 1-5 的合法 Likert 值。
- `behavior_coverage`：行为轴空间覆盖度。
- `overall_alignment`：人格主轴与行为轴之间的平均对齐度。
- `axis_correlations`：每个人格轴与对应行为轴的相关性。
- `axis_std_mean`：不同行为人格之间是否拉开差异。
- `response_entropy`：Likert 响应分布是否退化。
- `neutral_rate`：是否过度集中在中间分。
- `axis_score_consistency`：重复运行时同一 persona-survey 的行为轴一致性。
- `seconds_per_call`：单次 persona-survey 模拟耗时。

## Smoke 命令

用于确认流程跑通：

```bash
python scripts/run_mega_persona_simulator_audit.py \
  --persona-mode mock \
  --n 12 \
  --persona-seed 17 \
  --survey-seed 17017 \
  --shadow-surveys 3 \
  --items-per-shadow-survey 6 \
  --repeats 1 \
  --backends llm,concordia,concordia-native \
  --simulator-model-key llm.simulator_model \
  --shadow-max-workers 3 \
  --output-dir data/results/mega_persona_simulator_audit_smoke_20260624
```

## Medium 命令

用于形成初步比较结论：

```bash
python scripts/run_mega_persona_simulator_audit.py \
  --persona-mode mock \
  --n 60 \
  --persona-seed 17 \
  --survey-seed 17017 \
  --shadow-surveys 8 \
  --items-per-shadow-survey 10 \
  --repeats 2 \
  --backends llm,concordia,concordia-native \
  --simulator-model-key llm.simulator_model \
  --shadow-max-workers 8 \
  --output-dir data/results/mega_persona_simulator_audit_medium_20260624
```

## Student-Realistic 对比命令

用于比较真实学生机制模拟器与现有 baseline，并加入盲评版 `student-realistic-v2`：

```bash
python scripts/run_mega_persona_simulator_audit.py \
  --persona-mode mock \
  --n 60 \
  --persona-seed 17 \
  --survey-seed 17017 \
  --shadow-surveys 8 \
  --items-per-shadow-survey 10 \
  --repeats 2 \
  --backends llm,concordia-native,student-realistic,student-realistic-v2 \
  --simulator-model-key llm.simulator_model \
  --shadow-max-workers 8 \
  --output-dir data/results/mega_persona_simulator_audit_student_realistic_v2_medium_20260624
```

`student-realistic` 会在每条 simulation 的 `metadata` 中保存：

- `student_state`
- `context_appraisal`
- `mechanism`

这些 trace 用于解释学生在当前情境下为何给出某种行为回答。

`student-realistic-v2` 进一步保存：

- `trait_vector`
- `response_style`
- `item_mechanisms`

同时，v2 的模拟 prompt 不暴露 `primary_axes` 或 `axis_weights`，因此更适合作为正式实验中的 clean baseline evaluator。

## Formal 建议

正式结论建议：

- `n >= 100`
- `shadow-surveys >= 12`
- `items-per-shadow-survey >= 12`
- `repeats >= 3`
- 使用相同 frozen hashes 报告结果

如果 `concordia-native` 质量更好但耗时明显更高，可以考虑只在 final evaluation 或 simulator audit 中使用它，而不是每一代进化都使用。

## 输出文件

脚本会输出：

- `manifest.json`：实验配置与 frozen input hash。
- `frozen_personas.json`：冻结人格集。
- `frozen_surveys.json`：冻结问卷集。
- `simulations/<backend>/repeat_XX.json`：每个 backend 每次重复的原始模拟结果。
- `summary.json`：完整机器可读结果。
- `summary.md`：汇报可读表格。
- `run.log`：控制台日志副本。
