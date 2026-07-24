# MegaPersona Student-Realistic Simulator Audit Medium

日期：2026-06-24  
Batch：`mega_persona_simulator_audit_student_realistic_medium_20260624`  
结果目录：`data/results/mega_persona_simulator_audit_student_realistic_medium_20260624`

## 1. 实验目的

本批实验用于验证新增的 `student-realistic` 模拟器是否比现有 `llm` 和 `concordia-native` 更适合作为后续人格进化实验的 shadow-survey evaluator。

`student-realistic` 的设计目标不是复刻 Concordia agent，而是把模拟过程显式拆成：

- 人格特质读取：思维方式、动机、自我调节、社交与心理健康字段。
- 情境评估：威胁、机会、外部评价、同伴压力、支持、模糊性、截止压力。
- 潜在学生状态：压力、疲劳、自主感、胜任感、社交安全感、任务兴趣、回避倾向、恢复能力。
- 题目反应策略：根据题目语义与潜在状态生成 Likert 回答。

因此它更接近“真实学生在具体问卷情境中的状态化反应”，而不是只让 LLM 直接扮演人格回答问卷。

## 2. 实验设置

本批使用与前一轮 medium v2 相同的 frozen input：

- persona mode：`mock`
- personas：`60`
- shadow surveys：`8`
- items per survey：`10`
- repeats：`2`
- backends：`llm`, `concordia-native`, `student-realistic`
- 每个 backend 每次 repeat 调用数：`480`
- 三个 backend 总调用数：`2880`
- 总 item response 数：`28800`
- persona hash：`9cd20ad32d7172a00ce4321f910d33cb89deeb394885615e4479f2aea2027af8`
- survey hash：`87aa73b7fd10441e8c335181f589d7c7cf20e51457edfe974bb11ba78c30244e`
- shadow max workers：`8`

日志检查：

- 三组 backend 均完成所有调用。
- 每个 repeat 均完成 `480/480` calls。
- 未发现 `ERROR`、`Traceback`、`failed` 或失败中断。
- 总耗时约 `1355.5s`，约 `22.6` 分钟。

## 3. 主要结果

| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm` | 0.3580 | 0.7599 | 1.0000 | 0.7722 | 0.2884 | 0.0986 | 0.53 | 0.9813 |
| `concordia-native` | 0.3365 | 0.7840 | 1.0000 | 0.7717 | 0.2620 | 0.0886 | 0.49 | 0.9772 |
| `student-realistic` | 0.3870 | 0.7942 | 1.0000 | 0.7863 | 0.2065 | 0.1113 | 0.39 | 0.9783 |

本批中 `student-realistic` 在四个关键指标上均为最高：

- Coverage：`0.3870`
- Overall alignment：`0.7942`
- Response entropy：`0.7863`
- Axis std：`0.1113`

同时 complete response rate 和 valid value rate 都是 `1.0000`，说明新增模拟器没有牺牲可解析性。

## 4. 与 Baseline 的差异

以 `llm` 为参照：

| Backend | Coverage Δ | Alignment Δ | Entropy Δ | Axis Std Δ |
|---|---:|---:|---:|---:|
| `concordia-native` | -0.0215 | +0.0241 | -0.0005 | -0.0100 |
| `student-realistic` | +0.0290 | +0.0342 | +0.0141 | +0.0127 |

这说明 `student-realistic` 不只是提升了 alignment，也同时提升了行为覆盖度和行为离散度。这个结果比前一轮 `llm` 与 `concordia-native` 的取舍更清楚：前一轮是各有优势，本轮 `student-realistic` 在主要指标上更均衡。

## 5. 每轴相关性

| Backend | cognitive_abstraction | motivation_autonomy | self_regulation_resilience |
|---|---:|---:|---:|
| `llm` | 0.1107 | 0.5665 | 0.7606 |
| `concordia-native` | 0.3489 | 0.4395 | 0.7842 |
| `student-realistic` | 0.4608 | 0.5164 | 0.9125 |

解读：

- `student-realistic` 的 cognitive correlation 最高，说明它更能把“思维方式/抽象水平”转化为问卷行为差异。
- `student-realistic` 的 self-regulation correlation 明显最高，达到 `0.9125`，说明潜在状态建模对自我调节类题目非常有效。
- motivation correlation 略低于 `llm`，但高于 `concordia-native`；考虑到 `student-realistic` 同时有更高 coverage 和 alignment，这个取舍是可以接受的。

## 6. 响应分布

| Backend | Neutral Rate | Entropy | Extreme Rate 特征 |
|---|---:|---:|---|
| `llm` | 0.2884 | 0.7722 | 偏中性，4 分较多 |
| `concordia-native` | 0.2620 | 0.7717 | 相对保守，覆盖略低 |
| `student-realistic` | 0.2065 | 0.7863 | 中间分更少，分布更展开 |

`student-realistic` 的 neutral rate 明显降低，但 entropy 反而升高，说明它不是简单地把回答推向极端，而是减少了默认中间分，并提高了 2/4/5 等反应的可区分度。

## 7. 阶段性结论

本批 medium audit 支持以下判断：

1. `student-realistic` 可以作为下一阶段更优先的主模拟器候选。
2. 相比 `llm`，它在 coverage、alignment、entropy、axis std 上同时提高。
3. 相比 `concordia-native`，它保留了较高 alignment，同时显著提高 coverage 和 axis correlation。
4. 它的优势来自显式的“人格特质 - 情境评估 - 潜在状态 - 题目反应”链条，这比单纯 prompt 扮演更可解释。
5. 但当前实验仍是 mock persona 上的 medium audit，不能直接作为最终结论；下一步需要在真实 LLM 生成 persona 上复测。

## 8. 后续建议

### 8.1 先把 `student-realistic` 作为主 evaluator 跑一次小型 OpenEvolve

建议先做一轮短实验，验证它是否能有效引导进化，而不只是 offline audit 表现好：

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-model-key llm.simulator_model \
  --simulator-backend student-realistic \
  --n 24 \
  --seeds 17,23 \
  --generations 2 \
  --num-islands 6 \
  --children-per-island 1 \
  --elite-count 2 \
  --shadow-surveys 6 \
  --validation-shadow-surveys 3 \
  --test-shadow-surveys 3 \
  --items-per-shadow-survey 10 \
  --max-workers 3 \
  --shadow-max-workers 8 \
  --output-dir data/results/mega_persona_openevolve_student_realistic_smoke_20260624
```

### 8.2 再做真实 persona audit

如果小型 OpenEvolve 结果正常，再用该轮生成的人格做一次 simulator audit，避免只在 mock persona 上判断模拟器优劣。

### 8.3 暂时不建议继续扩大 simulator 对比

本阶段目标不是证明某个模拟器绝对最优，而是选择一个足够好、可解释、能推动进化的 evaluator。当前结果已经足以支持进入下一阶段进化实验。

