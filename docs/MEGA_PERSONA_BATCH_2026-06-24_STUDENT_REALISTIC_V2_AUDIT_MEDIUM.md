# MegaPersona Student-Realistic V2 Simulator Audit Medium

日期：2026-06-24  
Batch：`mega_persona_simulator_audit_student_realistic_v2_medium_20260624`  
结果目录：`data/results/mega_persona_simulator_audit_student_realistic_v2_medium_20260624`

## 1. 实验目的

本批实验用于检验新增的 `student-realistic-v2` 是否能在保持“盲评”前提下，超过或接近上一版 `student-realistic`。

v2 与 v1 的核心区别：

- v1：机制提示更强，会暴露 `primary_axes` 与 `axis_weights`，因此指标较高，但存在信息泄漏风险。
- v2：不暴露 `primary_axes` 与 `axis_weights`，只使用人格证据、情境评估、潜在状态、回答风格与题目机制解释，因此更适合作为 clean baseline。

## 2. 实验设置

本批实验使用与前面 medium audit 相同的 frozen input：

- persona mode：`mock`
- personas：`60`
- shadow surveys：`8`
- items per survey：`10`
- repeats：`2`
- backends：`llm`, `concordia-native`, `student-realistic`, `student-realistic-v2`
- 每个 backend 每次 repeat 调用数：`480`
- 总调用数：`3840`
- 总 item response 数：`38400`
- persona hash：`9cd20ad32d7172a00ce4321f910d33cb89deeb394885615e4479f2aea2027af8`
- survey hash：`87aa73b7fd10441e8c335181f589d7c7cf20e51457edfe974bb11ba78c30244e`
- shadow max workers：`8`

日志检查：

- 四组 backend 均完成所有调用。
- 每个 repeat 均完成 `480/480` calls。
- 未发现 `ERROR`、`WARNING`、`Traceback`、`failed` 或 fallback。
- 总耗时约 `1572.9s`，约 `26.2` 分钟。

## 3. 主要结果

| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm` | 0.3545 | 0.7589 | 1.0000 | 0.7732 | 0.2885 | 0.0969 | 0.37 | 0.9836 |
| `concordia-native` | 0.3375 | 0.7860 | 1.0000 | 0.7682 | 0.2629 | 0.0901 | 0.46 | 0.9815 |
| `student-realistic` | 0.3780 | 0.7930 | 1.0000 | 0.7849 | 0.2112 | 0.1100 | 0.39 | 0.9780 |
| `student-realistic-v2` | 0.3025 | 0.7723 | 1.0000 | 0.7810 | 0.2558 | 0.0723 | 0.41 | 0.9788 |

本批结果显示：

- `student-realistic` 仍然是综合表现最好的 backend。
- `student-realistic-v2` 虽然完整率正常，但 coverage 和 axis std 明显下降。
- v2 的 alignment 高于 `llm`，但低于 `concordia-native` 和 `student-realistic`。

## 4. V1 与 V2 对比

| Metric | `student-realistic` | `student-realistic-v2` | Delta |
|---|---:|---:|---:|
| Coverage | 0.3780 | 0.3025 | -0.0755 |
| Alignment | 0.7930 | 0.7723 | -0.0207 |
| Entropy | 0.7849 | 0.7810 | -0.0039 |
| Neutral | 0.2112 | 0.2558 | +0.0446 |
| Axis Std | 0.1100 | 0.0723 | -0.0377 |
| Mean Pairwise Distance | 0.2461 | 0.1595 | -0.0867 |

v2 最大问题不是 JSON 失败，也不是整体 entropy 极低，而是行为空间明显收缩：

- Coverage 从 `0.3780` 降到 `0.3025`。
- Axis Std 从 `0.1100` 降到 `0.0723`。
- Mean Pairwise Distance 从 `0.2461` 降到 `0.1595`。

这说明 v2 的回答更稳定、更干净，但个体差异被压扁了。

## 5. 每轴相关性

| Backend | cognitive_abstraction | motivation_autonomy | self_regulation_resilience |
|---|---:|---:|---:|
| `llm` | 0.1166 | 0.5331 | 0.7603 |
| `concordia-native` | 0.3846 | 0.4404 | 0.7832 |
| `student-realistic` | 0.4636 | 0.4971 | 0.9126 |
| `student-realistic-v2` | 0.3896 | 0.4979 | 0.7800 |

解读：

- v2 的 cognitive correlation 不差，接近 `concordia-native`。
- v2 的 motivation correlation 与 v1 基本持平。
- v2 最大损失在 self-regulation resilience，从 v1 的 `0.9126` 降到 `0.7800`。

这说明 v2 的 blind design 并没有完全失败，但它没有充分把压力、疲劳、回避、恢复能力转化成更强的行为差异。

## 6. 响应分布诊断

`student-realistic-v2` 的响应分布：

- Repeat 1：`1=0`, `2=775`, `3=1229`, `4=2222`, `5=574`
- Repeat 2：`1=1`, `2=778`, `3=1227`, `4=2205`, `5=589`

问题很明显：

- v2 几乎不用 `1` 分。
- `4` 分占比过高。
- `5` 分比 v1 更多，但低端反应不足。
- 中间分高于 v1。

这会导致两个后果：

1. 行为分布向积极自我报告偏移。
2. 低韧性、低动机、低认知灵活性人格无法被充分拉开。

## 7. 阶段性结论

本批实验支持以下结论：

1. `student-realistic-v2` 的运行稳定性是合格的，完整率与合法率均为 `1.0000`。
2. v2 的 clean baseline 思路是正确的，因为它避免了显式泄漏目标轴与 item weights。
3. 但当前 v2 版本不适合直接替代 v1 作为主 evaluator，因为 coverage 和 axis std 下降明显。
4. 当前后续主线仍建议使用 `student-realistic`，而不是 v2。
5. v2 应作为“科学性更强但需要再校准”的候选分支继续优化。

## 8. 后续优化方向

### 8.1 保留 blind 设计，但增强低端反应

v2 不应该重新暴露 `primary_axes` 或 `axis_weights`，否则会失去 clean baseline 的意义。应该改的是 response policy：

- 当 stress、avoidance、fatigue 高且 competence/support 低时，明确允许 `1/2`。
- 当 defensive orientation 高时，对成长、自主、坚持类题目不能默认给 `4`。
- 对 self-regulation items，要求把“想做”和“能持续做到”分开。

### 8.2 把 response style 从提示词变成更强约束

当前 v2 虽然提供了 `response_style`，但 LLM 没有充分执行。建议增加一层显式的 item-level response calibration：

- 先让代码根据 trait/state/context 为每个 item 生成 blind evidence profile。
- evidence profile 不包含 axis weights，只包含：
  - supportive evidence
  - contradictory evidence
  - likely response band：low / mixed / high
- prompt 要求 LLM 在该 band 内选择具体 1-5 分。

### 8.3 不立即用 v2 跑 OpenEvolve

当前 v2 会让行为空间变窄，用它做 evaluator 可能会削弱进化信号。因此建议：

- 短期：继续使用 `student-realistic` 跑 OpenEvolve。
- 中期：做 `student-realistic-v2.1`，修复低端反应不足与 self-regulation 区分不足。
- v2.1 通过 audit 后，再考虑替换主 evaluator。

## 9. 下一步建议

建议下一步不是扩大实验，而是先修 v2：

1. 新增 blind evidence profile。
2. 强化 low-score policy。
3. 对 self-regulation resilience 单独校准。
4. 再跑同规模 medium audit。

判断标准：

- Coverage 至少回到 `0.35+`。
- Axis Std 至少回到 `0.095+`。
- Alignment 不低于 `0.785`。
- self-regulation correlation 不低于 `0.85`。

