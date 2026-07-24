# MegaPersona Simulator Audit Medium V2

日期：2026-06-24  
Batch：`mega_persona_simulator_audit_medium_v2_20260624`  
结果目录：`data/results/mega_persona_simulator_audit_medium_v2_20260624`

## 1. 实验目的

本批实验用于验证上一轮针对 `motivation_autonomy` 轴的模拟器优化是否有效。

优化内容包括：

- 在 simulator prompt 中显式区分“内在自主动机”和“外部压力驱动的努力”。
- 暴露 `intrinsic_motivation`、`external_pressure_sensitivity`、`failure_sensitivity` 等字段。
- 在 Concordia native 的 action spec 和 `behavior_calibration` component 中加入 motivation-autonomy rule。
- 对 recognition、security、belonging、avoidance、fear of failure 等外部/防御性动机进行中低分校准。

## 2. 实验设置

本批与 medium v1 使用相同 frozen input：

- persona mode：`mock`
- personas：`60`
- shadow surveys：`8`
- items per survey：`10`
- repeats：`2`
- 每个 backend 每次 repeat 调用数：`480`
- 三个 backend 总调用数：`2880`
- 总 item response 数：`28800`
- persona hash：`9cd20ad32d7172a00ce4321f910d33cb89deeb394885615e4479f2aea2027af8`
- survey hash：`87aa73b7fd10441e8c335181f589d7c7cf20e51457edfe974bb11ba78c30244e`

日志检查：

- 三组 backend 均完成所有调用。
- 每个 repeat 均完成 `480/480` calls。
- 未发现 `ERROR`、`Traceback`、JSON fallback 或失败中断。

总耗时约 `1124.8s`，约 `18.7` 分钟。

## 3. Medium V2 主要结果

| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm` | 0.3530 | 0.7595 | 1.0000 | 0.7717 | 0.2873 | 0.0993 | 0.34 | 0.9812 |
| `concordia` | 0.2845 | 0.7403 | 1.0000 | 0.7765 | 0.2356 | 0.0729 | 0.37 | 0.9827 |
| `concordia-native` | 0.3360 | 0.7858 | 1.0000 | 0.7722 | 0.2640 | 0.0891 | 0.46 | 0.9818 |

## 4. 与 Medium V1 对比

### 4.1 Motivation 轴显著改善

| Backend | V1 motivation corr | V2 motivation corr | Delta |
|---|---:|---:|---:|
| `llm` | 0.2389 | 0.5536 | +0.3146 |
| `concordia` | 0.2010 | 0.4271 | +0.2261 |
| `concordia-native` | 0.2140 | 0.4062 | +0.1922 |

这说明针对 motivation-autonomy 的校准是有效的。三组 backend 都更能区分“自主动机”与“外部压力驱动的努力”。

### 4.2 `concordia-native` 仍保持最高 alignment

| Backend | V2 Alignment |
|---|---:|
| `llm` | 0.7595 |
| `concordia` | 0.7403 |
| `concordia-native` | 0.7858 |

`concordia-native` 的 overall alignment 仍然最高，说明它在整体现象上仍是最贴合人格结构的模拟器。

### 4.3 但 `concordia-native` 的覆盖度略有回落

| Metric | Medium V1 | Medium V2 | Delta |
|---|---:|---:|---:|
| Coverage | 0.3475 | 0.3360 | -0.0115 |
| Axis Std | 0.0973 | 0.0891 | -0.0082 |
| Mean Pairwise Distance | 0.2256 | 0.2050 | -0.0206 |
| Entropy | 0.8282 | 0.7722 | -0.0561 |

motivation 校准提高了轴相关性，但也让 native backend 更保守，行为分布略微收缩。

### 4.4 `llm` 在 coverage 和 motivation 相关性上反超

`llm` V2：

- Coverage：`0.3530`
- motivation corr：`0.5536`
- Axis Std：`0.0993`

这是 V2 中 coverage 和 motivation correlation 的最高值。说明 shared LLM prompt 对 motivation 优化非常敏感，普通 LLM backend 在该轴上获得了明显收益。

## 5. 每轴相关性

| Backend | cognitive_abstraction | motivation_autonomy | self_regulation_resilience |
|---|---:|---:|---:|
| `llm` | 0.0964 | 0.5536 | 0.7543 |
| `concordia` | 0.1431 | 0.4271 | 0.7005 |
| `concordia-native` | 0.3752 | 0.4062 | 0.8082 |

解读：

- `llm` 在 motivation axis 上最好。
- `concordia-native` 在 cognitive abstraction 与 self-regulation resilience 上仍明显更好。
- `concordia-native` 的整体 alignment 最高，说明它的优势来自更均衡的三轴结构，而不是单轴极高。

## 6. 响应分布变化

V2 中三组 neutral rate 都高于 V1：

| Backend | V1 Neutral | V2 Neutral | Delta |
|---|---:|---:|---:|
| `llm` | 0.1856 | 0.2873 | +0.1017 |
| `concordia` | 0.1845 | 0.2356 | +0.0511 |
| `concordia-native` | 0.2518 | 0.2640 | +0.0122 |

这说明 motivation 校准让模型更谨慎，尤其是 `llm`。这对减少“泛积极回答”有帮助，但如果继续加强，可能导致中间分过多。

## 7. 阶段性结论

本批 medium v2 支持以下结论：

1. `motivation_autonomy` 的针对性优化有效，三组 backend 的 motivation correlation 均明显上升。
2. `concordia-native` 仍保持最高 overall alignment。
3. `llm` 在 coverage 和 motivation correlation 上表现最好。
4. `concordia-native` 在 cognitive abstraction 与 self-regulation resilience 上表现最好。
5. 当前最合理的判断不是“单一模拟器绝对胜出”，而是：
   - `llm` 更擅长动机轴区分和覆盖；
   - `concordia-native` 更擅长结构化人格-行为一致性；
   - `concordia` style 版本仍然不如两者。

## 8. 后续建议

### 8.1 不再继续加强 motivation prompt

V2 已经显著提高 motivation correlation，但也带来 entropy 下降和 neutral rate 上升。因此不建议继续强化中低分校准，否则可能造成行为过度保守。

### 8.2 做轻量回调

建议对 `concordia-native` 做轻量回调：

- 保留 motivation-autonomy rule。
- 弱化“中间分优先”的倾向。
- 在 action spec 中加入：如果正负证据明确，不要默认选择 3。
- 对 1/5 分增加使用条件，提升 entropy 和 coverage。

### 8.3 下一阶段实验

建议优先做 evolution backend A/B，而不是继续 simulator prompt 微调：

- A 组：`llm` simulator
- B 组：`concordia-native` simulator

原因：

- simulator audit 已经证明两者各有优势；
- 下一步真正关键的问题是：不同 evaluator 是否会引导 OpenEvolve 产生不同质量的 persona generator。

如果需要更正式的 simulator 结论，再跑 formal audit：

```bash
python scripts/run_mega_persona_simulator_audit.py \
  --persona-mode mock \
  --n 100 \
  --persona-seed 17 \
  --survey-seed 17017 \
  --shadow-surveys 12 \
  --items-per-shadow-survey 12 \
  --repeats 3 \
  --backends llm,concordia-native \
  --simulator-model-key llm.simulator_model \
  --shadow-max-workers 12 \
  --output-dir data/results/mega_persona_simulator_audit_formal_20260624
```
