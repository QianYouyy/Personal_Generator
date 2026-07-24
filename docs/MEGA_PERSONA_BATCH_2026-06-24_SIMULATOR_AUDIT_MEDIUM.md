# MegaPersona Simulator Audit Medium

日期：2026-06-24  
Batch：`mega_persona_simulator_audit_medium_20260624`  
结果目录：`data/results/mega_persona_simulator_audit_medium_20260624`

## 1. 实验目的

本批实验用于在更大样本上比较三种 shadow-survey simulator：

- `llm`
- `concordia`
- `concordia-native`

与 smoke V2 相比，本批扩大了 persona 和 survey 数量，并加入 `repeats=2`，用于观察模拟器的稳定性。

## 2. 实验设置

- persona mode：`mock`
- personas：`60`
- shadow surveys：`8`
- items per survey：`10`
- repeats：`2`
- backends：`llm, concordia, concordia-native`
- 每个 backend 每次 repeat 调用数：`60 × 8 = 480`
- 每个 backend 总调用数：`480 × 2 = 960`
- 三个 backend 总调用数：`2880`
- 总 item response 数：`2880 × 10 = 28800`
- persona hash：`9cd20ad32d7172a00ce4321f910d33cb89deeb394885615e4479f2aea2027af8`
- survey hash：`87aa73b7fd10441e8c335181f589d7c7cf20e51457edfe974bb11ba78c30244e`

日志检查：

- 三组 backend 均完成所有调用。
- 每个 repeat 均完成 `480/480` calls。
- 未发现 `ERROR`、`Traceback`、JSON fallback 或失败中断。

总耗时约 `1263.0s`，约 `21.1` 分钟。

## 3. 主要结果

| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm` | 0.3200 | 0.7451 | 1.0000 | 0.8403 | 0.1856 | 0.0978 | 0.46 | 0.9802 |
| `concordia` | 0.2830 | 0.7271 | 1.0000 | 0.8204 | 0.1845 | 0.0722 | 0.45 | 0.9764 |
| `concordia-native` | 0.3475 | 0.7897 | 1.0000 | 0.8282 | 0.2518 | 0.0973 | 0.40 | 0.9753 |

## 4. 关键发现

### 4.1 `concordia-native` 在主要质量指标上最好

相对 `llm` baseline：

| Metric | `llm` | `concordia-native` | Delta |
|---|---:|---:|---:|
| Coverage | 0.3200 | 0.3475 | +0.0275 |
| Alignment | 0.7451 | 0.7897 | +0.0446 |
| Axis Std | 0.0978 | 0.0973 | -0.0005 |
| Mean Pairwise Distance | 0.2237 | 0.2256 | +0.0019 |
| Sec/Call | 0.4610 | 0.4012 | -0.0598 |

`concordia-native` 的覆盖度和人格-行为对齐度均高于 `llm`，行为区分度与 `llm` 基本持平，且单次调用更快。

### 4.2 `concordia` style 版本仍然较弱

`concordia` style prompt adapter 在本批实验中低于 `llm` 与 `concordia-native`：

- Coverage：`0.2830`
- Alignment：`0.7271`
- Axis Std：`0.0722`

这说明仅靠 Concordia-style prompt 不足以带来明显收益。真正有效的是 native agent/component + calibration 的组合。

### 4.3 `concordia-native` 的行为塌缩问题已基本解决

对比 smoke V1：

- V1 `concordia-native` coverage：`0.1040`
- Medium `concordia-native` coverage：`0.3475`
- V1 axis std：`0.0290`
- Medium axis std：`0.0973`

中等规模下仍保持较高 coverage，说明第一轮修复不是小样本偶然。

### 4.4 三组稳定性都较高

重复一致性：

| Backend | Axis Consistency | Response Consistency |
|---|---:|---:|
| `llm` | 0.9802 | 0.9750 |
| `concordia` | 0.9764 | 0.9728 |
| `concordia-native` | 0.9753 | 0.9711 |

`concordia-native` 稳定性略低于 `llm`，但差距很小，且仍处于可接受范围。

## 5. 每轴相关性

| Backend | cognitive_abstraction | motivation_autonomy | self_regulation_resilience |
|---|---:|---:|---:|
| `llm` | 0.0985 | 0.2389 | 0.7632 |
| `concordia` | 0.1623 | 0.2010 | 0.6761 |
| `concordia-native` | 0.4757 | 0.2140 | 0.8357 |

重要变化：

- smoke 中 cognitive abstraction 相关性为负；medium 中三组均转为正。
- `concordia-native` 在 cognitive abstraction 与 self-regulation resilience 上明显高于其他组。
- motivation autonomy 相关性三组都偏低，说明该轴或对应 shadow-survey items 仍需后续检查。

## 6. 响应分布

### `llm`

两次 repeat 合计趋势接近：

- 1 分较少
- 4/5 分占比较高
- neutral rate：`0.1856`
- entropy：`0.8403`

### `concordia-native`

响应更保守：

- 2/3 分更多
- 5 分更少
- neutral rate：`0.2518`
- entropy：`0.8282`

这符合修复后的设计：native backend 会更多使用负向证据和情境压力，不再默认高分。但需要注意，neutral rate 高于 `llm`，正式实验中要继续观察是否产生过度保守倾向。

## 7. 结论

本批 medium 实验支持以下阶段性结论：

1. 三模拟器离线审计流程稳定可用。
2. `concordia-native` 在中等规模下取得最佳 coverage 和 alignment。
3. `concordia-native` 的行为塌缩问题已基本修复。
4. `concordia` style 版本作为中间方案价值有限，后续重点应放在 native backend。
5. 仍需检查 motivation autonomy 轴的测量敏感性。

当前可以进入下一阶段：

- 继续做更大规模 simulator audit；
- 或者用 `concordia-native` 作为 evaluator 跑一次小规模 OpenEvolve 对比实验。

## 8. 下一步建议

### 8.0 已完成的针对性优化

根据本批结果，`motivation_autonomy` 相关性在三组 backend 中均偏低。因此已对模拟器进行一次针对性优化：

- 在 shared LLM simulator prompt 中显式区分“内在自主动机”和“外部压力驱动的努力”。
- 在 persona prompt 中暴露 `intrinsic_motivation`、`external_pressure_sensitivity`、`failure_sensitivity` 等动机字段。
- 在 Concordia native system prompt、action spec 和 `behavior_calibration` component 中加入 motivation-autonomy rule。
- 强调 growth、curiosity、creative efficacy、effort 类 item 不能仅凭“会努力”给高分，而要判断是否来自 internal interest 和 self-endorsed goals。
- 对 recognition、security、belonging、avoidance、fear of failure 等动机加入低/中分校准。

下一次 simulator audit 应重点观察：

- `motivation_autonomy` axis correlation 是否上升；
- `neutral_rate` 是否过度升高；
- `coverage` 与 `alignment` 是否保持不退化。

### 8.1 Simulator Formal Audit

建议正式版：

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

### 8.2 Evolution Backend A/B

如果暂时不跑 formal，可以先做进化闭环小实验：

- A 组：`--simulator-backend llm`
- B 组：`--simulator-backend concordia-native`

保持相同 `n`、`seeds`、`num-islands`、`generations`、`children-per-island` 和 frozen shadow surveys，比较最终 generator 在同一 test set 上的表现。
