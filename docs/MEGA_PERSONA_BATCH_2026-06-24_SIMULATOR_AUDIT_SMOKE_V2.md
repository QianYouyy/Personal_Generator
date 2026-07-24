# MegaPersona Simulator Audit Smoke V2

日期：2026-06-24  
Batch：`mega_persona_simulator_audit_smoke_v2_20260624`  
结果目录：`data/results/mega_persona_simulator_audit_smoke_v2_20260624`

## 1. 实验目的

本批实验用于验证 `concordia-native` 第一轮修复是否解决行为压缩问题。

修复内容包括：

- 增加 `behavior_calibration` Concordia context component。
- 在 native acting prompt 中加入反理想化校准规则。
- 显式引入 blind spot、stress load、risk factors、peer influence、habit stability 等负向证据。
- 要求逐 item 独立判断，避免整份问卷复用同一种积极态度。
- 明确 1-5 分行为锚点，允许低分回答。

## 2. 实验设置

本批与 V1 smoke 使用完全相同的 frozen input：

- persona mode：`mock`
- personas：`12`
- shadow surveys：`3`
- items per survey：`6`
- repeats：`1`
- 每个 backend 调用数：`12 × 3 = 36`
- 每个 backend 响应 item 数：`36 × 6 = 216`
- persona hash：`ee07aa2905e2f2998c4320c7f65480c6b958a9ab1eadedda29915c6b0c0c1417`
- survey hash：`5e018da01080990cbd17e33d9f05e2b55b5684dc0d998bf72af7b12d11910e57`

日志检查：三组 backend 均完成 `36/36` calls，未发现 `ERROR`、`Traceback` 或失败中断。

## 3. V2 主要结果

| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llm` | 0.3450 | 0.7571 | 1.0000 | 0.8059 | 0.1806 | 0.1438 | 0.77 |
| `concordia` | 0.2470 | 0.7198 | 1.0000 | 0.7780 | 0.1713 | 0.1081 | 0.87 |
| `concordia-native` | 0.3720 | 0.7793 | 1.0000 | 0.8277 | 0.2407 | 0.1391 | 0.69 |

## 4. V1 到 V2 的变化

### 4.1 `concordia-native` 显著改善

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| Coverage | 0.1040 | 0.3720 | +0.2680 |
| Alignment | 0.7056 | 0.7793 | +0.0737 |
| Entropy | 0.7671 | 0.8277 | +0.0606 |
| Axis Std | 0.0290 | 0.1391 | +0.1101 |
| Mean Pairwise Distance | 0.0638 | 0.3357 | +0.2719 |
| Sec/Call | 0.82 | 0.69 | -0.13 |

这说明第一轮修复基本解决了 V1 中的行为压缩问题。

### 4.2 响应分布更合理

`concordia-native` 的响应分布从 V1：

```text
1: 0, 2: 40, 3: 31, 4: 109, 5: 36
```

变为 V2：

```text
1: 0, 2: 50, 3: 52, 4: 82, 5: 32
```

V2 明显减少了默认 4 分倾向，增加了 2/3 分，说明模型不再把人格普遍模拟成积极、理想化的自我陈述。

### 4.3 行为轴均值不再过度偏高

`concordia-native` 行为轴均值从 V1：

- cognitive abstraction：0.9097
- motivation autonomy：0.7618
- self regulation resilience：0.5162

变为 V2：

- cognitive abstraction：0.5347
- motivation autonomy：0.5021
- self regulation resilience：0.3856

这更符合 shadow survey 的行为测量逻辑：它不应直接把人格描述中的价值观、理想和成长倾向翻译成高分，而应反映压力、习惯、风险因素和情境下的真实行为倾向。

## 5. 与 `llm` baseline 对比

V2 中 `concordia-native` 已经在部分指标超过 `llm`：

- Coverage：`0.3720` > `0.3450`
- Alignment：`0.7793` > `0.7571`
- Entropy：`0.8277` > `0.8059`
- Sec/Call：`0.69` < `0.77`

但 `llm` 的 `Axis Std` 略高：

- `llm`：0.1438
- `concordia-native`：0.1391

两者差距很小。当前 smoke 只能说明 `concordia-native` 修复有效，不能直接得出正式结论。

## 6. 仍需注意的问题

### 6.1 样本量仍小

本批只有：

- 12 个 personas
- 3 份 surveys
- repeats = 1

因此不能报告稳定性，也不能做置信区间意义上的正式比较。

### 6.2 cognitive abstraction 相关性仍为负

三组 backend 的 cognitive abstraction correlation 都为负：

| Backend | cognitive_abstraction corr |
|---|---:|
| `llm` | -0.2188 |
| `concordia` | -0.1243 |
| `concordia-native` | -0.1748 |

这可能说明：

1. 当前 cognitive abstraction 轴与 shadow survey item 的投影关系不够稳定。
2. mock persona 的 cognitive 维度表达和问卷项之间存在反向或混淆。
3. 需要检查 cognitive 相关 item 的 wording、direction、axis weights。

这不是 Concordia 单独的问题，而是三组共同出现的测量问题。

## 7. 结论

本批 V2 smoke 结论：

1. `concordia-native` 的行为压缩问题已明显改善。
2. 新增 `behavior_calibration` 和反理想化规则有效。
3. `concordia-native` 目前已经具备进入 medium simulator audit 的条件。
4. 下一步应扩大样本并加入 repeats，评估稳定性和置信区间。

建议下一步运行 medium：

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
