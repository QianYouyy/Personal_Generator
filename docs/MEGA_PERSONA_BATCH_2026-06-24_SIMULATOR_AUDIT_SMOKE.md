# MegaPersona Simulator Audit Smoke

日期：2026-06-24  
Batch：`mega_persona_simulator_audit_smoke_20260624`  
结果目录：`data/results/mega_persona_simulator_audit_smoke_20260624`

## 1. 实验目的

本批实验用于验证三种 shadow-survey simulator 的离线对比流程是否跑通，并观察初步质量差异：

- `llm`
- `concordia`
- `concordia-native`

该实验固定同一批人格与同一批 shadow surveys，不进入 OpenEvolve 进化闭环，因此只评估模拟器本身。

## 2. 实验设置

- persona mode：`mock`
- personas：`12`
- shadow surveys：`3`
- items per survey：`6`
- repeats：`1`
- 每个 backend 调用数：`12 × 3 = 36`
- 每个 backend 响应 item 数：`36 × 6 = 216`
- persona hash：`ee07aa2905e2f2998c4320c7f65480c6b958a9ab1eadedda29915c6b0c0c1417`
- survey hash：`5e018da01080990cbd17e33d9f05e2b55b5684dc0d998bf72af7b12d11910e57`

## 3. 主要结果

| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llm` | 0.3150 | 0.7341 | 1.0000 | 0.8098 | 0.1806 | 0.1322 | 1.05 |
| `concordia` | 0.2400 | 0.7158 | 1.0000 | 0.7774 | 0.1806 | 0.1142 | 0.74 |
| `concordia-native` | 0.1040 | 0.7056 | 1.0000 | 0.7671 | 0.1435 | 0.0290 | 0.82 |

三组均完成所有 item，说明接口、JSON 解析与 scoring 流程正常。

## 4. 初步观察

### 4.1 `llm` 暂时最好

`llm` 在本批 smoke 中同时取得最高：

- 行为覆盖度：`0.3150`
- 人格-行为对齐度：`0.7341`
- 响应熵：`0.8098`
- 行为区分度：`axis_std_mean = 0.1322`

这说明普通 LLM simulator 目前反而更能把不同人格拉开。

### 4.2 `concordia` 可用，但没有超过 `llm`

`concordia` 的合法率正常，速度最快，但覆盖度和区分度低于 `llm`：

- coverage 比 `llm` 低 `0.0750`
- alignment 比 `llm` 低 `0.0183`
- axis std 比 `llm` 低 `0.0180`

目前可以作为对照组，但还不能说明有质量优势。

### 4.3 `concordia-native` 出现行为压缩

`concordia-native` 的主要问题不是格式错误，而是行为空间明显压缩：

- coverage：`0.1040`
- axis std：`0.0290`
- mean pairwise distance：`0.0638`
- cognitive/motivation 相关性偏弱或为负

它的行为轴均值偏高：

- cognitive abstraction：`0.9097`
- motivation autonomy：`0.7618`
- self regulation resilience：`0.5162`

这说明当前 native agent/component 版本可能过度强调积极、自洽、理想化回答，导致不同人格之间区分不够。

## 5. 结论

本批 smoke 的结论是：

1. 三模拟器离线审计流程已经跑通。
2. 三组 backend 都能输出合法完整结果。
3. 当前 `llm` 是最稳的 baseline。
4. `concordia-native` 虽然已经接入原生 Concordia agent 生命周期，但当前 prompt/component/action spec 还需要校准，否则容易出现行为塌缩。

本批只是 smoke，`repeats=1` 且样本量较小，不能作为正式统计结论。

## 6. 下一步

建议先修正 `concordia-native` 的行为压缩问题，再跑 medium：

- 降低 component context 中的理想化倾向。
- 在 action prompt 中明确要求区分人格差异，不要默认积极回答。
- 增加 negative evidence 和 conflict memory 的权重。
- 在 native backend 中记录 agent trace，抽样检查回答依据。
- 使用 `repeats=2` 或 `3` 评估稳定性。

修正后再跑：

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

## 7. 后续修复记录

已针对 `concordia-native` 的行为压缩问题完成第一轮代码修正：

- 新增 `behavior_calibration` Concordia context component。
- 在 native acting prompt 中加入反理想化校准规则。
- 强调 blind spot、stress load、risk factors、peer influence、habit stability 等负向证据。
- 要求每个 item 独立校准，避免整份问卷复用同一种积极态度。
- 明确 1-5 分行为锚点，允许低分回答。
- 为测试增加 `behavior_calibration` 和 calibration prompt 断言，避免后续回归。

建议先用相同 frozen setting 复跑 smoke，检查 `concordia-native` 的 `coverage`、`axis_std_mean` 和 `mean_pairwise_distance` 是否回升。
