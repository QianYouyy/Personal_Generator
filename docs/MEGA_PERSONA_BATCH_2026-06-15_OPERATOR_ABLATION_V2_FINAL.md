# MegaPersona Operator Ablation V2 Final

日期：2026-06-15  
批次目录：`data/results/mega_persona_operator_ablation_v2_20260614`

## 1. 实验状态

本次 operator ablation 已完整跑完，共评估 52 个 candidate。

- 人格规模：`n=6`
- seeds：`17, 23`
- shadow surveys：train 3 份，validation 2 份，test 2 份
- 每份 shadow survey item 数：8
- parent replay：2 次
- 每个 operator × mode：2 次重复
- best candidate：`ablation_0039_op04_within_bucket_contrast_mixed_r02`
- best fitness：`0.225122`
- parent replay mean fitness：`0.205497`
- parent replay std：`0.001698`

## 2. 主要结论

这次实验支持一个比较明确的判断：新版 operator 有效果，但效果集中在少数 operator 上，不是所有 operator 都稳定变好。

最优结果来自：

- operator：`op04_within_bucket_contrast`
- mode：`mixed`
- fitness：`0.225122`
- 相对 parent replay mean：`+0.019626`
- schema fitness：`0.383166`
- validation behavior coverage：`0.332000`
- validation shadow alignment：`0.764655`

这个提升约为 parent replay std 的 11.6 倍，因此不像是单纯随机波动。

## 3. Operator 排名

按每个 operator 的平均 fitness 排名：

| Rank | Operator | N | Mean Fitness | Best Fitness | Delta vs Parent | 判断 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `op04_within_bucket_contrast` | 6 | 0.212417 | 0.225122 | +0.006921 | 稳定正向 |
| 2 | `op07_high_axis_cost` | 6 | 0.206709 | 0.212294 | +0.001212 | 轻微正向 |
| 3 | `op06_low_axis_fidelity` | 6 | 0.206615 | 0.210942 | +0.001118 | 轻微正向 |
| 4 | `op03_shadow_survey_alignment` | 6 | 0.205043 | 0.211965 | -0.000454 | 接近 parent |
| 5 | `op05_failure_recovery_cycle` | 6 | 0.204603 | 0.212074 | -0.000894 | 有个别好点 |
| 6 | `op08_validation_conservatism` | 6 | 0.196353 | 0.207396 | -0.009143 | 不稳定 |
| 7 | `op02_behavioral_evidence` | 6 | 0.192664 | 0.218873 | -0.012832 | 上限高但波动大 |
| 8 | `op01_axis_decoupling` | 6 | 0.153990 | 0.208681 | -0.051506 | 高风险 |

## 4. Mutation Mode 结果

按 mode 聚合：

| Mode | N | Mean Fitness | Best Fitness | Delta vs Parent | 解释 |
|---|---:|---:|---:|---:|---|
| `parent_replay` | 2 | 0.205497 | 0.207195 | 0.000000 | baseline |
| `operator_only` | 16 | 0.197964 | 0.212635 | -0.007532 | 中位数较高，但受崩坏样本拖累 |
| `prompt_only` | 16 | 0.197823 | 0.211543 | -0.007674 | 单独改 prompt 不够稳定 |
| `mixed` | 16 | 0.196111 | 0.225122 | -0.009386 | 上限最高，方差最大 |
| `numeric_only` | 2 | 0.178248 | 0.200240 | -0.027249 | 不建议单独使用 |

结论：`mixed` 是最有探索价值的模式，但需要配合 operator 选择和失败保护；`operator_only` 更保守，适合作为稳定探索分支。

## 5. Top Candidate

| Rank | Candidate | Mode | Operator | Fitness | Schema | Val Coverage | Val Alignment |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `ablation_0039_op04_within_bucket_contrast_mixed_r02` | mixed | op04 | 0.225122 | 0.383166 | 0.332000 | 0.764655 |
| 2 | `ablation_0014_op04_within_bucket_contrast_mixed_r01` | mixed | op04 | 0.218896 | 0.375460 | 0.328500 | 0.754584 |
| 3 | `ablation_0008_op02_behavioral_evidence_mixed_r01` | mixed | op02 | 0.218873 | 0.378160 | 0.338500 | 0.729316 |
| 4 | `ablation_0033_op02_behavioral_evidence_mixed_r02` | mixed | op02 | 0.213830 | 0.373125 | 0.304000 | 0.756954 |
| 5 | `ablation_0013_op04_within_bucket_contrast_operator_only_r01` | operator_only | op04 | 0.212635 | 0.372216 | 0.296000 | 0.763831 |

Top 5 中有 3 个来自 `op04_within_bucket_contrast`，说明该 operator 的收益不是偶然单点。

## 6. 失败与风险

本次没有发现重复人格问题：

- near duplicate rate mean：全体为 0
- validity rate mean：0.9615
- 非满分 validity candidate：3 个

低分/崩坏 candidate：

| Candidate | Mode | Operator | Fitness | Validity | 问题 |
|---|---|---|---:|---:|---|
| `ablation_0029_op01_axis_decoupling_operator_only_r02` | operator_only | op01 | 0.107469 | 0.5 | schema 与 alignment 明显崩 |
| `ablation_0030_op01_axis_decoupling_mixed_r02` | mixed | op01 | 0.000000 | 0.0 | 双 seed 全部失败 |
| `ablation_0031_op02_behavioral_evidence_prompt_only_r02` | prompt_only | op02 | 0.100390 | 0.5 | validity 半数失败 |

`op01_axis_decoupling` 应暂时降权或移出正式进化 operator pool。它的意图是让轴解耦，但实际容易破坏 schema 合法性和人格-行为对齐。

## 7. 是否存在过拟合信号

这次 train-validation gap 不像典型过拟合。

Top candidate 的 train/validation 表现：

- train behavior coverage：0.2725
- validation behavior coverage：0.3320
- train-validation coverage gap：-0.0595
- train shadow alignment：0.7621
- validation shadow alignment：0.7647
- alignment gap：-0.0025

也就是说 best candidate 不是只在 train survey 上变好，validation 反而更高。因此当前更像是“真实泛化改善”，不是 train shadow survey 过拟合。

但注意：validation shadow surveys 只有 2 份，重复数也只有 2 次，所以这个结论仍然是中等置信度，不是最终统计结论。

## 8. 后续建议

下一轮正式进化建议：

1. 保留并提高 `op04_within_bucket_contrast` 的采样概率。
2. 保留 `op02_behavioral_evidence`，但只允许以 `mixed` 或受控形式出现，因为它上限高但不稳定。
3. 降权或暂停 `op01_axis_decoupling`，除非增加 schema guard。
4. 减少 `numeric_only` 单独探索比例，它目前收益弱且容易拖低平均表现。
5. 正式 OpenEvolve 跑法中，让 `mixed` 作为主探索，`operator_only` 作为稳定分支。

一句话结论：新版 operator pool 已经比旧版本更有信号，其中 `op04_within_bucket_contrast` 是当前最值得进入正式 OpenEvolve 主流程的核心 operator。
