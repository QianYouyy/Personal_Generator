# Goal-Conditioned Generator + Persona Archive 设计方案

记录日期：2026-07-24
状态：**方案已确认方向，尚未实现**

本文档记录下一阶段的主设计：用 MAP-Elites 风格的 Persona Archive 作为课程与簿记层，
进化一个 goal-conditioned 大人生生成器 Gθ，并最终以"清空 archive 重新生成全空间"
作为泛化验证协议。

---

## 1. 动机：为什么换评估结构

2026-07-24 中长程 run（`mega_persona_v3_fixed_deficit_rotation_single_call_deepseek_n8_g15_seed17_23_20260724`）确认：

- 平台期形态未变：gen0 基线 0.2315 → gen2 0.2632 → gen11 0.2671（+0.004，在噪声带内），
  全部可分辨收益在前 2 代拿完（噪声下限 2σ≈0.009–0.017，见该 run 的 `noise_floor/`）。
- MCTS 六个 v3 算子 mean reward 全为负：多数子代不如父代，搜索原地打转。
- 根因判断：单个 genome 的多指标被压成一个标量 fitness，n=8 评估分辨率下信号被噪声淹没。

第 0 步行为格子分析（`scripts/report_behavior_grid_occupancy.py`，报告在该 run 的 `behavior_grid/`）：

- 3 轴 × 3 桶 = 27 格；slot 采样器要求覆盖 23 格，池子实际覆盖 21 格（77.8%）。
- 但 global best 候选只覆盖 9 格 = 池子并集的 42.9%；候选中位数 7 格。
- MAP-Elites 视角组装 archive：21 格、平均最优质量 0.876（1-MAE），
  需要 **21 个不同候选** 各自贡献 persona——single-best 选择把池子里的领域专家全部丢弃。
- 占用极不均匀：最热格子吞了约 30% 的 persona（587/1931），5 个格子仅 2–3 次命中
  （极端组合如 `(0,0,0)`、`(1,0,2)` 疑似不可达）。
- 行为压缩：realized/target 标准差比 = 0.77 / 0.51 / 0.53，
  motivation 与 self-regulation 轴只交付被要求 spread 的一半。
- 占用增长 gen1→17 格后 14 代只爬到 21 格：没有针对新格子的选择压力，新格子只是副产品。

结论：搜索能反复产生领域专家，但单点选择 + 标量 fitness 留不住、也引导不到新格子。

## 2. 最终设计（已确认）

| 要素 | 设计 |
|---|---|
| 进化对象 | Goal-Conditioned Generator Gθ（genome 仍是进化表面） |
| 格子定义 | 由 target axis 定义目标任务（3 轴 × N 桶；先用 2 桶 8 格跑通，再加细到 3 桶 27 格） |
| 行为验证 | 用 observed axis（shadow simulator 的 axis_scores）验证是否真正到达目标 |
| Cell quality | persona 质量 − 目标误差 − 评估不确定性（不确定性 v1 降级为低观测降权，见 4.2） |
| Archive 用途 | 课程选择、能力诊断、成功案例缓存 |
| 最终交付 | **Generator Gθ，而不是 archive population** |
| 最终验证 | 清空 archive 后重新生成整个人格空间（detachment 测试） |

理论谱系：PGA-ME（archive 辅助训练 conditioned policy）+ Go-Explore（archive 作状态缓存，
清档重测检验能力内化）。区别仅在于 conditioned policy 的训练方式从梯度换成 LLM 变异 genome。

核心价值：**信用分配变密**。reward 从"一个 genome 一个标量"变成
"目标格子 X → observed 是否到达 X"，每次评估有明确因果结构，
genome 的 blueprint/axis-expression 编辑可与 per-cell 结果对齐，
变异有效性审计（declared/actual edits）可继续沿用。

## 3. 架构落点（关键约束）

不引入第二个进化机制。落点在 **evaluator 层**：

```text
OpenEvolve 岛 + MCTS 主循环（不动）
  ↓
Evaluator 内部：
  课程 slot 采样（替代随机 quota 采样）
    → 生成 + shadow 评估（复用现有链路）
    → per-cell 命中簿记（PersonaArchive）
    → coverage-aware fitness + 新格子软 bonus
```

- slot 采样器本来就产 `target_axes`，只需把"随机 quota"换成"课程加权"
  （未访问格子 + 弱格子 + 少量随机保持探索）。每个子代仍评估 8 个目标，
  **单次评估成本不变**，信号密度完全不同。
- Archive 是 evaluator 内部簿记，phenotype 去重、变异审计、噪声下限工具全部继续可用。
- 不违反 AGENTS.md "避免第二个进化机制" 的约束。

## 4. 开工前已敲定的决策

1. **格子可达性预算**：每格失败 K 次（暂定 12 次）后标 `suspected_unreachable`，
   降权并单独报告，防止课程死磕不可达格子烧光预算。
2. **不确定性项降级**：v1 用 `quality − target_error` 为主；
   观测次数 < n_min 的格子只降权不惩罚；不确定性记日志，
   若格子排名在重复评估间翻转再加连续惩罚项。
3. **reward 形状**：fitness = 命中格子质量加权和 + 新格子首中软 bonus；
   bonus 是软加成而非硬门槛（吸取 structured reward 的 coverage/quality 互搏教训）。
4. **课程采样替代随机 slot 采样**：未访问/弱格子优先，保留少量随机探索。
5. **必须带基线**：最终验证协议同时跑对照 genome
   `openevolve_000086_a6337eb2f13c`（2026-07-24 run 的 global best），否则收益无法归因。
6. **成本中性**：单次子代评估成本不变；最终验证协议约等于 3–4 个子代评估，一次性。

## 5. 最终验证协议（论文 headline 评估）

1. 进化结束，冻结 Gθ*。
2. 清空 archive。
3. 对全部目标格子逐一查询 Gθ*（每格 n 个 persona），用 observed axis 测：
   - cell 命中率（observed 落入目标格子的比例）；
   - 平均目标 MAE；
   - 最低 persona 质量。
4. 对冻结的再生成 population 跑 sealed test（split integrity 不变）。
5. 与基线 genome 同协议对照。

## 6. 实现顺序（尚未开始）

1. `PersonaArchive` 模块（evaluator 内部簿记：per-cell 命中、质量、观测数、可达性标记）。
2. 课程 slot 采样（加权替代随机 quota）。
3. per-cell reward + coverage-aware fitness（软 bonus）。
4. 最终验证协议脚本 + `openevolve_000086` 基线对照。
5. 2 桶 8 格 smoke → 3 桶 27 格中长程。

相关记录：

- 行为格子分析脚本：`scripts/report_behavior_grid_occupancy.py`
- 证据 run：`data/results/mega_persona_v3_fixed_deficit_rotation_single_call_deepseek_n8_g15_seed17_23_20260724/`
  （`behavior_grid/`、`noise_floor/`）
