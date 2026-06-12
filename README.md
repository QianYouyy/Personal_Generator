# Persona Generator

本项目最初复刻论文《Persona Generators: Generating Diverse Synthetic Personas at Scale》的核心 pipeline，并将 AlphaEvolve 替换为自研 **Open-Evolve** 进化引擎。

当前实验主线已经调整为：在 DeepMind 的“空间覆盖/进化优化”思想和 HACHIMI 的“结构化大人格/规则校验/Shadow Survey”思想之上，做一个新的 **Schema-Constrained + Coverage-Guided MegaPersona** 实验。也就是说，本项目不再以完整复刻 HACHIMI 为目标，而是把它作为参考框架，探索“高质量、合法、可控、且覆盖人格空间”的大人格生成方法。

## 当前实验路径

### 创新主线：MegaPersona-Evolve

```
目标分布 / 配额槽
    ↓
Primary Axes 撒点
    ↓
MegaPersona Schema + 多 Agent Prompt
    ↓
Symbolic Validator
    ↓
Shadow Survey / 行为问卷模拟
    ↓
Shadow Simulator: 人格 → Likert 回答 → 行为轴
    ↓
Train / Held-out Shadow Evaluation
    ↓
Coverage × Validity × Diversity × Behavior Alignment
    ↓
Durable Open-Evolve 优化采样 / 轴变换 / Prompt Profile
    ↓
manifest + checkpoint + per-candidate artifacts
```

核心变化：
- 不再把 HACHIMI 的“学业画像”作为主要组件；改为 **思维方式、动机、自我调节、心理韧性** 等更适合泛化人格实验的维度。
- DeepMind pipeline 保留为 baseline 和基础设施：问卷生成、模拟、覆盖度指标、Open-Evolve。
- 新实验的优化目标不只是“生成多样”，而是同时满足 **Schema 合法、语义不重复、目标空间覆盖好、问卷行为差异稳定**。
- Shadow Survey 可以参考 HACHIMI 的做法，但优先使用非学业构念：价值观、动机、自我调节、归属感、心理健康、创造力、风险倾向、社会关系等。
- Evolution 使用固定评分尺子和 held-out shadow surveys，避免候选通过”改评分规则”或”记住训练问卷”虚高。
- 评分公式统一为乘法门控乘积（batch 实验与 evolution 共用 `compute_experiment_score`）：
  ```
  score = schema_fitness × (0.5 + 0.5 × behavior_coverage) × (0.5 + 0.5 × shadow_alignment) × generation_rate
  ```
  设计意图：behavior coverage 或 shadow alignment 极低时会被门控因子惩罚（最低 ×0.5），确保行为塌缩或对齐差的候选无法获得高分。
- 长跑实验会保存 `manifest.json`、`checkpoint.json` 和每个候选的完整 `result.json`，支持断点续跑。

### 当前代码状态

| 模块 | 状态 | 说明 |
|------|------|------|
| DeepMind-style baseline | ✅ 可运行 | 原始问卷生成、人格生成、模拟、评估、Open-Evolve |
| MegaPersona Schema | ✅ 已加入 | `src/mega_persona/schema.py` |
| 认知-动机 Prompt | ✅ 已加入 | `src/mega_persona/prompts.py` |
| Slot Sampler | ✅ 已加入 | `src/mega_persona/slots.py` |
| Shadow Survey | ✅ 已加入 | `src/mega_persona/shadow_survey.py` |
| Shadow Simulator | ✅ 已加入 | `src/mega_persona/shadow_simulator.py` |
| Rule-based Baseline | ✅ 已加入 | `src/mega_persona/template_generator.py` |
| MegaPersona Schema Evaluator | ✅ 已加入 | `src/mega_persona/evaluation.py`（schema 层面：合法率、近重复、覆盖度、diversity） |
| Experiment Score & Batch Runner | ✅ 已加入 | `src/mega_persona/experiment.py` + `scripts/run_mega_persona_experiment.py` |
| Fixed MegaPersona Generator | ✅ 已加入 | `src/mega_persona/generator.py` |
| Durable Open-Evolve | ✅ 已加入 | `src/mega_persona/evolution.py`, `scripts/run_mega_persona_evolution.py` |
| Held-out Evaluation | ✅ 已加入 | Evolution fitness 使用 held-out shadow behavior |
| Prompt Profile Genome | ✅ 已加入 | LLM mode 下 evolution 可注入 prompt addendum |
| Parallel Evaluation | ✅ 已加入 | `--max-workers` 并行评估候选 |
| Experiment Manifest | ✅ 已加入 | `manifest.json` 记录命令、配置、Python/Git 状态 |
| Result Visualization | ✅ 已加入 | `src/mega_persona/visualization.py`, `scripts/visualize_mega_persona_results.py` |
| Symbolic Validator | ✅ 已加入 | `src/mega_persona/validator.py` |
| MegaPersona CLI | ✅ 已加入 | `scripts/generate_mega_personas.py` |
| Schema smoke test | ✅ 已加入 | `scripts/test_mega_persona_schema.py` |
| Experiment smoke test | ✅ 已加入 | `scripts/test_mega_persona_experiment.py` |
| Generator smoke test | ✅ 已加入 | `scripts/test_mega_persona_generator.py` |
| Runner smoke test | ✅ 已加入 | `scripts/test_mega_persona_runner.py` |
| Evolution smoke test | ✅ 已加入 | `scripts/test_mega_persona_evolution.py` |

## DeepMind Baseline 架构

```
问卷生成器(QGenerator)
    ↓ 输出 q=(c, D, I)
人格生成器(PersonaGenerator) ← 被优化的对象 φ
    ↓ 输入 c, D, N → 输出 P={p₁...pₙ}
Concordia 模拟器(Simulator)
    ↓ 输入 P, I → 输出 Z={z₁...zₙ} ⊂ ℝ^K
多样性评估器(Evaluator)
    ↓ 输入 Z → 输出 M(Z) = [m₁...m₆]
Open-Evolve 进化引擎
    ↓ 输入 M(Z), φ → 输出变异后的 φ'
```

## 目录结构

```
.
├── configs/
│   └── default.yaml         # 全局配置：所有阶段 model 统一在此管理
├── data/
│   ├── questionnaires/      # 预生成问卷（JSON，已加入 .gitignore）
│   ├── generated_personas/  # 生成的人格描述
│   └── results/             # 评估结果与进化日志
├── docs/
│   ├── ROADMAP.md           # 分步实施路线图
│   └── MEGA_PERSONA_EXPERIMENT.md # 新实验完整设计
├── scripts/
│   ├── generate_questionnaires.py   # 生成 50 份问卷入口
│   ├── generate_mega_personas.py     # 新实验: 生成 MegaPersona population
│   ├── run_mega_persona_experiment.py # 新实验: 多 seed batch runner
│   ├── run_mega_persona_evolution.py # 新实验: 可恢复 Open-Evolve 进化
│   ├── visualize_mega_persona_results.py # 新实验: 输出散点图/曲线/指标图
│   ├── test_mega_persona_schema.py  # MegaPersona Schema/Validator smoke test
│   ├── test_mega_persona_experiment.py # MegaPersona sampling/survey/evaluation smoke test
│   ├── test_mega_persona_generator.py  # MegaPersona multi-agent generator smoke test
│   ├── test_mega_persona_runner.py     # MegaPersona batch runner smoke test
│   ├── test_mega_persona_evolution.py  # MegaPersona durable evolution smoke test
│   ├── test_qgenerator.py           # 问卷生成器 Mock 测试
│   ├── test_persona_generator.py    # 人格生成器 Mock 测试
│   ├── test_evaluator.py            # 评估器 Mock 测试
│   ├── test_e2e_mock.py             # 端到端 Mock 测试
│   └── test_open_evolve.py          # Open-Evolve Mock 测试
├── src/
│   ├── qgenerator/          # 模块1: 问卷生成器 ✅
│   │   ├── generator.py
│   │   ├── prompts.py
│   │   └── fewshot_data.py
│   ├── persona_generator/   # 模块2: 人格生成器 ✅
│   │   ├── generator.py
│   │   ├── seeds.py
│   │   └── prompts.py
│   ├── simulator/           # 模块3: Concordia 模拟器 ✅
│   │   └── concordia_sim.py
│   ├── evaluator/           # 模块4: 多样性评估器 ✅
│   │   └── metrics.py
│   ├── open_evolve/         # 模块5: Open-Evolve 进化引擎 ✅
│   │   ├── engine.py
│   │   ├── mutator.py
│   │   ├── evaluator.py
│   │   └── code_templates.py
│   ├── mega_persona/        # 新实验: 结构化大人格 Schema/Prompt/Validator ✅
│   │   ├── schema.py
│   │   ├── prompts.py
│   │   ├── slots.py
│   │   ├── shadow_survey.py
│   │   ├── shadow_simulator.py
│   │   ├── template_generator.py
│   │   ├── evaluation.py
│   │   ├── generator.py
│   │   ├── evolution.py
│   │   ├── visualization.py
│   │   └── validator.py
│   └── utils/
│       ├── config.py        # 配置加载器
│       └── llm_client.py    # LLM 调用封装
├── main.py                  # 主入口：集成所有模块，一键运行进化
├── .env.development         # API Key（已加入 .gitignore）
├── .env.example             # 示例模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env.development
```

编辑 `.env.development`：

```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
```

> ⚠️ `.env.development` 已加入 `.gitignore`，**不会上传到 Git 仓库**。

---

## 快速开始

### 一键运行进化（主入口）

```bash
python main.py --generations 100
```

完整参数：

```bash
python main.py \
  --train data/questionnaires/train.json \
  --test data/questionnaires/test.json \
  --generations 100 \
  --eval-model llm.qgenerator_model
```

输出：
- `data/results/best_persona_generator.py` — 最优人格生成器代码
- `data/results/final_evaluation.json` — Test 集最终评估结果
- `data/results/checkpoint_gen_*.json` — 每轮 checkpoint

### 生成 50 份问卷

```bash
python scripts/generate_questionnaires.py
```

输出保存到 `data/questionnaires/`：
- `train.json` — 30 份
- `val.json` — 10 份
- `test.json` — 10 份

### 验证 MegaPersona Schema

```bash
python scripts/test_mega_persona_schema.py
python scripts/test_mega_persona_experiment.py
python scripts/test_mega_persona_generator.py
python scripts/test_mega_persona_runner.py
python scripts/test_mega_persona_evolution.py
```

这些脚本会检查：
- Primary Axes 是否能稳定抽取
- Validator 是否能通过合法样本
- Validator 是否能拦截明显崩坏的人格样本
- Slot Sampler 是否能产生配额槽和目标轴
- Shadow Survey 是否能投影到 Primary Axes
- Shadow Simulator 是否能把人格投影为 Likert 回答和行为轴
- MegaPersona batch 是否能计算合法率、重复率、覆盖度和 fitness
- 固定版多 Agent generator 是否能合并白板并产出合法 MegaPersona
- Batch runner 是否能多 seed 汇总并导出 JSON/Markdown
- Evolution 是否能持久化每次评估、写 manifest/checkpoint，并从 checkpoint resume

### 运行 MegaPersona 新实验

完整实验设计见 [docs/MEGA_PERSONA_EXPERIMENT.md](docs/MEGA_PERSONA_EXPERIMENT.md)。

先生成 slots 和 shadow surveys，不调用 LLM：

```bash
python scripts/generate_mega_personas.py --n 25 --seed 17 --dry-run
```

运行规则模板 baseline，不调用 LLM，但会生成完整人格、评估和 shadow behavior：

```bash
python scripts/generate_mega_personas.py --n 25 --seed 17 --mock
```

接入真实 LLM 生成 MegaPersona population：

```bash
python scripts/generate_mega_personas.py --n 25 --seed 17 --model-key llm.persona_model
```

运行多 seed batch 实验并导出 `summary.json` / `summary.md`：

```bash
python scripts/run_mega_persona_experiment.py \
  --mode mock \
  --n 25 \
  --seeds 17,23,31
```

运行可恢复的 MegaPersona Open-Evolve：

```bash
python scripts/run_mega_persona_evolution.py \
  --n 25 \
  --seeds 17,23,31 \
  --generations 20 \
  --population-size 8 \
  --max-workers 4 \
  --heldout-shadow-surveys 4 \
  --output-dir data/results/mega_persona_evolution_run
```

断线或中断后继续：

```bash
python scripts/run_mega_persona_evolution.py \
  --n 25 \
  --seeds 17,23,31 \
  --generations 20 \
  --population-size 8 \
  --max-workers 4 \
  --heldout-shadow-surveys 4 \
  --output-dir data/results/mega_persona_evolution_run \
  --resume
```

使用真实 LLM，并让 evolution 注入 prompt profile：

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --n 10 \
  --seeds 17 \
  --generations 5 \
  --population-size 4 \
  --max-workers 1 \
  --heldout-shadow-surveys 4 \
  --output-dir data/results/mega_persona_llm_evolution_run
```

将单次生成、batch summary 或 evolution run 转成 PNG 图（默认）：

```bash
python scripts/visualize_mega_persona_results.py \
  --input data/results/mega_persona_evolution_run
```

生成交互式 HTML 报告（Plotly.js，支持 3D 旋转缩放，浏览器直接打开）：

```bash
python scripts/visualize_mega_persona_results.py \
  --input data/results/mega_persona_evolution_run \
  --format html
```

`generate_mega_personas.py` 的输出默认保存到 `data/generated_personas/`，包含：
- 目标 slots
- 生成的人格 JSON
- 每个人格的 validator 报告
- population-level evaluation
- 初始 shadow surveys
- shadow survey responses
- persona-behavior alignment report

`run_mega_persona_experiment.py` 和 `run_mega_persona_evolution.py` 的输出默认保存到 `data/results/`。进化实验会额外保存：
- `manifest.json`
- `checkpoint.json`
- `final_summary.json`
- `final_summary.md`
- `candidates/candidate_*.json`
- `generations/generation_*.json`
- `evaluations/eval_*/result.json`
- `figures/*.png`（运行可视化脚本后生成）

---

## 当前进度与后续计划

### 已完成闭环

当前主实验闭环已经完成：

```text
slots
  -> mock / LLM MegaPersona generation
  -> schema validation
  -> train + held-out shadow surveys
  -> behavior simulation
  -> held-out evaluation
  -> durable Open-Evolve
  -> manifest/checkpoint/artifacts
```

可直接运行的主流程：

| 目标 | 命令 |
|------|------|
| 单次离线 baseline | `python scripts/generate_mega_personas.py --n 25 --seed 17 --mock` |
| 多 seed baseline | `python scripts/run_mega_persona_experiment.py --mode mock --n 25 --seeds 17,23,31` |
| 可恢复 evolution baseline | `python scripts/run_mega_persona_evolution.py --generator-mode mock --n 25 --seeds 17,23,31 --generations 20 --population-size 8 --max-workers 4 --heldout-shadow-surveys 4 --output-dir data/results/mega_persona_evolution_run` |
| LLM prompt-profile evolution | `python scripts/run_mega_persona_evolution.py --generator-mode llm --model-key llm.persona_model --n 10 --seeds 17 --generations 5 --population-size 4 --max-workers 1 --heldout-shadow-surveys 4 --output-dir data/results/mega_persona_llm_evolution_run` |
| 结果可视化 | `python scripts/visualize_mega_persona_results.py --input data/results/mega_persona_evolution_run` |

### 下一步建议

| 优先级 | 任务 | 目的 |
|--------|------|------|
| P1 | 加 LLM Shadow Simulator | 用 LLM 回答 shadow survey，和规则版 simulator 对照 |
| P2 | 更细粒度 prompt mutation | 从 coarse prompt profile 扩展到 agent-specific prompt fragments |
| P2 | 增加 held-out survey families | 用不同构念组合测试泛化能力 |
| P2 | 统计报告增强 | 多 run 均值/方差、best-vs-baseline 显著性和 ablation 表 |

## 配置驱动设计

所有阶段模型统一在 `configs/default.yaml` 管理：

```yaml
llm:
  qgenerator_model:  "gpt-4o"
  persona_model:     "gemma-3-27b-it"
  simulator_model:   "gpt-4o-mini"
  evaluator_model:   "gpt-4o"
  mutator_model:     "gpt-4o"
  feedback_model:    "gpt-4o"
```

---

## 模块详情

### 1. 问卷生成器 (QGenerator) ✅

**Few-shot 设计：** 4 份成熟问卷（BFI/DASS/SVO/NFCS）的真实数据作为代码示例，教 LLM "新问卷该怎么写"。

```python
from src.qgenerator.generator import QGenerator

llm = LLMClient.from_config("llm.qgenerator_model")
gen = QGenerator(llm)
q = gen.generate("2035年AGI导致失业", k_dimensions=2)
# → Questionnaire(context, dimensions, items)
```

### 2. 人格生成器 (PersonaGenerator) ✅

被优化的核心对象 **φ**，3 个初始种子：

| 种子 | Stage 1 策略 | Stage 2 策略 |
|------|-------------|-------------|
| **seed1** | 串行逐个生成，主动错开 | 形成性记忆 |
| **seed2** | 小批次自回归（每批 5 个） | 形成性记忆 |
| **seed3** | 准随机蒙特卡洛采样 | 形成性记忆 |

```python
from src.persona_generator.generator import PersonaGenerator

gen = PersonaGenerator(llm, seed="seed1")
personas = gen.generate(context, dimensions, n=25)
# → List[Persona]
```

### 3. Concordia 模拟器 (Simulator) ✅

适当性逻辑（Logic of Appropriateness）：
1. 识别情境 → 2. 回忆人格 → 3. 做出选择

**每道题后重置记忆**，防止顺序效应。

```python
from src.simulator.concordia_sim import ConcordiaSimulator

sim = ConcordiaSimulator(llm)
Z = sim.simulate(personas, questionnaire)
# → Z.shape = (N, K)
```

### 4. 多样性评估器 (Evaluator) ✅

6 个指标，统一适应度方向（越大越好）：

| 指标 | 算法 | 方向 |
|------|------|------|
| Coverage | 10,000 随机点球覆盖比例 | 越大越好 |
| ConvexHull | scipy ConvexHull 体积 | 越大越好 |
| AvgDist | 两两欧氏距离均值 | 越大越好 |
| MinDist | 最近点对距离 | 越大越好 |
| Dispersion | 最大空白区半径 | 取负值 |
| KL Divergence | 经验分布 vs Sobol 分布 | 取负值 |

```python
from src.evaluator.metrics import DiversityMetrics

metrics = DiversityMetrics(coverage_radius=0.2)
fitness = metrics.fitness(Z)
```

### 5. Open-Evolve 进化引擎 ✅

#### 5.1 架构

| 组件 | 说明 |
|------|------|
| **6 个岛屿** | 独立进化，每岛 6 个精英位（对应 6 个指标） |
| **变异算子** | LLM 改写 Python 代码（25 种策略，基于 AlphaEvolve 论文） |
| **评估器** | `exec()` 执行代码 → 生成人格 → 模拟 → 评估 |
| **灭绝机制** | 自适应间隔，差岛从最优岛复制精英 |
| **反馈机制** | 观察 empirical data 指导改进 |

```python
from src.open_evolve.engine import OpenEvolve

engine = OpenEvolve(mutator, evaluator, questionnaires)
best = engine.run(max_generations=100)
# → 最优人格生成器代码
```

#### 5.2 精英更新机制（论文原机制）

**核心原则：单科状元即可登基**

每个岛屿维护 **6 个独立精英位**，每个精英位对应一个指标：

| 精英位 | 指标 |
|--------|------|
| Elite-1 | Coverage |
| Elite-2 | Convex Hull |
| Elite-3 | Avg Distance |
| Elite-4 | Min Distance |
| Elite-5 | Dispersion |
| Elite-6 | KL Divergence |

**更新规则：**
- 新候选 φ' 评估后得到 6 个分数
- 只要 **任意一个指标** 打破历史纪录，就采用 φ'
- φ' 只替换它打破纪录的那个精英位，其他 5 个位置不变
- 被挤掉的旧代码直接淘汰

**示例：**
```
φ_G 的得分: Coverage=0.72, ConvexHull=0.35, AvgDist=2.05, ...
历史纪录:    Coverage=0.75, ConvexHull=0.30, AvgDist=2.10, ...
              ↓没破          ↓破了！         ↓没破
结果: φ_G 成为新的 Elite-2（Convex Hull 冠军），其他位置不变
```

#### 5.3 综合评分机制（全局最优 & 灭绝）

岛屿内部按单科更新，但**全局最优选择**和**灭绝时选最优岛屿**使用多指标综合评分：

| 指标 | 权重 | 说明 |
|------|------|------|
| Coverage | **40%** | 覆盖度，核心指标 |
| Convex Hull | 15% | 凸包体积，空间覆盖 |
| Avg Distance | 15% | 平均距离，分散度 |
| Min Distance | 10% | 最小距离，避免聚集 |
| Dispersion | 10% | 离散度 |
| KL Divergence | 10% | 越接近均匀分布越好 |

**使用场景：**
- `get_global_best()`: 选择全局最优代码（用于最终输出）
- `_get_best_island()`: 灭绝时选择最优岛屿（作为种子复制给其他岛屿）

**设计理由：**
- 岛屿内部保留论文机制（单科状元），鼓励多样性探索
- 全局层面使用综合评分，避免过度偏向单一指标

### 6. MegaPersona Schema 实验模块 ✅

新实验模块位于 `src/mega_persona/`，用于承接 HACHIMI-style 大人格生成，但不复刻其学业中心 schema。

当前核心组件：

| 文件 | 作用 |
|------|------|
| `schema.py` | 定义 MegaPersona 结构，包含人口统计、认知-动机、自我调节、价值观、社交创造力、心理健康等组件 |
| `prompts.py` | 定义认知-动机 Agent 的 prompt 模板 |
| `slots.py` | 生成配额槽，并在 Primary Axes 内做 Sobol 撒点 |
| `shadow_survey.py` | 定义非学业 shadow surveys，并将回答投影到主轴 |
| `shadow_simulator.py` | 规则版行为模拟器，将 MegaPersona 映射为 Likert 回答和行为轴 |
| `template_generator.py` | 规则模板 baseline，用于离线生成可比较的 MegaPersona population |
| `evaluation.py` | 聚合法率、覆盖度、距离、多样性和近重复惩罚 |
| `generator.py` | 固定版 HACHIMI-style 多 Agent 生成流水线 |
| `evolution.py` | MegaPersona 专用 Open-Evolve，进化采样/轴变换/shadow survey 选择/LLM prompt profile，并用 held-out shadow surveys 评估；评分公式与 batch 实验统一为乘法门控乘积 |
| `validator.py` | 定义硬规则校验，避免人格全高、全低、字段冲突、表现与动机不一致等问题 |

当前 Primary Axes：
- `cognitive_abstraction`: 抽象/具体思维倾向
- `motivation_autonomy`: 自主/外控动机倾向
- `self_regulation_resilience`: 自我调节与心理韧性

这些轴用于配额槽内的蒙特卡洛撒点、schema-level coverage、shadow behavior projection 和 held-out evaluation。

---

## 分步实施规划

### Baseline 路线

| Phase | 模块 | 状态 | 说明 |
|-------|------|------|------|
| Phase 1 | 问卷生成器 (50 份) | ✅ 完成 | BFI/DASS/SVO/NFCS few-shot，train/val/test 划分 |
| Phase 1 | 人格生成器 (3 seed) | ✅ 完成 | Stage 1→Stage 2，串行/小批次/蒙特卡洛 |
| Phase 2 | Concordia 模拟器 | ✅ 完成 | 适当性逻辑 + 记忆重置 + 回答映射 |
| Phase 2 | 多样性评估器 (6 指标) | ✅ 完成 | Coverage 半径校准 + 多问卷聚合 |
| Phase 3 | Open-Evolve 引擎 | ✅ 完成 | 岛屿 + 变异 + 灭绝 + 反馈 |
| **Phase 4** | **集成主入口** | ✅ **完成** | `main.py` 一键运行 + checkpoint + 日志 |
| Phase 5 | 可视化 | ✅ 完成 | 2D/3D 散点图 + 进化曲线 + 岛屿热力图 |
| Phase 5 | 单元测试 | ⏳ 待开始 | 补全测试覆盖 |

### 新实验路线：Schema-Constrained MegaPersona

| Phase | 目标 | 状态 | 关键动作 |
|-------|------|------|----------|
| Phase 0 | 明确创新定位 | ✅ 完成 | HACHIMI 作为参考项目，DeepMind 作为覆盖度/进化 baseline |
| Phase 1 | 重设大人格 Schema | ✅ 完成 | 将学业组件替换为思维方式、动机、自我调节、心理韧性 |
| Phase 2 | 设计 Shadow Survey | ✅ 完成初版 | 不直接沿用学业问卷，优先选价值观、动机、心理健康、归属感、风险、创造力等构念 |
| Phase 3 | 实现 Slot Sampler | ✅ 完成初版 | 配额槽 + Primary Axes 蒙特卡洛撒点 |
| Phase 4 | 固定版多 Agent 生成器 | ✅ 完成初版 | 先手写 Prompt 与约束，生成 50-100 个 MegaPersona 做人工检查 |
| Phase 5 | 评估器扩展 | ✅ 完成初版 | `Validity × Coverage × Diversity × NearDuplicatePenalty` |
| Phase 6 | Batch 实验 Runner | ✅ 完成初版 | 多 seed 运行，导出 JSON/Markdown 汇总报告 |
| Phase 7 | 接入 Open-Evolve | ✅ 完成初版 | 进化配额权重、轴变换、shadow survey seed 和 LLM prompt profile；固定评分尺子，用 held-out shadow surveys 评估，每次评估持久化，可 resume |
| Phase 8 | Manifest + 并行评估 | ✅ 完成初版 | `manifest.json` 记录实验环境；`--max-workers` 并行评估候选 |
| Phase 9 | 可视化 | ✅ 完成初版 | 进化曲线、slot/persona/behavior 空间散点、best genome、指标图 |
| Phase 10 | 统计报告 | ⏳ 下一步 | baseline 对照、ablation 表、多 run 显著性 |
| Phase 11 | LLM Shadow Simulator | ⏳ 下一步 | 用 LLM 生成 shadow survey 回答，提高行为模拟真实性 |

---

## 安全提示

- **永远不要提交 `.env.development`**。已配置 `.gitignore` 自动排除。
- **不要提交生成的数据文件**（`data/**/*.json`）。
- 只提交 `.env.example`；`.env` 和 `.env.development` 都不要提交。

---

## License

MIT
