# Persona Generator

复刻论文《Persona Generators: Generating Diverse Synthetic Personas at Scale》核心 pipeline，将 AlphaEvolve 替换为自研 **Open-Evolve** 进化引擎。

## 架构概览

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
│   └── ROADMAP.md           # 分步实施路线图
├── scripts/
│   ├── generate_questionnaires.py   # 生成 50 份问卷入口
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
│   └── utils/
│       ├── config.py        # 配置加载器
│       └── llm_client.py    # LLM 调用封装
├── main.py                  # 主入口：集成所有模块，一键运行进化
├── .env.development         # API Key（已加入 .gitignore）
├── .env                     # 空模板（可安全提交）
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
OPENAI_API_BASE=https://api.openai.com/v1
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

---

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

---

## 分步实施规划

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

---

## 安全提示

- **永远不要提交 `.env.development`**。已配置 `.gitignore` 自动排除。
- **不要提交生成的数据文件**（`data/**/*.json`）。
- `.env` 和 `.env.example` 可安全提交（不含真实 Key）。

---

## License

MIT
