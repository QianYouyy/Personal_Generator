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
│   └── test_qgenerator.py           # Mock 测试（不调用 API）
├── src/
│   ├── qgenerator/          # 模块1: 问卷生成器
│   │   ├── generator.py
│   │   ├── prompts.py
│   │   └── fewshot_data.py  # 4 份成熟问卷的真实数据（BFI/DASS/SVO/NFCS）
│   ├── persona_generator/   # 模块2: 人格生成器（3 个 seed）
│   ├── simulator/           # 模块3: Concordia 模拟器
│   ├── evaluator/           # 模块4: 多样性评估器（6 指标）
│   ├── open_evolve/         # 模块5: Open-Evolve 进化引擎
│   │   ├── engine.py
│   │   └── mutator.py
│   └── utils/
│       ├── config.py        # 配置加载器（YAML 点号访问）
│       └── llm_client.py    # LLM 调用封装（支持 from_config）
├── .env                     # API Key（已加入 .gitignore，不上传）
├── .env.example             # 环境变量模板（可安全提交）
├── .gitignore               # 保护密钥和数据
├── requirements.txt
└── README.md
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 当前阶段仅需 `openai` 和 `python-dotenv`，完整依赖（含 Concordia）在后续阶段安装。

### 2. 配置 API Key

创建开发环境配置文件并填入你的 Key：

```bash
cp .env.example .env.development
```

编辑 `.env.development`：

```bash
OPENAI_API_KEY=sk-your-key-here
# 如使用第三方中转，修改 base URL
OPENAI_API_BASE=https://api.openai.com/v1
```

> ⚠️ `.env.development` 已加入 `.gitignore`，**不会上传到 Git 仓库**。
> 
> 加载优先级：`.env.development` > `.env`
> - `.env.development`：存放真实 API Key（已忽略，不上传）
> - `.env`：空模板（可安全提交到仓库）
> - `.env.example`：示例模板（可安全提交）

---

## 配置驱动设计

**所有阶段使用的模型统一在 `configs/default.yaml` 管理**，代码中零硬编码。修改一处即可切换全局模型：

```yaml
llm:
  qgenerator_model:  "gpt-4o"           # Phase 1: 问卷生成
  persona_model:     "gemma-3-27b-it"   # Phase 2: 人格文本生成
  simulator_model:   "gpt-4o-mini"      # Phase 2: Agent 内部推理
  evaluator_model:   "gpt-4o"           # Phase 2: 一致性校验
  mutator_model:     "gpt-4o"           # Phase 3: 代码改写
  feedback_model:    "gpt-4o"           # Phase 3: 反馈分析
```

各模块温度、并行参数也在配置中：

```yaml
qgenerator:
  items_per_dimension: 5
  stage1_temperature: 0.8
  stage2_temperature: 0.7

open_evolve:
  num_islands: 10
  extinction_interval: 100
```

### 在代码中使用配置

```python
from src.utils.llm_client import LLMClient
from src.utils.config import get_config

# 从配置文件创建 LLMClient（推荐）
llm = LLMClient.from_config("llm.qgenerator_model")

# 读取任意配置项
cfg = get_config()
temp = cfg.get("qgenerator.stage1_temperature")  # 0.8
```

---

## 问卷生成器 (QGenerator)

### Few-shot 设计：与论文原文一致

论文 3.2 节指出，问卷生成器使用 **4 份成熟且公开可用的心理学问卷** 作为 few-shot 示例：

| 问卷 | 英文全称 | 测量内容 | 题项数 |
|------|---------|---------|--------|
| **BFI** | Big Five Inventory | 大五人格（开放性、尽责性、外向性、宜人性、神经质） | 15 题（示例子集） |
| **DASS** | Depression Anxiety Stress Scale | 抑郁、焦虑、压力 | 9 题（示例子集） |
| **SVO** | Social Value Orientation | 社会价值取向（利己/利他/竞争） | 6 题（情境选择题） |
| **NFCS** | Need for Closure Scale | 认知闭合需要（对确定性的偏好） | 12 题（示例子集） |

**实现方式：**

1. 从每份问卷中提取真实的 **上下文 c**、**多样性轴 D**、**题项 I**
2. 将它们转换成 **Concordia 兼容的 Python 代码格式**（可直接 `exec()` 执行）
3. 把这些代码作为 few-shot 示例喂给 LLM
4. 再给它一个新的简短描述 `ĉ`（如"2035 年 AGI 失业反应"），让它仿照这 4 个示例的结构生成新问卷

**本质：** 用"已有成熟量表的标准格式"教模型"新问卷该怎么写"。

代码见 [`src/qgenerator/fewshot_data.py`](src/qgenerator/fewshot_data.py)。

### 数据结构

```python
# 全局 5 点李克特量表
AGREEMENT_SCALE = [
    "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"
]

# Concordia 兼容的 Question 对象
@dataclass
class Question:
    preprompt: str      # 前置提示（如"Rate your agreement:"）
    statement: str      # 题项陈述（如"I have a vivid imagination."）
    choices: List[str]  # 选项列表（使用全局 AGREEMENT_SCALE）
    dimension: str      # 所属维度（英文标识符，如"openness"）

# 问卷结构
@dataclass
class Questionnaire:
    context: str           # c: 详细上下文
    dimensions: List[str]  # D: 多样性轴，K ∈ {2, 3}
    items: List[Question]  # I: 题项列表（扁平列表，每个 Question 含 dimension 字段）
```

### 生成流程（两步提示）

**Step 1 — 扩展上下文 & 提议多样性轴：**
- 输入：简短描述 `ĉ` + 4 份 few-shot 示例代码
- 输出：JSON `{context: str, dimensions: List[str]}`
- 维度命名：英文 snake_case（如 `agi_threat_appraisal`）

**Step 2 — 生成题项：**
- 输入：上下文 `c` + 单个维度 `dim`
- 输出：可直接 `exec()` 的 Python 代码
- 代码必须定义 `questions = [Question(...), ...]`
- 所有题项使用统一的 `AGREEMENT_SCALE`

---

## 快速开始

### Mock 测试（不调用 API，验证代码结构）

```bash
python scripts/test_qgenerator.py
```

输出示例：

```
============================================================
问卷生成器 - Mock 测试
============================================================
✅ 生成成功!
  多样性轴 (2 个):
    • 适应倾向 (Adaptability)
    • 风险承受 (RiskTolerance)
  题项:
    [适应倾向] 5 题
      - [正向] 我会积极学习 AGI 相关的新技能...
      - [反向] 我宁可降低生活标准...
  Concordia 格式转换: 10 个 Question 对象
============================================================
所有测试通过!
============================================================
```

### 生成 50 份问卷（需配置 API Key）

```bash
python scripts/generate_questionnaires.py
```

输出保存到 `data/questionnaires/`：
- `train.json` — 30 份
- `val.json` — 10 份
- `test.json` — 10 份

---

## 分步实施规划

| Phase | 模块 | 状态 | 说明 |
|-------|------|------|------|
| Phase 1 | 问卷生成器 (QGenerator) | ✅ 已完成 | 4 份成熟问卷作为 few-shot，两阶段生成，50 份问卷 |
| Phase 1 | 3 个 Seed 人格生成器 | 🚧 骨架待实现 | seed1/2/3 的 Stage 1/2 策略 |
| Phase 2 | Concordia 模拟器 | 🚧 骨架待实现 | Agent 实例化 + 适当性逻辑 |
| Phase 2 | 多样性评估器 | 🚧 骨架待实现 | 6 指标 + Coverage 半径校准 |
| Phase 3 | Open-Evolve 引擎 | 🚧 骨架待实现 | 岛屿系统 + 变异算子 + 灭绝机制 |
| Phase 4 | 集成 & 可视化 | ⏳ 待开始 | 主循环 + checkpoint + 图表 |
| Phase 5 | 测试 & 文档 | ⏳ 待开始 | 单元测试 + API 文档 |

详细规划见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

---

## 安全提示

- **永远不要提交 `.env` 文件**。已配置 `.gitignore` 自动排除。
- **不要提交生成的数据文件**（`data/**/*.json`、日志等）。
- 如需分享配置模板，使用 `.env.example`（不含真实 Key）。

---

## License

MIT
