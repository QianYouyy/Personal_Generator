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
│   │   └── prompts.py
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

复制模板并填入你的 Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
OPENAI_API_KEY=sk-your-key-here
# 如使用第三方中转，修改 base URL
OPENAI_API_BASE=https://api.openai.com/v1
```

> ⚠️ `.env` 已加入 `.gitignore`，**不会上传到 Git 仓库**。

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
| Phase 1 | 问卷生成器 (QGenerator) | ✅ 已完成 | 两阶段生成，50 份问卷，train/val/test 划分 |
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
