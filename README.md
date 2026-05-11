# Persona Generator

复刻论文《Persona Generators: Generating Diverse Synthetic Personas at Scale》核心 pipeline，将 AlphaEvolve 替换为自研 Open-Evolve 进化引擎。

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
├── configs/               # 配置文件
├── data/
│   ├── questionnaires/    # 预生成问卷
│   ├── generated_personas/# 生成的人格
│   └── results/           # 评估结果
├── docs/                  # 文档
├── scripts/               # 脚本
├── src/
│   ├── qgenerator/        # 模块1: 问卷生成器
│   ├── persona_generator/ # 模块2: 人格生成器
│   ├── simulator/         # 模块3: Concordia 模拟器
│   ├── evaluator/         # 模块4: 多样性评估器
│   ├── open_evolve/       # 模块5: Open-Evolve 进化引擎
│   └── utils/             # 工具函数
├── tests/                 # 测试
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt
```

## 分步实施规划

见 `docs/ROADMAP.md`
