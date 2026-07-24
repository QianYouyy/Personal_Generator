# MegaPersona Concordia Integration Notes

日期：2026-06-24

## 1. 集成目标

本次集成目标是把 MegaPersona 的 shadow survey simulator 从单纯 prompt-based role-play 扩展为可走 Concordia 原生 agent/component 生命周期的模拟器。

当前项目中有三类 simulator backend：

| Backend | 含义 | 状态 |
|---|---|---|
| `llm` | 直接用 LLM role-play persona 回答 shadow survey | 稳定 baseline |
| `concordia` | Concordia-style prompt adapter，模拟 agent/component 推理过程 | 已接入 |
| `concordia-native` | 使用 Concordia `EntityAgent`、`ContextComponent`、`ActingComponent` 的原生生命周期 | 本次新增 |

## 2. 新增代码

核心文件：

```text
src/mega_persona/concordia_adapter.py
```

新增内容：

- `LLMClientLanguageModel`
  - 将项目现有 `LLMClient.generate()` 包装成 Concordia `LanguageModel`。
- `StaticContextComponent`
  - 用于 identity、cognition、motivation、social context、health context 等静态人格组件。
- `EpisodicMemoryComponent`
  - 保存 MegaPersona 叙事片段和 survey observation，作为轻量 episodic memory。
- `SurveyActingComponent`
  - Concordia acting component，读取所有 component context 后输出 shadow survey JSON。
- `build_concordia_agent_bundle()`
  - 将 `MegaPersona` 构造为 Concordia `EntityAgent`。
- `survey_observation()` 和 `survey_action_spec()`
  - 将 shadow survey 转换为 Concordia observation/action request。

接入文件：

```text
src/mega_persona/shadow_simulator.py
```

新增：

- `ConcordiaNativeShadowSimulator`
- `build_shadow_simulator(..., backend="concordia-native")`

命令行入口：

```text
scripts/run_mega_persona_evolution.py
```

现在支持：

```bash
--simulator-backend concordia-native
```

## 3. MegaPersona 到 Concordia Agent 的映射

当前最小映射如下：

| MegaPersona 字段 | Concordia 组件 |
|---|---|
| demographics, values_identity | `identity` |
| thinking_style, learning_orientation, decision_pattern, challenge_response | `cognition` |
| motivation_system, self_regulation | `motivation` |
| social_creative_profile | `social_context` |
| mental_health_context | `health_context` |
| narratives, family context, aspiration, moral tension, academic tendency | `memory` |

这些组件会在 `agent.act()` 前通过 Concordia 的 `pre_act()` 生命周期汇总，然后交给 `SurveyActingComponent` 生成问卷 JSON。

## 4. 与 Concordia-style 的区别

`concordia` backend：

- 本质是一个 prompt adapter。
- 保留原有 `LLMShadowSimulator` 调用方式。
- prompt 中要求 LLM 按 Concordia 的 agent/component 思路思考。

`concordia-native` backend：

- 真实构造 Concordia `EntityAgent`。
- 使用多个 Concordia `ContextComponent`。
- 调用 `agent.observe()` 存储 survey scene。
- 调用 `agent.act()` 生成 survey response。
- 输出仍兼容项目的 `ShadowSurveySimulation`。

因此，`concordia-native` 更适合后续做“模拟器科学性”实验；`concordia` 可以作为低成本中间版本。

## 5. 最小运行命令

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-backend concordia-native \
  --simulator-model-key llm.simulator_model \
  --n 6 \
  --seeds 17 \
  --generations 1 \
  --num-islands 2 \
  --children-per-island 1 \
  --shadow-surveys 2 \
  --validation-shadow-surveys 1 \
  --test-shadow-surveys 1 \
  --items-per-shadow-survey 6 \
  --candidate-max-workers 1 \
  --persona-max-workers 1 \
  --shadow-max-workers 2 \
  --output-dir data/results/mega_persona_concordia_native_smoke_20260624
```

## 6. 验证状态

已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  src/mega_persona/concordia_adapter.py \
  src/mega_persona/shadow_simulator.py \
  src/mega_persona/__init__.py \
  scripts/run_mega_persona_evolution.py \
  scripts/test_mega_persona_experiment.py

PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_experiment.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_openevolve_adapter.py
```

新增测试覆盖：

- `ConcordiaNativeShadowSimulator` 可以构造 Concordia agent。
- mock LLM 可以通过 `agent.observe()` / `agent.act()` 生成 survey JSON。
- 输出可以被 `score_shadow_survey()` 正常转为 axis scores。

## 7. 后续建议

下一步建议按小规模 A/B 验证：

1. `llm`
2. `concordia`
3. `concordia-native`

对比指标：

- best validation fitness
- sealed test alignment
- sealed test behavior coverage
- valid persona rate
- timeout/error count
- simulator runtime

如果 `concordia-native` 行为覆盖更高但耗时明显增加，可以考虑只把它用于 final evaluation 或 simulator audit，而不是每代进化选择。
