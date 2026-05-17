"""变异算子 - 使用 LLM 改写人格生成器代码."""

import random
from typing import List


MUTATION_PROMPTS = [
    "修改 Stage 1，要求覆盖极端坐标（如每个轴的 0.1 和 0.9 分位点）。在 prompt 中加入明确要求。",
    "将 Stage 2 改为生成核心信念和价值观，而非童年形成性记忆。",
    "添加第一人称叙述格式（'我倾向于...' 而非第三人称）。",
    "在 Stage 1 中加入反事实约束：'如果此人格处于对立面，会如何'。",
    "修改 Stage 2，要求生成具体的行为模式而非抽象描述。",
    "在 Stage 1 中使用聚类策略，确保人格分布在不同象限。",
    "将 Stage 2 的输出改为结构化 JSON 格式。",
    "修改 Stage 1，先采样一个'锚点人格'，其余人格围绕它对称生成。",
    "增加 Stage 1 的温度参数，鼓励更多样化输出。",
    "修改 Stage 2，限制人格描述长度为 100-200 词，更精炼。",
    "在 Stage 1 prompt 中加入'避免中庸'的约束。",
    "修改 Stage 2，要求包含具体的社会角色和职业背景。",
    "在 Stage 1 中加入'对比生成'策略：先生成一个极端人格，再生成其对立面。",
    "修改 Stage 2，要求用 bullet points 列出人格特征。",
    "在 Stage 1 中加入多样性检查：'确保新的人格与已生成的在至少一个维度上差异显著'。",
]


MUTATION_SYSTEM_PROMPT = """You are an expert code mutation engine for evolutionary optimization.
Your task is to modify a Python persona generator class according to a given mutation instruction.

Rules:
1. You MUST preserve the class name "SeedPersonaGenerator"
2. You MUST preserve the method signatures: __init__, stage1, stage2
3. You MUST preserve the two-stage structure (stage1 → stage2)
4. You should ONLY modify the internal logic, prompts, and strategies inside stage1 and stage2
5. Output ONLY the complete modified Python code, no explanations, no markdown
6. The code must be valid Python that can be exec()'ed directly
"""


class Mutator:
    """LLM 驱动的代码变异算子."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def mutate(self, parent_code: str, prompt: str = None) -> str:
        """对父代代码进行变异.

        Args:
            parent_code: 父代人格生成器代码 φ
            prompt: 变异指令（None 则随机从库中抽取）

        Returns:
            str: 子代代码 φ'
        """
        if prompt is None:
            prompt = random.choice(MUTATION_PROMPTS)

        mutation_prompt = f"""【变异指令】
{prompt}

【父代代码】
```python
{parent_code}
```

【任务】
请根据变异指令修改上述代码。
只输出完整的修改后 Python 代码，不要任何解释或 markdown 标记。
确保代码可以直接被 exec() 执行。
"""

        resp = self.llm.generate(
            mutation_prompt,
            system_prompt=MUTATION_SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=2048,
        )

        # 清理输出
        code = resp.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # 基本验证
        if "class SeedPersonaGenerator" not in code:
            raise ValueError("变异后的代码缺少 SeedPersonaGenerator 类")
        if "def stage1" not in code or "def stage2" not in code:
            raise ValueError("变异后的代码缺少 stage1 或 stage2 方法")

        return code

    def add_prompt(self, prompt: str):
        """向提示词库添加新变异指令."""
        MUTATION_PROMPTS.append(prompt)

    @staticmethod
    def get_prompts() -> List[str]:
        """获取所有变异指令."""
        return MUTATION_PROMPTS.copy()
