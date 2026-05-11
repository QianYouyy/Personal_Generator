"""变异算子 - 使用 LLM 改写代码."""

import random
from typing import List


MUTATION_PROMPTS = [
    "修改 Stage 1，要求覆盖极端坐标（如每个轴的 0.1 和 0.9 分位点）",
    "将 Stage 2 改为生成核心信念和价值观，而非童年形成性记忆",
    "添加第一人称叙述格式（'我倾向于...' 而非第三人称）",
    "在 Stage 1 中加入反事实约束：'如果此人格处于对立面，会如何'",
    "修改 Stage 2，要求生成具体的行为模式而非抽象描述",
    "在 Stage 1 中使用聚类策略，确保人格分布在不同象限",
    "将 Stage 2 的输出改为结构化 JSON 格式",
    "修改 Stage 1，先采样一个'锚点人格'，其余人格围绕它对称生成",
]


class Mutator:
    """LLM 驱动的代码变异算子."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def mutate(self, parent_code: str, prompt: str = None) -> str:
        """对父代代码进行变异.

        Args:
            parent_code: 父代人格生成器代码 φ
            prompt: 变异指令（None 则随机从库中抽取）

        Returns:
            str: 子代代码 φ'
        """
        if prompt is None:
            prompt = random.choice(MUTATION_PROMPTS)

        # TODO: 构造 LLM 提示，要求只修改 Stage 1/2 内部逻辑
        # TODO: 解析返回的代码字符串
        raise NotImplementedError("Phase 3 实现中")

    def add_prompt(self, prompt: str):
        """向提示词库添加新变异指令."""
        MUTATION_PROMPTS.append(prompt)
