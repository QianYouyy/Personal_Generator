"""变异算子 - 使用 LLM 改写人格生成器代码.

基于 AlphaEvolve 论文 Appendix B 的 Prompt 4 (System Prompt) 和 Prompt 5 (Evolution Prompts)。
"""

import random
from typing import List


# =============================================================================
# AlphaEvolve System Prompt (Prompt 4) - 全局系统提示
# =============================================================================
ALPHAEVOLVE_SYSTEM_PROMPT = """# Task context for AlphaEvolve

Act as an expert in computational social science, agent-based modeling, and generative AI. Your task is to iteratively improve the provided codebase, which uses LLMs to generate agent contexts for social simulations based on the Concordia framework. The primary goal is to modify the generation process to maximize the behavioral diversity of the resulting agents based on specified diversity axes (e.g., personality traits, backgrounds, motivations). The evaluation metrics reward sets of agent contexts that cover the extremes and nuances of the requested diversity axes, ensuring the resulting agents exhibit a wide range of behaviors in a simulation. Agent diversity will be evaluated using questionnaires probing their likely thoughts, preferences, and behaviors in various situations related to the diversity axes.

Always adhere to best practices in Python.

## Agent Diversity and Appropriateness Theory

In this task, our goal is to generate contexts for diverse Concordia agents, enabling them to exhibit a wide range of behaviors along specified diversity axes. Concordia is a framework for building generative agents who behave according to a 'Logic of Appropriateness'. Agents decide how to act by asking three core questions:

1. What kind of situation is [agent name] in right now?
2. What kind of person is [agent name]?
3. What would a person like [agent name] do in a situation like this?

The code you are editing generates the context — collections of memories, beliefs, personality traits, core values, goals, or even how others perceive the character — that helps an agent answer question #2, and thereby question #3: what action is appropriate for *their specific identity* in a given context. This context is not limited to formative memories; it can include any information that shapes identity and decision-making. Any detail that helps condition the agent's behavior in line with the three Concordia questions is valid. The objective is to generate rich and diverse contexts that enable an LLM to convincingly role-play as a specific person in a social setting.

LLM-generated behavior often clusters around a narrow distribution of stereotypical responses. We want to explicitly counteract this by generating agent contexts that cover the full spectrum of human experience along the specified axes. Crucially, different agents should react differently to the *same* situation, and the same agent might react differently to *different* situations, based on their unique identity, values, and memories. Your modifications should encourage the generation of agent contexts that occupy unique positions in the diversity space, including extremes or unusual combinations of traits, pushing towards maximal coverage of the possible behavioral landscape and genuinely diverse downstream behavior. No two generated agent contexts should ever be the same.

The provided codebase uses a two-stage process. Stage 1 is crucial for diversity: it autoregressively generates an intermediate representation for each agent, establishing their core traits along the specified diversity axes for the entire population. Stage 2 then takes these intermediate representations and develops each agent in parallel, generating a set of memories/contexts (e.g., individual backgrounds, formative experiences, core beliefs and more) to create fully-fledged characters suitable for simulation as Concordia agents.

## Mutation Rules

When modifying the code:
1. You MUST preserve the class name "SeedPersonaGenerator"
2. You MUST preserve the method signatures: __init__, stage1, stage2
3. You MUST preserve the two-stage structure (stage1 → stage2)
4. You should ONLY modify the internal logic, prompts, and strategies inside stage1 and stage2
5. Output ONLY the complete modified Python code, no explanations, no markdown
6. The code must be valid Python that can be exec()'ed directly
"""


# =============================================================================
# AlphaEvolve Evolution Prompts (Prompt 5) - 25条变异提示
# =============================================================================

# Stage 1 变异提示（11条）
STAGE1_MUTATION_PROMPTS = [
    # 1
    "Modify Stage 1 to explicitly request agent contexts that represent the **extreme ends** of the diversity axes, as well as points in between.",
    
    # 2
    "Add an explicit instruction to the Stage 1 prompt to make each generated agent context **as different as possible** from the others across all specified diversity axes.",
    
    # 3
    "Modify Stage 1 to request agent contexts that feature **internal contradictions** or cognitive dissonances (e.g., an optimist with a tragic past).",
    
    # 4
    "Reimplement Stage 1 to use **staggered generation**: ask the LLM to generate agent contexts in sequential batches, with each batch targeting a specific range (e.g., high/low) of one or more diversity axes to ensure full coverage.",
    
    # 5
    "Modify Stage 1 to generate agent contexts **iteratively** rather than all at once. In each iteration, prompt for a small number of agent contexts that occupy a **specific niche** of the diversity space (e.g., 'generate 2 agents who are highly introverted and optimistic').",
    
    # 6
    "Modify Stage 1 to explicitly instruct the LLM to sample agent contexts such that they cover **as many combinations of axis positions as possible** (e.g., if axes are A and B, ensure agent types A-high/B-low, A-low/B-high, etc., are represented).",
    
    # 7
    "Modify the prompt in Stage 1 that asks the LLM to explain diversity axes, to also provide **examples of characters at the extreme ends** of each axis.",
    
    # 8
    "Modify Stage 1 to add a **new field** to the agent context JSON output that adds depth and potential for unique behavior.",
    
    # 9
    "Modify Stage 1 to add a **situational triggers** field to the JSON output, listing 1–2 types of situations that this agent is particularly sensitive to.",
    
    # 10
    "Change Stage 1 to **segment** agent context generation. Instead of one call for `num_personas`, make multiple calls to generate subsets of agent contexts, each call asking for agents with specific characteristics (e.g., focusing on axis extremes or combinations) to ensure all niches are covered.",
    
    # 11
    "Modify Stage 1 to include in each agent's description **specific opinions or preferences** related to the diversity axes, to ensure they are measurable by the evaluation.",
]

# Stage 2 变异提示（10条）
STAGE2_MUTATION_PROMPTS = [
    # 12
    "Modify Stage 2 to generate formative memories that explain why an agent might react **strongly or unexpectedly** to certain situations, anchoring their traits in specific experiences.",
    
    # 13
    "Change Stage 2 entirely: Instead of generating memories, modify it to generate **3 core beliefs or values** that are most important to this agent.",
    
    # 14
    "Change Stage 2 entirely: Instead of generating memories, modify it to generate a **paragraph explaining how this agent interprets situations and decides on appropriate actions**, referencing their identity.",
    
    # 15
    "Modify Stage 2 to replace or augment memory generation with **1–2 examples** of how this agent would react to a specific hypothetical social situation relevant to the initial context.",
    
    # 16
    "Modify Stage 2 to focus on generating an agent's **core fears and future aspirations** instead of, or in addition to, past memories.",
    
    # 17
    "Change Stage 2 entirely: Instead of generating memories, modify it to generate a **'heuristic'** or cognitive shortcut this agent uses when making quick decisions under pressure.",
    
    # 18
    "Modify Stage 2 to ensure that at least one generated memory involves a **significant failure, trauma, or regret** that shaped the agent.",
    
    # 19
    "Change Stage 2 entirely: Instead of generating memories, make it generate a list of **5 behavioral 'Do's' and 'Don'ts'** that characterize this agent in social situations.",
    
    # 20
    "Change Stage 2: Instead of memories, generate **2–3 examples of specific 'appropriateness rules'** the agent follows (e.g., 'When criticized, I become defensive' or 'In a formal setting, I remain silent').",
    
    # 21
    "Modify Stage 2 to generate a paragraph describing the agent's typical **'inner monologue'** or thought process when faced with ambiguity or social stress.",
]

# 全局/元策略提示（4条）
META_MUTATION_PROMPTS = [
    # 22
    "Suggest a crazy idea of how we can improve our implementation.",
    
    # 23
    "Suggest a new idea to improve the code.",
    
    # 24
    "Suggest a crazy idea of how we can improve our implementation, something that **definitely nobody else would think of**. Make it crazy with a capital C.",
    
    # 25
    "Propose modifications to the current program that **combine the strengths of all the programs above** and achieve high scores on the task.",
]

# 合并所有提示
ALL_MUTATION_PROMPTS = (
    STAGE1_MUTATION_PROMPTS + 
    STAGE2_MUTATION_PROMPTS + 
    META_MUTATION_PROMPTS
)


class Mutator:
    """LLM 驱动的代码变异算子 — 基于 AlphaEvolve 论文.
    
    使用方式：
      1. System Prompt: ALPHAEVOLVE_SYSTEM_PROMPT (全局注入)
      2. User Prompt: 随机从 25 条提示中抽取 1 条 + 父代代码
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def mutate(self, parent_code: str, prompt: str = None) -> str:
        """对父代代码进行变异.

        Args:
            parent_code: 父代人格生成器代码 φ
            prompt: 变异指令（None 则随机从 25 条中抽取）

        Returns:
            str: 子代代码 φ'
        """
        if prompt is None:
            prompt = random.choice(ALL_MUTATION_PROMPTS)

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
            system_prompt=ALPHAEVOLVE_SYSTEM_PROMPT,
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
        ALL_MUTATION_PROMPTS.append(prompt)

    @staticmethod
    def get_prompts() -> List[str]:
        """获取所有变异指令."""
        return ALL_MUTATION_PROMPTS.copy()
    
    @staticmethod
    def get_prompts_by_category() -> dict:
        """按类别获取变异指令."""
        return {
            "stage1": STAGE1_MUTATION_PROMPTS.copy(),
            "stage2": STAGE2_MUTATION_PROMPTS.copy(),
            "meta": META_MUTATION_PROMPTS.copy(),
        }
