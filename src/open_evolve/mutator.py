"""变异算子 - 使用 LLM 改写人格生成器代码.

基于 AlphaEvolve 论文 Appendix B 的 Prompt 4 (System Prompt) 和 Prompt 5 (Evolution Prompts)。
"""

import random
from typing import List

from src.utils.logger import logger


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
2. You MUST preserve the method signatures: __init__(self, llm_client), stage1(self, context, dimensions, n) -> list, stage2(self, context, dimensions, high_level_tags) -> list
3. You MUST preserve the two-stage structure (stage1 → stage2)
4. You should ONLY modify the internal logic, prompts, and strategies inside stage1 and stage2
5. Output ONLY the complete modified Python code, no explanations, no markdown
6. The code must be valid Python that can be exec()'ed directly

## HARD CONSTRAINTS - Violations will cause immediate rejection

Your modified code will be executed automatically by an evaluation pipeline. If it crashes or produces invalid output, the candidate is discarded. The following constraints are NON-NEGOTIABLE:

### Return Type Contracts
- **stage1 MUST return a Python `list` of strings.** Never return `None`, never return an empty list `[]`, never return a single string. The list length must be exactly `n` (the input parameter). Each element must be a non-empty, non-None string.
- **stage2 MUST return a Python `list` of strings.** Never return `None`, never return an empty list `[]`. The list length must be exactly `len(high_level_tags)`. Each element must be a non-empty, non-None string.

### Concurrency
- 使用 `concurrent.futures.ThreadPoolExecutor` 进行并行 LLM 调用，不要使用 asyncio。
- ThreadPoolExecutor 示例：
  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  
  results = [None] * n
  with ThreadPoolExecutor(max_workers=5) as executor:
      futures = {executor.submit(func, arg, i): i for i, arg in enumerate(args)}
      for future in as_completed(futures):
          i = futures[future]
          try:
              results[i] = future.result()
          except Exception:
              results[i] = "Error"
  ```
- 如果使用 `return_exceptions=True` 风格的 gather，必须过滤掉所有 Exception 对象和 None 值，用有效字符串替换。

### Error Handling
- Every LLM API call MUST have a fallback. If the API call fails or returns unexpected output, the function must still return a valid list of the correct length.
- NEVER let an unhandled exception propagate out of stage1 or stage2. Use try/except at the top level of both methods if necessary.

### Code Structure
- Do NOT add top-level code (code outside the class) that executes on import.
- Do NOT add external dependencies that require pip install.
- Do NOT modify the __init__ signature or remove the self.llm attribute.
- The code must be completely self-contained within the SeedPersonaGenerator class.
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
    
    支持状态持久化：get_state() / set_state() 用于 checkpoint 保存/恢复。
    """

    def __init__(
        self,
        llm_client,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        adaptive_temperature: bool = True,
        temp_range: tuple = (0.7, 1.2),
    ):
        self.llm = llm_client
        self.base_temperature = temperature
        self.max_tokens = max_tokens
        self.adaptive_temperature = adaptive_temperature
        self.temp_range = temp_range
        # 实例级别的提示词库（支持持久化）
        self._prompts = ALL_MUTATION_PROMPTS.copy()

    def _get_temperature(self, generation: int = 0, stagnation: int = 0) -> float:
        """动态计算变异温度.
        
        策略：
        - 前期（gen 1-3）：高温探索（1.0-1.2）
        - 中期（gen 4-7）：中温平衡（0.8-1.0）
        - 后期（gen 8+）：低温精细（0.7-0.8）
        - 停滞触发时：临时升温突破（+0.2）
        
        Args:
            generation: 当前轮数
            stagnation: 停滞轮数（该岛屿无改进的轮数）
            
        Returns:
            float: 实际使用的 temperature
        """
        if not self.adaptive_temperature:
            return self.base_temperature
        
        # 基础温度按阶段递减
        if generation <= 3:
            base = 1.1
        elif generation <= 7:
            base = 0.9
        else:
            base = 0.75
        
        # 停滞时升温突破
        if stagnation >= 3:
            base += 0.2
            logger.info(f"[变异] 检测到停滞 {stagnation} 轮，升温至 {base:.2f} 尝试突破")
        
        # 限制在范围内
        return max(self.temp_range[0], min(self.temp_range[1], base))

    def mutate(self, parent_code: str, prompt: str = None) -> str:
        """对父代代码进行变异.

        Args:
            parent_code: 父代人格生成器代码 φ
            prompt: 变异指令（None 则随机从实例提示词库中抽取）

        Returns:
            str: 子代代码 φ'
        """
        if prompt is None:
            prompt = random.choice(self._prompts)

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

    def mutate(self, parent_code: str, prompt: str = None, generation: int = 0, stagnation: int = 0) -> str:
        """对父代代码进行变异.

        Args:
            parent_code: 父代人格生成器代码 φ
            prompt: 变异指令（None 则随机从实例提示词库中抽取）
            generation: 当前轮数（用于动态温度）
            stagnation: 停滞轮数（用于动态温度）

        Returns:
            str: 子代代码 φ'
        """
        if prompt is None:
            prompt = random.choice(self._prompts)

        # 计算动态温度
        temperature = self._get_temperature(generation, stagnation)

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
            temperature=temperature,
            max_tokens=self.max_tokens,
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
        
        # 语法验证：尝试编译代码
        try:
            compile(code, "<mutated>", "exec")
        except SyntaxError as e:
            raise ValueError(f"变异后的代码存在语法错误: {e}")

        return code

    def add_prompt(self, prompt: str):
        """向提示词库添加新变异指令."""
        self._prompts.append(prompt)

    def get_prompts(self) -> List[str]:
        """获取所有变异指令."""
        return self._prompts.copy()
    
    def get_prompts_by_category(self) -> dict:
        """按类别获取变异指令（基于当前实例的提示词库）."""
        # 返回当前实例的所有提示，不保证分类准确（动态添加的提示无法分类）
        n_stage1 = len(STAGE1_MUTATION_PROMPTS)
        n_stage2 = len(STAGE2_MUTATION_PROMPTS)
        n_meta = len(META_MUTATION_PROMPTS)
        all_base = n_stage1 + n_stage2 + n_meta
        
        current = self._prompts
        return {
            "stage1": current[:n_stage1],
            "stage2": current[n_stage1:n_stage1 + n_stage2],
            "meta": current[n_stage1 + n_stage2:all_base],
            "dynamic": current[all_base:] if len(current) > all_base else [],
        }
    
    def get_state(self) -> dict:
        """获取变异算子状态（用于 checkpoint 保存）."""
        return {
            "base_temperature": self.base_temperature,
            "max_tokens": self.max_tokens,
            "adaptive_temperature": self.adaptive_temperature,
            "temp_range": self.temp_range,
            "prompts": self._prompts.copy(),
            "num_prompts": len(self._prompts),
        }
    
    def set_state(self, state: dict):
        """恢复变异算子状态（用于 checkpoint 恢复）."""
        self.base_temperature = state.get("base_temperature", 0.8)
        self.max_tokens = state.get("max_tokens", 4096)
        self.adaptive_temperature = state.get("adaptive_temperature", True)
        self.temp_range = tuple(state.get("temp_range", (0.7, 1.2)))
        if "prompts" in state:
            self._prompts = state["prompts"].copy()
        logger.info(f"变异算子状态已恢复: base_temp={self.base_temperature}, adaptive={self.adaptive_temperature}, prompts={len(self._prompts)}")
