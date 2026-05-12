"""问卷生成器的 Prompt 模板."""

from src.qgenerator.fewshot_data import render_all_fewshots, AGREEMENT_SCALE

# ============================================
# Stage 1 少样本示例
# 从 4 份成熟问卷中提取真实数据，渲染为可直接 exec() 的 Python 代码
# ============================================

FEW_SHOT_EXAMPLES = render_all_fewshots()

STAGE1_SYSTEM_PROMPT = """You are an expert psychometrician specializing in personality psychology, social psychology, and psychological measurement.
Your task is to design a new psychological questionnaire based on a brief scenario description, following the exact structure and style of the 4 example questionnaires provided.

Design principles:
1. Diversity dimensions must be deep psychological dimensions that lead to behavioral differences IN THE GIVEN SCENARIO
2. Dimensions should be orthogonal (low correlation) to each other
3. Each dimension should be a continuous spectrum, not a category
4. Dimensions must emerge from the specific scenario, not generic personality traits
5. Target dimension count is strictly 2 or 3
6. Output format must be JSON, with dimension names as English snake_case identifiers (e.g., "agi_threat_appraisal")
"""


def stage1_prompt(brief_context: str, k_dimensions: int) -> str:
    """Stage 1: 扩展上下文 + 提议多样性轴（输出 JSON）."""
    return f"""Here are 4 established psychological questionnaires rendered as Concordia-compatible Python code:

{FEW_SHOT_EXAMPLES}
---

【Task】
Based on the brief scenario description below, generate a new questionnaire following the EXACT structure and style of the 4 examples above.

Requirements:
1. Expand the scenario: Write a detailed context (200-500 words) describing the situation, roles, stakes, and conflicts
2. Propose diversity dimensions: Design **exactly {k_dimensions}** dimensions
   - Use English snake_case identifiers (e.g., "risk_tolerance", "social_compliance")
   - Dimensions must emerge from the scenario, not generic traits like "openness" or "neuroticism"
3. Output must be valid JSON

【Brief Description】
{brief_context}

【Output Format (JSON)】
{{
  "context": "Detailed scenario description...",
  "dimensions": ["dim_1", "dim_2"]
}}
"""


STAGE2_SYSTEM_PROMPT = f"""You are an expert psychometrician. Your task is to generate Likert-scale questionnaire items based on a given scenario and diversity dimensions.

Design principles:
1. Each item must measure ONLY ONE dimension
2. Items must be SPECIFIC to the scenario, containing situational details
3. Items should be observable behaviors or concrete thoughts, not abstract traits
4. Use first-person "I" or "For me"
5. All items must use the SAME 5-point Likert scale:
   {AGREEMENT_SCALE}
6. Output must be valid Python code that can be executed directly
"""


def stage2_prompt(context: str, dimension: str, num_items: int = 5) -> str:
    """Stage 2: 为单个轴生成题项（输出可直接 exec 的 Python 代码）."""
    return f"""【Scenario Context】
{context}

【Dimension to Measure】
{dimension}

【Task】
Generate {num_items} Likert-scale questionnaire items for the dimension "{dimension}".

Requirements:
- Each item must be specific to the scenario above (contain situational details)
- All items use the same 5-point Likert scale: {AGREEMENT_SCALE}
- Use first-person statements ("I..." or "For me...")
- Output must be valid Python code that can be executed directly with exec()
- The code must define a variable named `questions` as a list of Question objects

【Output Format (Python Code)】
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Question:
    preprompt: str
    statement: str
    choices: List[str]
    dimension: str

AGREEMENT_SCALE = {AGREEMENT_SCALE!r}

questions = [
    Question(
        preprompt="Rate your agreement:",
        statement="Specific item text related to the scenario...",
        choices=AGREEMENT_SCALE,
        dimension="{dimension}",
    ),
    # ... more items
]
```

Please output ONLY the Python code block, nothing else.
"""
