"""
4 份成熟心理学问卷的 few-shot 数据。

来源：
- BFI: Big Five Inventory (John et al., 1991)
- DASS: Depression Anxiety Stress Scale (Lovibond & Lovibond, 1995)
- SVO: Social Value Orientation (Murphy et al., 2011)
- NFCS: Need for Closure Scale (Webster & Kruglanski, 1994)

每份示例代码必须包含：
  - context: str（问卷主题描述）
  - dimensions: List[str]（多样性轴列表）
  - questions: List[Question]（按轴分组的题项）
  - AGREEMENT_SCALE: List[str]（5 点李克特选项）

用于作为 few-shot 示例，教 LLM "新问卷该怎么写"。
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


# ============================================
# 全局 5 点李克特量表选项
# ============================================

AGREEMENT_SCALE = [
    "Strongly Disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly Agree",
]


# ============================================
# Concordia 兼容的 Question 对象
# ============================================

@dataclass
class Question:
    """Concordia 兼容的问卷题项.

    字段与论文技术路线一致：
      - preprompt: 前置提示（如"请根据以下情境回答"）
      - statement: 题项陈述（如"我在人群中感到充满活力"）
      - choices: 选项列表（使用全局 AGREEMENT_SCALE）
      - dimension: 所属维度（英文标识符，如"openness"）
    """
    preprompt: str
    statement: str
    choices: List[str]
    dimension: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(**data)


# ============================================
# 辅助函数：生成 Concordia 兼容的 Python 代码
# ============================================

def render_questionnaire_code(
    brief: str,
    context: str,
    dimensions: List[str],
    questions: List[Question],
) -> str:
    """将问卷数据渲染为可直接 exec() 的 Python 代码字符串."""
    lines = [
        f"# {brief}",
        f'context = """{context}"""',
        "",
        f"dimensions = {dimensions!r}",
        "",
        "AGREEMENT_SCALE = [",
    ]
    for choice in AGREEMENT_SCALE:
        lines.append(f'    "{choice}",')
    lines.append("]")
    lines.append("")
    lines.append("questions = [")
    for q in questions:
        lines.append("    Question(")
        lines.append(f'        preprompt="{q.preprompt}",')
        lines.append(f'        statement="{q.statement}",')
        lines.append(f'        choices=AGREEMENT_SCALE,')
        lines.append(f'        dimension="{q.dimension}",')
        lines.append("    ),")
    lines.append("]")
    return "\n".join(lines)


# ============================================
# 示例 1: BFI (Big Five Inventory)
# ============================================

BFI_CONTEXT = """This questionnaire measures the Big Five personality traits.
Please rate how much you agree with each statement based on your typical behavior and feelings.
These traits reflect fundamental differences in how people think, feel, and behave."""

BFI_DIMENSIONS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

BFI_QUESTIONS = [
    # openness
    Question(preprompt="Rate your agreement:", statement="I have a vivid imagination.", choices=AGREEMENT_SCALE, dimension="openness"),
    Question(preprompt="Rate your agreement:", statement="I am not interested in abstract ideas.", choices=AGREEMENT_SCALE, dimension="openness"),
    Question(preprompt="Rate your agreement:", statement="I appreciate artistic and aesthetic experiences.", choices=AGREEMENT_SCALE, dimension="openness"),
    # conscientiousness
    Question(preprompt="Rate your agreement:", statement="I am always prepared and pay attention to details.", choices=AGREEMENT_SCALE, dimension="conscientiousness"),
    Question(preprompt="Rate your agreement:", statement="I tend to leave my belongings around.", choices=AGREEMENT_SCALE, dimension="conscientiousness"),
    Question(preprompt="Rate your agreement:", statement="I get chores done right away without procrastinating.", choices=AGREEMENT_SCALE, dimension="conscientiousness"),
    # extraversion
    Question(preprompt="Rate your agreement:", statement="I feel energized when I am around people.", choices=AGREEMENT_SCALE, dimension="extraversion"),
    Question(preprompt="Rate your agreement:", statement="I do not like being the center of attention.", choices=AGREEMENT_SCALE, dimension="extraversion"),
    Question(preprompt="Rate your agreement:", statement="I start conversations with others actively.", choices=AGREEMENT_SCALE, dimension="extraversion"),
    # agreeableness
    Question(preprompt="Rate your agreement:", statement="I am concerned about the feelings of others.", choices=AGREEMENT_SCALE, dimension="agreeableness"),
    Question(preprompt="Rate your agreement:", statement="I am not really interested in other people's problems.", choices=AGREEMENT_SCALE, dimension="agreeableness"),
    Question(preprompt="Rate your agreement:", statement="I am willing to help others.", choices=AGREEMENT_SCALE, dimension="agreeableness"),
    # neuroticism
    Question(preprompt="Rate your agreement:", statement="I get stressed out easily.", choices=AGREEMENT_SCALE, dimension="neuroticism"),
    Question(preprompt="Rate your agreement:", statement="I seldom feel blue or down.", choices=AGREEMENT_SCALE, dimension="neuroticism"),
    Question(preprompt="Rate your agreement:", statement="My mood changes frequently over small things.", choices=AGREEMENT_SCALE, dimension="neuroticism"),
]


# ============================================
# 示例 2: DASS (Depression Anxiety Stress Scale)
# ============================================

DASS_CONTEXT = """This questionnaire measures your emotional state over the past week, including depression, anxiety, and stress.
Please rate how much each statement applied to you during the past week.
These emotional responses reflect differences in how people react to pressure and challenges."""

DASS_DIMENSIONS = ["depression", "anxiety", "stress"]

DASS_QUESTIONS = [
    # depression
    Question(preprompt="Over the past week:", statement="I found it difficult to work up the initiative to do things.", choices=AGREEMENT_SCALE, dimension="depression"),
    Question(preprompt="Over the past week:", statement="I felt that I had nothing to look forward to.", choices=AGREEMENT_SCALE, dimension="depression"),
    Question(preprompt="Over the past week:", statement="I felt down-hearted and blue.", choices=AGREEMENT_SCALE, dimension="depression"),
    # anxiety
    Question(preprompt="Over the past week:", statement="I experienced dryness of my mouth.", choices=AGREEMENT_SCALE, dimension="anxiety"),
    Question(preprompt="Over the past week:", statement="I experienced breathing difficulty.", choices=AGREEMENT_SCALE, dimension="anxiety"),
    Question(preprompt="Over the past week:", statement="I was worried about situations in which I might panic.", choices=AGREEMENT_SCALE, dimension="anxiety"),
    # stress
    Question(preprompt="Over the past week:", statement="I found myself getting agitated.", choices=AGREEMENT_SCALE, dimension="stress"),
    Question(preprompt="Over the past week:", statement="I found it difficult to relax.", choices=AGREEMENT_SCALE, dimension="stress"),
    Question(preprompt="Over the past week:", statement="I felt that I was using a lot of nervous energy.", choices=AGREEMENT_SCALE, dimension="stress"),
]


# ============================================
# 示例 3: SVO (Social Value Orientation)
# ============================================

SVO_CONTEXT = """This questionnaire measures your social value orientation in resource allocation scenarios.
Please choose between two allocation options in each scenario.
These choices reflect your fundamental tendency toward cooperation and competition."""

SVO_DIMENSIONS = ["prosocial", "individualistic", "competitive"]

# SVO 使用情境选择题而非 Likert 陈述句，但统一为 Question 格式
SVO_CHOICES = ["Option A", "Option B"]

SVO_QUESTIONS = [
    Question(preprompt="Choose one option:", statement="You get 480, Other gets 480  vs  You get 540, Other gets 280", choices=SVO_CHOICES, dimension="prosocial"),
    Question(preprompt="Choose one option:", statement="You get 500, Other gets 500  vs  You get 500, Other gets 100", choices=SVO_CHOICES, dimension="prosocial"),
    Question(preprompt="Choose one option:", statement="You get 520, Other gets 520  vs  You get 580, Other gets 300", choices=SVO_CHOICES, dimension="individualistic"),
    Question(preprompt="Choose one option:", statement="You get 480, Other gets 200  vs  You get 540, Other gets 540", choices=SVO_CHOICES, dimension="individualistic"),
    Question(preprompt="Choose one option:", statement="You get 480, Other gets 120  vs  You get 480, Other gets 480", choices=SVO_CHOICES, dimension="competitive"),
    Question(preprompt="Choose one option:", statement="You get 500, Other gets 100  vs  You get 500, Other gets 500", choices=SVO_CHOICES, dimension="competitive"),
]


# ============================================
# 示例 4: NFCS (Need for Closure Scale)
# ============================================

NFCS_CONTEXT = """This questionnaire measures your need for cognitive closure—the preference for certainty, clear structure, and definitive answers.
Please rate how much you agree with each statement based on your typical thinking habits.
These traits reflect differences in how people handle uncertainty and ambiguous information."""

NFCS_DIMENSIONS = ["decisiveness", "need_for_structure", "discomfort_with_ambiguity", "closed_mindedness"]

NFCS_QUESTIONS = [
    # decisiveness
    Question(preprompt="Rate your agreement:", statement="I think that having clear rules and order at work is essential for success.", choices=AGREEMENT_SCALE, dimension="decisiveness"),
    Question(preprompt="Rate your agreement:", statement="I do not like situations that are uncertain.", choices=AGREEMENT_SCALE, dimension="decisiveness"),
    Question(preprompt="Rate your agreement:", statement="I dislike questions which could be answered in many different ways.", choices=AGREEMENT_SCALE, dimension="decisiveness"),
    # need_for_structure
    Question(preprompt="Rate your agreement:", statement="I like to have friends who are unpredictable.", choices=AGREEMENT_SCALE, dimension="need_for_structure"),
    Question(preprompt="Rate your agreement:", statement="I find that a well-ordered life with regular hours suits my temperament.", choices=AGREEMENT_SCALE, dimension="need_for_structure"),
    Question(preprompt="Rate your agreement:", statement="I enjoy having a clear and structured mode of life.", choices=AGREEMENT_SCALE, dimension="need_for_structure"),
    # discomfort_with_ambiguity
    Question(preprompt="Rate your agreement:", statement="When dining out, I like to go to places where I have been before so that I know what to expect.", choices=AGREEMENT_SCALE, dimension="discomfort_with_ambiguity"),
    Question(preprompt="Rate your agreement:", statement="I feel uncomfortable when I don't understand the reason why an event occurred in my life.", choices=AGREEMENT_SCALE, dimension="discomfort_with_ambiguity"),
    Question(preprompt="Rate your agreement:", statement="I feel irritated when one person disagrees with what everyone else in a group believes.", choices=AGREEMENT_SCALE, dimension="discomfort_with_ambiguity"),
    # closed_mindedness
    Question(preprompt="Rate your agreement:", statement="I hate to change my plans at the last minute.", choices=AGREEMENT_SCALE, dimension="closed_mindedness"),
    Question(preprompt="Rate your agreement:", statement="I don't like to go into a situation without knowing what I can expect from it.", choices=AGREEMENT_SCALE, dimension="closed_mindedness"),
    Question(preprompt="Rate your agreement:", statement="When I have made a decision, I feel relieved.", choices=AGREEMENT_SCALE, dimension="closed_mindedness"),
]


# ============================================
# 组装所有 few-shot 示例
# ============================================

FEWSHOT_QUESTIONNAIRES = [
    {
        "brief": "BFI (Big Five Inventory) - 大五人格量表",
        "context": BFI_CONTEXT,
        "dimensions": BFI_DIMENSIONS,
        "questions": BFI_QUESTIONS,
    },
    {
        "brief": "DASS (Depression Anxiety Stress Scale) - 抑郁焦虑压力量表",
        "context": DASS_CONTEXT,
        "dimensions": DASS_DIMENSIONS,
        "questions": DASS_QUESTIONS,
    },
    {
        "brief": "SVO (Social Value Orientation) - 社会价值取向量表",
        "context": SVO_CONTEXT,
        "dimensions": SVO_DIMENSIONS,
        "questions": SVO_QUESTIONS,
    },
    {
        "brief": "NFCS (Need for Closure Scale) - 认知闭合需求量表",
        "context": NFCS_CONTEXT,
        "dimensions": NFCS_DIMENSIONS,
        "questions": NFCS_QUESTIONS,
    },
]


def render_all_fewshots() -> str:
    """渲染所有 few-shot 示例为 Prompt 字符串（可直接 exec 的 Python 代码）."""
    parts = []
    for i, q in enumerate(FEWSHOT_QUESTIONNAIRES, 1):
        code = render_questionnaire_code(
            brief=q["brief"],
            context=q["context"],
            dimensions=q["dimensions"],
            questions=q["questions"],
        )
        parts.append(f"【示例 {i}: {q['brief']}】\n{code}\n")
    return "\n".join(parts)
