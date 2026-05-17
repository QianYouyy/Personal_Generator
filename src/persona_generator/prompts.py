"""人格生成器的 Prompt 模板."""

from typing import List


# ============================================
# Stage 1: 高层定位标签生成
# ============================================

STAGE1_SYSTEM_PROMPT = """You are an expert persona designer for social simulation research.
Your task is to generate diverse synthetic personas that represent different positions along given diversity dimensions.

Key principles:
1. Each persona must have a CLEAR and SPECIFIC position on each dimension
2. Personas should be DIVERSE - avoid clustering in the middle of dimensions
3. Personas should feel REAL and COHERENT - their traits should make sense together
4. Use natural language descriptions, not just coordinate values
"""


def stage1_seed1_prompt(context: str, dimensions: List[str], n: int, generated_so_far: List[str] = None) -> str:
    """seed1 Stage 1: 默认 Concordia 风格，串行生成.

    Args:
        context: 问卷情境描述
        dimensions: 多样性轴列表
        n: 需要生成的人格总数
        generated_so_far: 已生成的高层标签列表（用于错开）
    """
    existing = ""
    if generated_so_far:
        existing = "\n【已生成的人格定位】\n" + "\n".join(
            f"  {i+1}. {tag}" for i, tag in enumerate(generated_so_far)
        ) + "\n\n请确保新的人格与上述已生成的人格在维度空间中保持距离，避免重叠。"

    dims_str = "\n".join(f"  - {dim}" for dim in dimensions)

    return f"""【情境】
{context}

【多样性轴】
{dims_str}

【任务】
请生成 1 个新的人格高层定位标签（high-level persona tag）。

要求：
1. 明确指定该人格在每个维度上的位置（如"高适应力、中等风险承受"）
2. 使用具体、生动的语言描述，让人格有辨识度
3. 整体控制在 2-3 句话
4. 必须与已生成的人格错开，确保多样性空间覆盖{existing}

【输出格式】
直接输出人格定位标签文本，不要加任何前缀或解释。
"""


def stage1_seed2_prompt(context: str, dimensions: List[str], batch_size: int, batch_idx: int, total_batches: int) -> str:
    """seed2 Stage 1: 小批次自回归，每批生成 batch_size 个.

    减少上下文依赖：每批独立生成，不携带之前所有的历史。
    """
    dims_str = "\n".join(f"  - {dim}" for dim in dimensions)

    return f"""【情境】
{context}

【多样性轴】
{dims_str}

【任务】
这是第 {batch_idx + 1}/{total_batches} 批生成。
请一次性生成 {batch_size} 个不同的人格高层定位标签。

要求：
1. 每个标签明确指定在各维度上的位置
2. {batch_size} 个标签之间必须相互错开，覆盖不同区域
3. 使用具体、生动的语言
4. 每标签 2-3 句话

【输出格式】
用编号列表输出，每行一个标签：
1. [标签内容]
2. [标签内容]
...
"""


def stage1_seed3_coordinate_to_text(context: str, dimensions: List[str], coordinates: List[float]) -> str:
    """seed3 Stage 1: 将蒙特卡洛采样坐标翻译为文字描述.

    Args:
        coordinates: K 维坐标列表，每个值在 [0, 1] 区间
    """
    coord_str = "\n".join(
        f"  - {dim}: {coord:.2f} (0=极低, 1=极高)"
        for dim, coord in zip(dimensions, coordinates)
    )

    return f"""【情境】
{context}

【多样性轴坐标】
{coord_str}

【任务】
请将上述坐标翻译成一段自然语言的人格高层定位标签。
描述一个在该情境下、具有这些维度坐标特征的完整人格轮廓。
要求：
1. 用 2-3 句话描述这个人格的核心特征
2. 语言要生动、有辨识度
3. 直接输出标签文本，不加前缀
"""


# ============================================
# Stage 2: 细节扩展（形成性记忆）
# ============================================

STAGE2_SYSTEM_PROMPT = """You are an expert character writer and developmental psychologist.
Your task is to expand a brief persona tag into a rich, detailed character description.

The description should follow a formative memory approach:
1. Begin with childhood experiences that shaped the person's core traits
2. Describe key life events and how they influenced beliefs and behaviors
3. Explain the person's decision-making logic and worldview
4. Include speaking style, habits, and social patterns
5. Keep it grounded in the specific scenario context

Write in third person ("He/She/They"). Target length: 200-400 words.
"""


def stage2_prompt(context: str, dimensions: List[str], high_level_tag: str) -> str:
    """Stage 2: 将高层标签扩展为完整人格描述."""
    dims_str = ", ".join(dimensions)

    return f"""【情境】
{context}

【多样性轴】
{dims_str}

【高层定位标签】
{high_level_tag}

【任务】
请将上述高层定位标签扩展为一段完整、丰富的人格描述。

要求：
1. 从童年形成性记忆写起（是什么经历塑造了这个人的核心特质？）
2. 描述核心信念和价值观（这个人相信什么？什么对他/她最重要？）
3. 描述行为逻辑（面对决策时，这个人会怎么思考？）
4. 描述说话风格和社交模式（这个人怎么与人交流？）
5. 必须与上述情境和维度定位一致
6. 用第三人称"他/她"撰写
7. 长度 200-400 词

【输出格式】
直接输出人格描述文本，不要加标题或分段标记。
"""
