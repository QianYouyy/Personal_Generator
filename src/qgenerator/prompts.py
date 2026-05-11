"""问卷生成器的 Prompt 模板."""

# 少样本示例，展示高质量问卷的多样性轴和题项风格
FEW_SHOT_EXAMPLES = """
【示例 1: BFI (大五人格量表)】
维度: 开放性(Openness), 尽责性(Conscientiousness), 外向性(Extraversion), 宜人性(Agreeableness), 神经质(Neuroticism)
题项风格: "我喜欢抽象思考" / "我做事有条理" / "我在人群中感到 energized" / "我关心他人的感受" / "我容易感到焦虑"
每个维度 8-10 题，Likert 5 点量表

【示例 2: DASS (抑郁焦虑压力量表)】
维度: 抑郁(Depression), 焦虑(Anxiety), 压力(Stress)
题项风格: "我觉得做任何事都很费劲" / "我感到口干舌燥" / "我很难放松下来"
每个维度 7 题，Likert 4 点量表（本系统使用 5 点）

【示例 3: SVO (社会价值取向量表)】
维度: 亲社会(Prosocial), 个体主义(Individualistic), 竞争(Competitive)
题项风格: 通过在资源分配情境中的选择来测量，如"你得到 X，对方得到 Y"

【示例 4: NFCS (认知闭合需求量表)】
维度: 决断性(Decisiveness), 结构需求(Need for Structure), 对模糊的不适(Discomfort with Ambiguity)
题项风格: "我不喜欢问题没有明确答案的情况" / "我更喜欢有秩序的、按部就班的生活"
"""

STAGE1_SYSTEM_PROMPT = """你是一位资深心理学量表设计专家，精通人格心理学、社会心理学和心理测量学。
你的任务是根据用户提供的简短情境描述，设计一个能够捕捉该情境下人格差异多样性轴的心理学问卷。

设计原则：
1. 多样性轴必须是**在该情境下能够导致行为差异**的深层心理维度
2. 轴与轴之间应尽量**正交**（低相关）
3. 每个轴应该是**连续谱**（而非分类），两端代表截然不同的倾向
4. 避免过于宽泛的人格维度，要贴合具体情境
"""


def stage1_prompt(brief_context: str, k_dimensions: int) -> str:
    """Stage 1: 扩展上下文 + 提议多样性轴."""
    return f"""{FEW_SHOT_EXAMPLES}

---

【任务】
基于以下简短情境描述，完成两个步骤：

步骤 1 - 扩展情境:
将简短描述扩展为 2-3 段的详细情境说明（c）。增加具体的背景信息、时间地点、涉及的社会角色、潜在冲突等，使情境更加丰满和具体。

步骤 2 - 设计多样性轴:
为该情境设计 **{k_dimensions} 个**多样性轴（D）。
要求：
- 每个轴用 1-2 个中文字命名，并附英文翻译
- 为每个轴写一段说明（50-100 字），解释它在此情境下如何导致不同行为
- 明确标注轴的两端（如 "乐观 ↔ 悲观"）
- 轴的数量严格为 {k_dimensions} 个

【简短描述】
{brief_context}

【输出格式】
请用以下 JSON 格式输出（不要包含 markdown 代码块标记）：
{{
  "context": "扩展后的详细情境...",
  "dimensions": [
    {{
      "name_cn": "轴中文名",
      "name_en": "AxisName",
      "description": "轴的说明...",
      "poles": ["左端标签", "右端标签"]
    }}
  ]
}}
"""


STAGE2_SYSTEM_PROMPT = """你是一位资深心理学量表设计专家。你的任务是根据给定的情境和多样性轴，设计 Likert 5 点量表题项。

设计原则：
1. 每个题项必须**只测量一个轴**，不能交叉
2. 题项要**具体、可观察、避免双重否定**
3. 题项要贴合情境，不能是通用人格题
4. 每个轴的题项要覆盖轴的两端（正向和反向计分）
5. 使用第一人称"我"或"对我来说"
"""


def stage2_prompt(context: str, dimension: dict, num_items: int = 5) -> str:
    """Stage 2: 为单个轴生成题项."""
    dim_name = dimension["name_cn"]
    poles = dimension.get("poles", ["低", "高"])
    return f"""【情境】
{context}

【多样性轴】
轴名: {dim_name}
说明: {dimension.get('description', '')}
两端: {poles[0]} ↔ {poles[1]}

【任务】
为该轴设计 {num_items} 个 Likert 5 点量表题项。
要求：
- 其中约一半为正向计分（高分对应 {poles[1]}）
- 约一半为反向计分（高分对应 {poles[0]}）
- 每题标注是正向(P)还是反向(N)
- 题项要具体、贴合情境

【输出格式】
请用以下 JSON 格式输出（不要包含 markdown 代码块标记）：
{{
  "items": [
    {{
      "text": "题项内容",
      "scoring": "P"
    }}
  ]
}}
"""
