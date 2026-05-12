"""问卷生成器 (QGenerator)."""

import ast
import json
import random
import re
from typing import List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from src.qgenerator import prompts
from src.qgenerator.fewshot_data import Question
from src.utils.config import get_config


@dataclass
class Questionnaire:
    """问卷数据结构（与论文技术路线一致）.

    Fields:
      - context: str      # c: 详细上下文
      - dimensions: List[str]  # D: 多样性轴，K ∈ {2, 3}
      - items: List[Question]  # I: 题项列表（扁平列表，每个 Question 含 dimension 字段）
      - brief: str        # 原始简短描述
    """

    context: str
    dimensions: List[str]
    items: List[Question]
    brief: str = ""

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "context": self.context,
            "dimensions": self.dimensions,
            "items": [q.to_dict() for q in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Questionnaire":
        return cls(
            brief=data.get("brief", ""),
            context=data["context"],
            dimensions=data["dimensions"],
            items=[Question.from_dict(q) for q in data["items"]],
        )

    def to_concordia_questions(self) -> List[dict]:
        """转换为 Concordia 兼容的 Question 对象列表."""
        return [q.to_dict() for q in self.items]

    def items_by_dimension(self) -> dict:
        """按维度名分组返回题项（便于查看）."""
        result = {}
        for dim in self.dimensions:
            result[dim] = [q for q in self.items if q.dimension == dim]
        return result


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    text = text.strip()
    return json.loads(text)


def _extract_python_code(text: str) -> str:
    """从 LLM 输出中提取 Python 代码块."""
    text = text.strip()
    # 尝试提取 ```python ... ``` 代码块
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 如果没有标记，尝试直接返回
    return text


def _exec_questions_code(code: str) -> List[Question]:
    """执行 Python 代码，提取 questions 列表.

    安全说明：此函数使用 exec() 执行 LLM 生成的代码。
    在生产环境中应考虑使用受限执行环境（如 sandbox）。
    """
    # 在受限命名空间中执行
    namespace = {}
    exec(code, namespace)

    if "questions" not in namespace:
        raise ValueError("Python 代码未定义 'questions' 变量")

    raw_questions = namespace["questions"]

    # 转换为 Question 对象列表
    questions = []
    for q in raw_questions:
        if isinstance(q, dict):
            questions.append(Question(**q))
        elif hasattr(q, "preprompt"):
            questions.append(Question(
                preprompt=q.preprompt,
                statement=q.statement,
                choices=list(q.choices),
                dimension=q.dimension,
            ))
        else:
            raise ValueError(f"未知的 question 类型: {type(q)}")

    return questions


class QGenerator:
    """自动生成带多样性轴的心理学问卷.

    model 和 temperature 从 configs/default.yaml 读取。
    """

    def __init__(
        self,
        llm_client,
        items_per_dimension: int = None,
        stage1_temp: float = None,
        stage2_temp: float = None,
    ):
        self.llm = llm_client

        # 从配置文件读取默认值
        cfg = get_config()
        self.items_per_dimension = items_per_dimension or cfg.get("qgenerator.items_per_dimension", 5)
        self.stage1_temp = stage1_temp or cfg.get("qgenerator.stage1_temperature", 0.8)
        self.stage2_temp = stage2_temp or cfg.get("qgenerator.stage2_temperature", 0.7)

    def generate(
        self,
        brief_context: str,
        k_dimensions: int = 2,
    ) -> Questionnaire:
        """两阶段生成问卷.

        Stage 1: 扩展 brief_context → 详细上下文 c + 多样性轴 D (JSON)
        Stage 2: 按每个轴生成 Likert 题项 I (Python 代码，exec 执行)
        """
        # ---- Stage 1 ----
        stage1_prompt = prompts.stage1_prompt(brief_context, k_dimensions)
        stage1_resp = self.llm.generate(
            stage1_prompt,
            system_prompt=prompts.STAGE1_SYSTEM_PROMPT,
            temperature=self.stage1_temp,
            max_tokens=2048,
        )
        stage1_data = _extract_json(stage1_resp)

        context = stage1_data["context"]
        dimensions = stage1_data["dimensions"]

        if len(dimensions) != k_dimensions:
            raise ValueError(
                f"期望 {k_dimensions} 个维度，实际得到 {len(dimensions)}"
            )

        # ---- Stage 2 ----
        all_questions = []
        for dim in dimensions:
            stage2_prompt = prompts.stage2_prompt(
                context, dim, self.items_per_dimension
            )
            stage2_resp = self.llm.generate(
                stage2_prompt,
                system_prompt=prompts.STAGE2_SYSTEM_PROMPT,
                temperature=self.stage2_temp,
                max_tokens=2048,
            )
            code = _extract_python_code(stage2_resp)
            questions = _exec_questions_code(code)

            # 验证所有题项的 dimension 是否正确
            for q in questions:
                if q.dimension != dim:
                    raise ValueError(
                        f"题项维度不匹配: 期望 '{dim}', 实际 '{q.dimension}'"
                    )

            all_questions.extend(questions)

        return Questionnaire(
            brief=brief_context,
            context=context,
            dimensions=dimensions,
            items=all_questions,
        )

    def batch_generate(
        self,
        brief_contexts: List[str],
        k_dimensions_list: List[int] = None,
    ) -> List[Questionnaire]:
        """批量生成问卷."""
        if k_dimensions_list is None:
            k_dimensions_list = [random.choice([2, 3]) for _ in brief_contexts]

        if len(k_dimensions_list) != len(brief_contexts):
            raise ValueError("brief_contexts 和 k_dimensions_list 长度不一致")

        questionnaires = []
        for i, (brief, k) in enumerate(zip(brief_contexts, k_dimensions_list)):
            print(f"[{i+1}/{len(brief_contexts)}] 生成问卷: {brief[:40]}...")
            try:
                q = self.generate(brief, k)
                questionnaires.append(q)
            except Exception as e:
                print(f"  ⚠️ 生成失败: {e}")
                continue

        return questionnaires

    @staticmethod
    def split(
        questionnaires: List[Questionnaire],
        train: int = None,
        val: int = None,
        test: int = None,
    ) -> Tuple[List[Questionnaire], List[Questionnaire], List[Questionnaire]]:
        """划分训练/验证/测试集."""
        cfg = get_config()
        train = train or cfg.get("qgenerator.train_split", 30)
        val = val or cfg.get("qgenerator.val_split", 10)
        test = test or cfg.get("qgenerator.test_split", 10)

        total = train + val + test
        if len(questionnaires) < total:
            raise ValueError(
                f"问卷数量不足: 需要 {total} 份，实际 {len(questionnaires)}"
            )

        random.shuffle(questionnaires)
        train_q = questionnaires[:train]
        val_q = questionnaires[train : train + val]
        test_q = questionnaires[train + val : total]
        return train_q, val_q, test_q

    @staticmethod
    def save(
        questionnaires: List[Questionnaire],
        path: str,
    ):
        """保存问卷到 JSON 文件."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [q.to_dict() for q in questionnaires]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(questionnaires)} 份问卷到 {path}")

    @staticmethod
    def load(path: str) -> List[Questionnaire]:
        """从 JSON 文件加载问卷."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Questionnaire.from_dict(d) for d in data]
