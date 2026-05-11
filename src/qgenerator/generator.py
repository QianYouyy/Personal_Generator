"""问卷生成器 (QGenerator)."""

import json
import random
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.qgenerator import prompts
from src.utils.config import get_config


@dataclass
class Questionnaire:
    """问卷数据结构."""

    context: str            # c: 详细上下文
    dimensions: List[dict]  # D: 多样性轴（含 name_cn, name_en, description, poles）
    items: dict             # I: 按轴名分组的题项列表
    brief: str = ""         # 原始简短描述（用于追溯）

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "context": self.context,
            "dimensions": self.dimensions,
            "items": self.items,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Questionnaire":
        return cls(
            brief=data.get("brief", ""),
            context=data["context"],
            dimensions=data["dimensions"],
            items=data["items"],
        )

    def to_concordia_questions(self) -> List[dict]:
        """转换为 Concordia 兼容的 Question 对象列表."""
        questions = []
        for dim in self.dimensions:
            dim_name = dim["name_cn"]
            for item in self.items.get(dim_name, []):
                questions.append({
                    "text": item["text"],
                    "dimension": dim_name,
                    "scoring": item.get("scoring", "P"),
                    "scale": 5,
                })
        return questions


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


class QGenerator:
    """自动生成带多样性轴的心理学问卷.

    model 和 temperature 从 configs/default.yaml 读取，无需硬编码。
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

        Stage 1: 扩展 brief_context → 详细上下文 c + 多样性轴 D
        Stage 2: 按每个轴生成 Likert 题项 I
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
        items_by_dimension = {}
        for dim in dimensions:
            dim_name = dim["name_cn"]
            stage2_prompt = prompts.stage2_prompt(
                context, dim, self.items_per_dimension
            )
            stage2_resp = self.llm.generate(
                stage2_prompt,
                system_prompt=prompts.STAGE2_SYSTEM_PROMPT,
                temperature=self.stage2_temp,
                max_tokens=2048,
            )
            stage2_data = _extract_json(stage2_resp)
            items_by_dimension[dim_name] = stage2_data.get("items", [])

        return Questionnaire(
            brief=brief_context,
            context=context,
            dimensions=dimensions,
            items=items_by_dimension,
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
