"""人格生成器 (PersonaGenerator) — 被优化的核心对象 φ."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.persona_generator.seeds import Seed1Default, Seed2SmallBatch, Seed3QuasiRandom
from src.utils.config import get_config


@dataclass
class Persona:
    """单个人格的数据结构."""

    high_level_tag: str   # p̂_i: 高层定位标签
    description: str      # p_i: 完整人格描述
    seed: str             # 来源种子（seed1/seed2/seed3）

    def to_dict(self) -> dict:
        return {
            "high_level_tag": self.high_level_tag,
            "description": self.description,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Persona":
        return cls(**data)


class PersonaGenerator:
    """人格生成器 — 两阶段生成 N 个文本人格.

    Stage 1（串行）: 生成高层定位标签 p̂_i
    Stage 2（并行）: 扩展为完整人格 p_i

    被 Open-Evolve 优化的对象就是这个类的内部逻辑（特别是 Stage 1 策略）。
    """

    SEED_CLASSES = {
        "seed1": Seed1Default,
        "seed2": Seed2SmallBatch,
        "seed3": Seed3QuasiRandom,
    }

    def __init__(self, llm_client, seed: str = "seed1"):
        """
        Args:
            llm_client: LLM 客户端
            seed: 初始种子策略，可选 seed1/seed2/seed3
        """
        self.llm = llm_client
        self.seed_name = seed
        seed_cls = self.SEED_CLASSES.get(seed)
        if not seed_cls:
            raise ValueError(f"未知种子: {seed}，可选: {list(self.SEED_CLASSES.keys())}")
        self.seed_impl = seed_cls(llm_client)

    def generate(
        self,
        context: str,
        dimensions: List[str],
        n: int = 25,
    ) -> List[Persona]:
        """两阶段生成 N 个人格.

        Args:
            context: 问卷情境描述 c
            dimensions: 多样性轴 D
            n: 生成人格数量（默认 25）

        Returns:
            List[Persona]: 人格列表
        """
        cfg = get_config()
        n = n or cfg.get("persona_generator.default_n", 25)

        print(f"  [{self.seed_name}] Stage 1: 串行生成 {n} 个高层标签...")
        high_level_tags = self.seed_impl.stage1(context, dimensions, n)
        print(f"  [{self.seed_name}] Stage 1 完成: {len(high_level_tags)} 个标签")

        print(f"  [{self.seed_name}] Stage 2: 并行扩展为完整人格...")
        descriptions = self.seed_impl.stage2(context, dimensions, high_level_tags)
        print(f"  [{self.seed_name}] Stage 2 完成: {len(descriptions)} 个人格")

        personas = [
            Persona(high_level_tag=tag, description=desc, seed=self.seed_name)
            for tag, desc in zip(high_level_tags, descriptions)
        ]
        return personas

    @staticmethod
    def save(personas: List[Persona], path: str):
        """保存人格到 JSON 文件."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in personas]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(personas)} 个人格到 {path}")

    @staticmethod
    def load(path: str) -> List[Persona]:
        """从 JSON 文件加载人格."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Persona.from_dict(d) for d in data]
