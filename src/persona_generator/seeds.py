"""3 个初始种子人格生成器."""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List

from src.persona_generator import prompts
from src.utils.config import get_config


class PersonaSeed(ABC):
    """人格生成器种子基类."""

    def __init__(self, llm_client):
        self.llm = llm_client
        cfg = get_config()
        self.stage1_temp = cfg.get("persona_generator.stage1_temperature", 0.9)
        self.stage2_temp = cfg.get("persona_generator.stage2_temperature", 0.8)

    @abstractmethod
    def stage1(self, context: str, dimensions: List[str], n: int) -> List[str]:
        """Stage 1: 生成高层定位标签 p̂_i.

        Returns:
            List[str]: 高层定位标签列表
        """
        pass

    def stage2(self, context: str, dimensions: List[str], high_level_tags: List[str]) -> List[str]:
        """Stage 2: 并行细节扩展.

        所有 seed 共享相同的 Stage 2 策略：形成性记忆。

        Returns:
            List[str]: 完整人格描述列表
        """
        personas = []
        for tag in high_level_tags:
            prompt = prompts.stage2_prompt(context, dimensions, tag)
            resp = self.llm.generate(
                prompt,
                system_prompt=prompts.STAGE2_SYSTEM_PROMPT,
                temperature=self.stage2_temp,
                max_tokens=1024,
            )
            personas.append(resp.strip())
        return personas


class Seed1Default(PersonaSeed):
    """seed1: 默认 Concordia 提示 + 形成性记忆.

    Stage 1 策略：串行逐个生成，每次把已生成的人格加入上下文，
    让模型主动错开，确保多样性空间覆盖。
    """

    def stage1(self, context: str, dimensions: List[str], n: int) -> List[str]:
        tags = []
        for i in range(n):
            prompt = prompts.stage1_seed1_prompt(
                context, dimensions, n, generated_so_far=tags
            )
            resp = self.llm.generate(
                prompt,
                system_prompt=prompts.STAGE1_SYSTEM_PROMPT,
                temperature=self.stage1_temp,
                max_tokens=256,
            )
            tag = resp.strip()
            # 去除可能的编号前缀
            tag = tag.lstrip("1234567890. ").strip()
            tags.append(tag)
        return tags


class Seed2SmallBatch(PersonaSeed):
    """seed2: 小批次自回归 + 形成性记忆.

    Stage 1 策略：每批生成少量人格（如 3-5 个），然后重置上下文。
    减少长上下文依赖，避免模型"疲劳"。
    """

    BATCH_SIZE = 5

    def stage1(self, context: str, dimensions: List[str], n: int) -> List[str]:
        tags = []
        total_batches = (n + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for batch_idx in range(total_batches):
            remaining = n - len(tags)
            batch_size = min(self.BATCH_SIZE, remaining)

            prompt = prompts.stage1_seed2_prompt(
                context, dimensions, batch_size, batch_idx, total_batches
            )
            resp = self.llm.generate(
                prompt,
                system_prompt=prompts.STAGE1_SYSTEM_PROMPT,
                temperature=self.stage1_temp,
                max_tokens=1024,
            )

            # 解析编号列表
            batch_tags = self._parse_numbered_list(resp)
            tags.extend(batch_tags[:batch_size])

        return tags[:n]

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        """解析编号列表文本."""
        lines = text.strip().split("\n")
        tags = []
        for line in lines:
            line = line.strip()
            # 去除编号前缀，如 "1. "、"1) "、"- "
            line = line.lstrip("1234567890.-) ").strip()
            if line and len(line) > 10:
                tags.append(line)
        return tags


class Seed3QuasiRandom(PersonaSeed):
    """seed3: 准随机蒙特卡洛采样 + 形成性记忆.

    Stage 1 策略：
    1. 在 K 维空间用 Sobol 序列均匀撒 N 个点（每个坐标 0-1）
    2. 对每个点，用 LLM 将坐标翻译为文字描述
    """

    def stage1(self, context: str, dimensions: List[str], n: int) -> List[str]:
        k = len(dimensions)

        # 使用 Sobol 序列在 K 维空间均匀采样
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=k, scramble=True)
            points = sampler.random(n=n)
        except Exception:
            # 回退：均匀随机采样
            points = np.random.rand(n, k)

        tags = []
        for i, coords in enumerate(points):
            prompt = prompts.stage1_seed3_coordinate_to_text(
                context, dimensions, coords.tolist()
            )
            resp = self.llm.generate(
                prompt,
                system_prompt=prompts.STAGE1_SYSTEM_PROMPT,
                temperature=self.stage1_temp,
                max_tokens=256,
            )
            tag = resp.strip()
            tag = tag.lstrip("1234567890. ").strip()
            tags.append(tag)

        return tags
