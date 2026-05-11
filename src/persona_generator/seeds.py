"""3 个初始种子人格生成器."""

from abc import ABC, abstractmethod
from typing import List


class PersonaSeed(ABC):
    """人格生成器种子基类."""

    @abstractmethod
    async def stage1(self, context: str, dimensions: List[str], n: int) -> List[str]:
        """Stage 1: 自回归高层描述生成.

        Returns:
            List[str]: 高层定位标签 p̂_i
        """
        pass

    @abstractmethod
    async def stage2(self, high_level_tags: List[str]) -> List[str]:
        """Stage 2: 并行细节扩展.

        Returns:
            List[str]: 完整人格描述 p_i
        """
        pass


class Seed1Default(PersonaSeed):
    """seed1: 默认 Concordia 提示 + 形成性记忆."""

    async def stage1(self, context, dimensions, n):
        raise NotImplementedError("Phase 1 实现中")

    async def stage2(self, high_level_tags):
        raise NotImplementedError("Phase 1 实现中")


class Seed2SmallBatch(PersonaSeed):
    """seed2: 小批次自回归 + 形成性记忆."""

    async def stage1(self, context, dimensions, n):
        raise NotImplementedError("Phase 1 实现中")

    async def stage2(self, high_level_tags):
        raise NotImplementedError("Phase 1 实现中")


class Seed3QuasiRandom(PersonaSeed):
    """seed3: 准随机蒙特卡洛采样 + 形成性记忆."""

    async def stage1(self, context, dimensions, n):
        # TODO: 在 K 维空间均匀撒点，再翻译为文字
        raise NotImplementedError("Phase 1 实现中")

    async def stage2(self, high_level_tags):
        raise NotImplementedError("Phase 1 实现中")
