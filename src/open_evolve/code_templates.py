"""初始种子的代码模板.

每个种子是一个完整的 Python 代码字符串，定义了一个 PersonaSeed 子类。
变异算子修改这些字符串来生成新的种子。
"""


SEED1_CODE = '''
class SeedPersonaGenerator:
    """seed1: 默认 Concordia 提示 + 形成性记忆."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def stage1(self, context: str, dimensions: list, n: int) -> list:
        """串行逐个生成高层定位标签，主动错开."""
        tags = []
        for i in range(n):
            existing_tags = "\\n".join(f"  {j+1}. {t}" for j, t in enumerate(tags))
            prompt = f"""【情境】
{context}

【多样性轴】
{"\\n".join(f"  - {d}" for d in dimensions)}

【已生成的人格定位】
{existing_tags if tags else "（暂无）"}

【任务】
请生成 1 个新的人格高层定位标签。
要求：
1. 明确指定在各维度上的位置（如"高适应力、中等风险承受"）
2. 与已生成的人格错开，确保多样性
3. 2-3 句话

直接输出标签文本，不要前缀。"""
            resp = self.llm.generate(prompt, temperature=0.9, max_tokens=256)
            tag = resp.strip().lstrip("0123456789. ").strip()
            tags.append(tag)
        return tags

    def stage2(self, context: str, dimensions: list, high_level_tags: list) -> list:
        """并行细节扩展 — 形成性记忆（从童年写起）."""
        personas = []
        for tag in high_level_tags:
            prompt = f"""【情境】
{context}

【高层定位标签】
{tag}

【任务】
将上述标签扩展为完整的人格描述（200-400 词）。
从童年形成性记忆写起，包含：
1. 童年经历如何塑造核心特质
2. 核心信念和价值观
3. 行为逻辑和决策方式
4. 说话风格和社交模式

用第三人称"他/她"。"""
            resp = self.llm.generate(prompt, temperature=0.8, max_tokens=1024)
            personas.append(resp.strip())
        return personas
'''


SEED2_CODE = '''
class SeedPersonaGenerator:
    """seed2: 小批次自回归 + 形成性记忆."""

    BATCH_SIZE = 5

    def __init__(self, llm_client):
        self.llm = llm_client

    def stage1(self, context: str, dimensions: list, n: int) -> list:
        """小批次生成，减少上下文依赖."""
        tags = []
        total_batches = (n + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for batch_idx in range(total_batches):
            remaining = n - len(tags)
            batch_size = min(self.BATCH_SIZE, remaining)
            prompt = f"""【情境】
{context}

【多样性轴】
{"\\n".join(f"  - {d}" for d in dimensions)}

【任务】
这是第 {batch_idx+1}/{total_batches} 批。
请生成 {batch_size} 个不同的人格高层定位标签。
要求：
1. 每个标签明确在各维度上的位置
2. {batch_size} 个标签之间必须相互错开
3. 每标签 2-3 句话

用编号列表输出。"""
            resp = self.llm.generate(prompt, temperature=0.9, max_tokens=1024)
            # 解析编号列表
            for line in resp.strip().split("\\n"):
                line = line.strip().lstrip("01234567890.-) ").strip()
                if line and len(line) > 10:
                    tags.append(line)
            if len(tags) >= n:
                break
        return tags[:n]

    def stage2(self, context: str, dimensions: list, high_level_tags: list) -> list:
        """并行细节扩展 — 形成性记忆."""
        personas = []
        for tag in high_level_tags:
            prompt = f"""【情境】
{context}

【高层定位标签】
{tag}

【任务】
将上述标签扩展为完整的人格描述（200-400 词）。
从童年形成性记忆写起，包含：
1. 童年经历如何塑造核心特质
2. 核心信念和价值观
3. 行为逻辑和决策方式
4. 说话风格和社交模式

用第三人称"他/她"。"""
            resp = self.llm.generate(prompt, temperature=0.8, max_tokens=1024)
            personas.append(resp.strip())
        return personas
'''


SEED3_CODE = '''
import numpy as np

class SeedPersonaGenerator:
    """seed3: 准随机蒙特卡洛采样 + 形成性记忆."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def stage1(self, context: str, dimensions: list, n: int) -> list:
        """在 K 维空间均匀撒点，再翻译为文字."""
        k = len(dimensions)
        # Sobol 采样
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=k, scramble=True)
            points = sampler.random(n=n)
        except Exception:
            points = np.random.rand(n, k)

        tags = []
        for coords in points:
            coord_str = "\\n".join(
                f"  - {dim}: {c:.2f} (0=极低, 1=极高)"
                for dim, c in zip(dimensions, coords)
            )
            prompt = f"""【情境】
{context}

【坐标定位】
{coord_str}

【任务】
将上述坐标翻译成一段人格高层定位标签。
描述一个具有这些维度坐标的完整人格轮廓。
2-3 句话，生动有辨识度。"""
            resp = self.llm.generate(prompt, temperature=0.9, max_tokens=256)
            tag = resp.strip().lstrip("01234567890. ").strip()
            tags.append(tag)
        return tags

    def stage2(self, context: str, dimensions: list, high_level_tags: list) -> list:
        """并行细节扩展 — 形成性记忆."""
        personas = []
        for tag in high_level_tags:
            prompt = f"""【情境】
{context}

【高层定位标签】
{tag}

【任务】
将上述标签扩展为完整的人格描述（200-400 词）。
从童年形成性记忆写起，包含：
1. 童年经历如何塑造核心特质
2. 核心信念和价值观
3. 行为逻辑和决策方式
4. 说话风格和社交模式

用第三人称"他/她"。"""
            resp = self.llm.generate(prompt, temperature=0.8, max_tokens=1024)
            personas.append(resp.strip())
        return personas
'''


SEED_CODES = {
    "seed1": SEED1_CODE,
    "seed2": SEED2_CODE,
    "seed3": SEED3_CODE,
}
