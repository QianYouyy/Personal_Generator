"""初始种子的代码模板.

每个种子是一个完整的 Python 代码字符串，定义了一个 PersonaSeed 子类。
变异算子修改这些字符串来生成新的种子。
"""


SEED1_CODE = '''
from concurrent.futures import ThreadPoolExecutor, as_completed

class SeedPersonaGenerator:
    """seed1: 默认 Concordia 提示 + 形成性记忆."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def _stage1_single(self, context: str, dimensions: list, n: int, i: int) -> str:
        """同步生成单个标签."""
        existing_tags = "\\n".join(f"  {j+1}. {t}" for j, t in enumerate([]))
        prompt = f"""【情境】
{context}

【多样性轴】
{"\\n".join(f"  - {d}" for d in dimensions)}

【已生成的人格定位】
（暂无）

【任务】
请生成 1 个新的人格高层定位标签。
要求：
1. 明确指定在各维度上的位置（如"高适应力、中等风险承受"）
2. 与其他人格错开，确保多样性
3. 2-3 句话

直接输出标签文本，不要前缀。"""
        resp = self.llm.generate(prompt, temperature=0.9, max_tokens=256)
        tag = resp.strip().lstrip("0123456789. ").strip()
        return tag

    def stage1(self, context: str, dimensions: list, n: int) -> list:
        """并行生成所有标签."""
        import time
        print(f"    [Stage1] 开始生成 {n} 个标签...")
        total_start = time.time()

        results = [None] * n
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._stage1_single, context, dimensions, n, i): i
                for i in range(n)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    tag = future.result()
                    results[i] = tag
                except Exception:
                    results[i] = "Error"

        total_elapsed = time.time() - total_start
        print(f"    [Stage1] 全部完成: {total_elapsed:.2f}s")
        return results

    def _stage2_single(self, context: str, dimensions: list, tag: str, idx: int) -> str:
        """同步生成单个人格."""
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
        resp = self.llm.generate(prompt, temperature=0.8, max_tokens=512)
        return resp.strip()

    def stage2(self, context: str, dimensions: list, high_level_tags: list) -> list:
        """并行细节扩展 — 形成性记忆."""
        import time
        print(f"    [Stage2] 开始生成 {len(high_level_tags)} 个人格...")
        total_start = time.time()

        results = [None] * len(high_level_tags)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._stage2_single, context, dimensions, tag, i): i
                for i, tag in enumerate(high_level_tags)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    persona = future.result()
                    results[i] = persona
                except Exception:
                    results[i] = "Error"

        total_elapsed = time.time() - total_start
        print(f"    [Stage2] 全部完成: {total_elapsed:.2f}s")
        return results
'''


SEED2_CODE = '''
from concurrent.futures import ThreadPoolExecutor, as_completed

class SeedPersonaGenerator:
    """seed2: 小批次自回归 + 形成性记忆."""

    BATCH_SIZE = 5

    def __init__(self, llm_client):
        self.llm = llm_client

    def stage1(self, context: str, dimensions: list, n: int) -> list:
        """小批次串行生成标签（保持自回归错开逻辑）."""
        import time
        print(f"    [Stage1] 开始生成 {n} 个标签（小批次串行）...")
        total_start = time.time()

        tags = []
        total_batches = (n + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for batch_idx in range(total_batches):
            remaining = n - len(tags)
            batch_size = min(self.BATCH_SIZE, remaining)

            batch_start = time.time()
            existing_tags = "\\n".join(f"  {j+1}. {t}" for j, t in enumerate(tags))
            prompt = f"""【情境】
{context}

【多样性轴】
{"\\n".join(f"  - {d}" for d in dimensions)}

【已生成的人格定位】
{existing_tags if tags else "（暂无）"}

【任务】
请生成 {batch_size} 个新的人格高层定位标签。
要求：
1. 明确指定在各维度上的位置
2. 与已生成的标签错开，确保多样性
3. 每个标签 2-3 句话

直接输出编号列表："""
            resp = self.llm.generate(prompt, temperature=0.9, max_tokens=1024)

            # 解析编号列表
            lines = resp.strip().split("\\n")
            batch_tags = []
            for line in lines:
                line = line.strip().lstrip("01234567890.-) ").strip()
                if line and len(line) > 10:
                    batch_tags.append(line)

            tags.extend(batch_tags[:batch_size])
            batch_elapsed = time.time() - batch_start
            print(f"    [Stage1] Batch {batch_idx+1}/{total_batches} 完成: {batch_elapsed:.2f}s ({len(batch_tags)}个标签)")

        total_elapsed = time.time() - total_start
        print(f"    [Stage1] 全部完成: {total_elapsed:.2f}s")
        return tags[:n]

    def _stage2_single(self, context: str, dimensions: list, tag: str, idx: int) -> str:
        """同步生成单个人格."""
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
        resp = self.llm.generate(prompt, temperature=0.8, max_tokens=512)
        return resp.strip()

    def stage2(self, context: str, dimensions: list, high_level_tags: list) -> list:
        """并行细节扩展 — 形成性记忆."""
        import time
        print(f"    [Stage2] 开始生成 {len(high_level_tags)} 个人格...")
        total_start = time.time()

        results = [None] * len(high_level_tags)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._stage2_single, context, dimensions, tag, i): i
                for i, tag in enumerate(high_level_tags)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    persona = future.result()
                    results[i] = persona
                except Exception:
                    results[i] = "Error"

        total_elapsed = time.time() - total_start
        print(f"    [Stage2] 全部完成: {total_elapsed:.2f}s")
        return results
'''


SEED3_CODE = '''
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

class SeedPersonaGenerator:
    """seed3: 准随机蒙特卡洛采样 + 形成性记忆."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def _stage1_single(self, context: str, dimensions: list, coords: list, i: int) -> str:
        """同步生成单个标签."""
        coords_str = ", ".join(f"{d}={c:.2f}" for d, c in zip(dimensions, coords))
        prompt = f"""【情境】
{context}

【人格坐标】
{coords_str}

【任务】
将上述坐标翻译为一段人格高层定位标签（2-3 句话）。
描述这个人在各维度上的位置和行为特征。"""
        resp = self.llm.generate(prompt, temperature=0.9, max_tokens=256)
        tag = resp.strip().lstrip("0123456789. ").strip()
        return tag

    def stage1(self, context: str, dimensions: list, n: int) -> list:
        """并行生成所有标签."""
        import time
        print(f"    [Stage1] 开始生成 {n} 个标签（蒙特卡洛采样）...")
        total_start = time.time()

        k = len(dimensions)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=k, scramble=True)
            m = int(np.ceil(np.log2(max(n, 1))))
            points = sampler.random_base2(m=m)[:n]
        except Exception:
            points = np.random.rand(n, k)

        results = [None] * n
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._stage1_single, context, dimensions, coords.tolist(), i): i
                for i, coords in enumerate(points)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    tag = future.result()
                    results[i] = tag
                except Exception:
                    results[i] = "Error"

        total_elapsed = time.time() - total_start
        print(f"    [Stage1] 全部完成: {total_elapsed:.2f}s")
        return results

    def _stage2_single(self, context: str, dimensions: list, tag: str, idx: int) -> str:
        """同步生成单个人格."""
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
        resp = self.llm.generate(prompt, temperature=0.8, max_tokens=512)
        return resp.strip()

    def stage2(self, context: str, dimensions: list, high_level_tags: list) -> list:
        """并行细节扩展 — 形成性记忆."""
        import time
        print(f"    [Stage2] 开始生成 {len(high_level_tags)} 个人格...")
        total_start = time.time()

        results = [None] * len(high_level_tags)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._stage2_single, context, dimensions, tag, i): i
                for i, tag in enumerate(high_level_tags)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    persona = future.result()
                    results[i] = persona
                except Exception:
                    results[i] = "Error"

        total_elapsed = time.time() - total_start
        print(f"    [Stage2] 全部完成: {total_elapsed:.2f}s")
        return results
'''


SEED_CODES = {
    "seed1": SEED1_CODE,
    "seed2": SEED2_CODE,
    "seed3": SEED3_CODE,
}
