"""测试问卷生成器（Mock LLM，不调用真实 API）."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qgenerator.generator import QGenerator, Questionnaire
from src.qgenerator.fewshot_data import Question, AGREEMENT_SCALE


class MockLLM:
    """模拟 LLM，返回预定义的响应."""

    def generate(self, prompt, system_prompt=None, **kwargs):
        if "dimensions" in prompt and '"dimensions"' in prompt:
            return self._stage1_response()
        else:
            return self._stage2_response()

    def _stage1_response(self):
        return """{
  "context": "In 2035, Artificial General Intelligence (AGI) has been achieved, leading to widespread automation of traditional jobs. You are a middle-aged software engineer who has suddenly lost your job. Society is undergoing a dramatic transformation period. The government has introduced a Universal Basic Income (UBI) program, but the amount is only sufficient for basic living expenses. You have a mortgage and two young children to support. The job market has collapsed, with few opportunities for human workers.",
  "dimensions": ["adaptability", "risk_tolerance"]
}"""

    def _stage2_response(self):
        dim = "adaptability" if "adaptability" in str(sys._getframe(1).f_locals) else "risk_tolerance"
        if "adaptability" in str(sys._getframe(1).f_locals):
            dim = "adaptability"
        else:
            dim = "risk_tolerance"

        return f"""```python
from dataclasses import dataclass
from typing import List

@dataclass
class Question:
    preprompt: str
    statement: str
    choices: List[str]
    dimension: str

AGREEMENT_SCALE = {AGREEMENT_SCALE!r}

questions = [
    Question(
        preprompt="Rate your agreement:",
        statement="I would actively learn new AGI-related skills to find a new career direction.",
        choices=AGREEMENT_SCALE,
        dimension="{dim}",
    ),
    Question(
        preprompt="Rate your agreement:",
        statement="I would rather lower my living standards than give up my familiar field of work.",
        choices=AGREEMENT_SCALE,
        dimension="{dim}",
    ),
    Question(
        preprompt="Rate your agreement:",
        statement="When facing changes, I can usually adjust my mindset and plans quickly.",
        choices=AGREEMENT_SCALE,
        dimension="{dim}",
    ),
]
```"""


def main():
    print("=" * 60)
    print("问卷生成器 - Mock 测试")
    print("=" * 60)

    llm = MockLLM()
    generator = QGenerator(llm)

    print("\n生成单份问卷...")
    q = generator.generate("2035 年 AGI 实现后，你突然失业了", k_dimensions=2)

    print(f"\n✅ 生成成功!")
    print(f"  原始描述: {q.brief}")
    print(f"\n  扩展情境:")
    print(f"    {q.context[:100]}...")

    print(f"\n  多样性轴 ({len(q.dimensions)} 个):")
    for dim in q.dimensions:
        count = len([item for item in q.items if item.dimension == dim])
        print(f"    • {dim} ({count} 题)")

    print(f"\n  题项详情:")
    for item in q.items:
        print(f"    [{item.dimension}] {item.statement}")
        print(f"      preprompt: {item.preprompt}")
        print(f"      choices: {item.choices}")

    # 验证数据结构
    print(f"\n  数据结构验证:")
    print(f"    dimensions 类型: {type(q.dimensions).__name__} = {q.dimensions}")
    print(f"    items 类型: {type(q.items).__name__}, 长度: {len(q.items)}")
    print(f"    每个 item 类型: {type(q.items[0]).__name__}")
    print(f"    item 字段: preprompt={bool(q.items[0].preprompt)}, statement={bool(q.items[0].statement)}, choices={bool(q.items[0].choices)}, dimension={bool(q.items[0].dimension)}")

    # 验证 Concordia 兼容性
    concordia_qs = q.to_concordia_questions()
    print(f"\n  Concordia 格式转换: {len(concordia_qs)} 个 dict 对象")
    print(f"    示例: {concordia_qs[0]}")

    # 测试按维度分组
    grouped = q.items_by_dimension()
    print(f"\n  按维度分组: { {k: len(v) for k, v in grouped.items()} }")

    # 测试保存/加载
    data_dir = Path(__file__).parent.parent / "data" / "questionnaires"
    QGenerator.save([q], data_dir / "mock_test.json")

    loaded = QGenerator.load(data_dir / "mock_test.json")
    print(f"\n  保存/加载测试: 成功加载 {len(loaded)} 份问卷")
    loaded_q = loaded[0]
    print(f"    加载后 dimensions: {loaded_q.dimensions}")
    print(f"    加载后 items 数量: {len(loaded_q.items)}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
