"""测试问卷生成器（Mock LLM，不调用真实 API）."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qgenerator.generator import QGenerator, Questionnaire


class MockLLM:
    """模拟 LLM，返回预定义的 JSON 响应."""

    def generate(self, prompt, system_prompt=None, **kwargs):
        # 判断是 Stage 1 还是 Stage 2
        if "步骤 1 - 扩展情境" in prompt:
            return self._stage1_response()
        else:
            return self._stage2_response()

    def _stage1_response(self):
        import json
        return json.dumps({
            "context": "这是一个详细的测试情境。在 2035 年，通用人工智能（AGI）已经实现，大量传统工作被自动化取代。你是一位曾经从事软件工程师职业的中年人，突然面临失业的困境。社会正在经历剧烈的转型期，政府推出了全民基本收入（UBI）计划，但金额仅够维持基本生活。你的家庭有房贷和两个孩子要抚养。",
            "dimensions": [
                {
                    "name_cn": "适应倾向",
                    "name_en": "Adaptability",
                    "description": "指个体在面对突变环境时，倾向于主动改变自身（如学习新技能、转换职业）还是坚持原有身份和生活方式。",
                    "poles": ["固守原状", "主动适应"]
                },
                {
                    "name_cn": "风险承受",
                    "name_en": "RiskTolerance",
                    "description": "指个体在不确定的未来面前，倾向于保守稳妥（如接受 UBI 过简朴生活）还是冒险激进（如创业、投资新兴领域）。",
                    "poles": ["风险规避", "风险偏好"]
                }
            ]
        }, ensure_ascii=False)

    def _stage2_response(self):
        import json
        return json.dumps({
            "items": [
                {"text": "我会积极学习 AGI 相关的新技能，寻找新的职业方向。", "scoring": "P"},
                {"text": "我宁可降低生活标准，也不愿意放弃我熟悉的工作领域。", "scoring": "N"},
                {"text": "面对变化，我通常能迅速调整自己的心态和计划。", "scoring": "P"},
                {"text": "我认为坚持自己原有的专业和身份比追逐潮流更重要。", "scoring": "N"},
                {"text": "我愿意尝试完全不同于以往的工作类型，即使这意味着从零开始。", "scoring": "P"},
            ]
        }, ensure_ascii=False)


def main():
    print("=" * 60)
    print("问卷生成器 - Mock 测试")
    print("=" * 60)

    llm = MockLLM()
    generator = QGenerator(llm, items_per_dimension=5)

    print("\n生成单份问卷...")
    q = generator.generate("2035 年 AGI 实现后，你突然失业了", k_dimensions=2)

    print(f"\n✅ 生成成功!")
    print(f"  原始描述: {q.brief}")
    print(f"\n  扩展情境:")
    print(f"    {q.context[:100]}...")

    print(f"\n  多样性轴 ({len(q.dimensions)} 个):")
    for dim in q.dimensions:
        print(f"    • {dim['name_cn']} ({dim['name_en']})")
        print(f"      {dim['description'][:60]}...")
        print(f"      两端: {dim['poles'][0]} ↔ {dim['poles'][1]}")

    print(f"\n  题项:")
    for dim_name, items in q.items.items():
        print(f"    [{dim_name}] {len(items)} 题")
        for item in items:
            direction = "正向" if item.get("scoring") == "P" else "反向"
            print(f"      - [{direction}] {item['text']}")

    # 测试保存/加载
    data_dir = Path(__file__).parent.parent / "data" / "questionnaires"
    QGenerator.save([q], data_dir / "mock_test.json")

    loaded = QGenerator.load(data_dir / "mock_test.json")
    print(f"\n  保存/加载测试: 成功加载 {len(loaded)} 份问卷")

    # 测试 Concordia 格式转换
    concordia_qs = loaded[0].to_concordia_questions()
    print(f"\n  Concordia 格式转换: {len(concordia_qs)} 个 Question 对象")
    print(f"    示例: {concordia_qs[0]}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
