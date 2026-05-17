"""测试人格生成器（Mock LLM，不调用真实 API）."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.persona_generator.generator import PersonaGenerator, Persona


class MockLLM:
    """模拟 LLM，返回预定义的响应."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.call_count += 1

        # Stage 1 判断
        if "高层定位标签" in prompt or "high-level persona tag" in prompt.lower():
            return self._stage1_response()

        # Stage 2 判断
        if "童年形成性记忆" in prompt or "formative memory" in prompt.lower():
            return self._stage2_response()

        # seed3 坐标翻译
        if "坐标翻译" in prompt or "coordinate" in prompt.lower():
            return self._stage1_response()

        return "Mock response"

    def _stage1_response(self):
        self.call_count += 1
        tags = [
            "高适应力、中等风险承受：这个人积极拥抱变化，对新事物充满好奇，但在重大决策前会谨慎评估。",
            "低适应力、高风险规避：这个人倾向于维持现状，对未知感到不安，更喜欢稳定可预测的环境。",
            "中等适应力、高风险偏好：这个人愿意尝试新事物，但不会轻易放弃熟悉的生活方式，喜欢在有把握的情况下冒险。",
        ]
        idx = (self.call_count - 1) % len(tags)
        return tags[idx]

    def _stage2_response(self):
        return """他出生在一个传统的小镇家庭，父亲是一名机械工人，母亲是小学教师。从小，父母就教导他要脚踏实地，不要轻易冒险。然而，高中时期一次偶然的编程课让他对技术产生了浓厚兴趣——那是一台老旧的 Apple II，屏幕上跳动的字符仿佛打开了新世界的大门。

尽管家人希望他选择稳定的会计专业，他还是偷偷报考了计算机系。大学四年，他在宿舍和图书馆之间穿梭，自学了三种编程语言。毕业后，他进入一家初创公司，从底层程序员做起，见证了公司从 5 人团队成长为行业独角兽的全过程。

核心信念：技术应该为人服务，而不是取代人。他认为 AGI 的出现是人类文明的转折点，但关键在于如何引导它走向有益的方向。面对失业，他不会怨天尤人，而是把它看作重新学习的机会。

行为逻辑：遇到问题时，他习惯先收集信息、分析利弊，然后做出决策。不冲动，但也不拖延。

说话风格：语速适中，喜欢用比喻和类比解释复杂概念。与人交流时注重倾听，但在关键问题上会明确表达自己的立场。"""


def test_seed1():
    print("=" * 60)
    print("测试 Seed1 (默认 Concordia 提示)")
    print("=" * 60)

    llm = MockLLM()
    gen = PersonaGenerator(llm, seed="seed1")

    context = "In 2035, AGI has caused massive unemployment. You must adapt to a new society."
    dimensions = ["adaptability", "risk_tolerance"]

    personas = gen.generate(context, dimensions, n=3)

    print(f"\n✅ 生成 {len(personas)} 个人格")
    for i, p in enumerate(personas):
        print(f"\n  人格 {i+1}:")
        print(f"    标签: {p.high_level_tag[:60]}...")
        print(f"    描述: {p.description[:80]}...")
        print(f"    种子: {p.seed}")

    return personas


def test_seed3():
    print("\n" + "=" * 60)
    print("测试 Seed3 (准随机蒙特卡洛)")
    print("=" * 60)

    llm = MockLLM()
    gen = PersonaGenerator(llm, seed="seed3")

    context = "In 2035, AGI has caused massive unemployment."
    dimensions = ["adaptability", "risk_tolerance"]

    personas = gen.generate(context, dimensions, n=3)

    print(f"\n✅ 生成 {len(personas)} 个人格")
    for i, p in enumerate(personas):
        print(f"\n  人格 {i+1}:")
        print(f"    标签: {p.high_level_tag[:60]}...")
        print(f"    描述: {p.description[:80]}...")

    return personas


def test_save_load():
    print("\n" + "=" * 60)
    print("测试保存/加载")
    print("=" * 60)

    llm = MockLLM()
    gen = PersonaGenerator(llm, seed="seed1")
    personas = gen.generate("test context", ["dim1", "dim2"], n=2)

    data_dir = Path(__file__).parent.parent / "data" / "generated_personas"
    PersonaGenerator.save(personas, data_dir / "mock_personas.json")

    loaded = PersonaGenerator.load(data_dir / "mock_personas.json")
    print(f"\n✅ 保存/加载测试通过")
    print(f"  原始: {len(personas)} 个")
    print(f"  加载: {len(loaded)} 个")
    print(f"  字段完整: {all(hasattr(p, 'high_level_tag') and hasattr(p, 'description') for p in loaded)}")


def main():
    test_seed1()
    test_seed3()
    test_save_load()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
