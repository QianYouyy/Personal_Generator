"""生成 50 份预定义情境的问卷."""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_client import LLMClient
from src.qgenerator.generator import QGenerator

# 50 个预定义情境主题（涵盖社会、职场、科技、伦理等多元场景）
BRIEF_CONTEXTS = [
    # 科技与 AI
    "2035 年 AGI 实现后，你突然失业了",
    "你的 AI 助手开始表现出'情感'，你会如何反应",
    "全民基本收入 UBI 实施后，你每天的生活",
    "脑机接口让你能直接'感受'他人的记忆",
    "你发现社交媒体算法正在精准操控你的情绪",
    "自动驾驶汽车面临经典的'电车难题'抉择",
    "人类意识可以被上传到云端，你面临选择",
    "基因编辑婴儿成为现实，社会正在适应",
    "虚拟现实比真实生活更吸引人，你的选择",
    "机器人获得了公民身份，你作为邻居的反应",

    # 职场与组织
    "公司宣布全面远程办公，但监控软件无处不在",
    "你的团队新来了一位能力超群但性格古怪的同事",
    "公司推行'996'制度，你是核心成员之一",
    "你被要求裁掉一半下属，由 AI 替代",
    "你发现了公司产品的严重安全隐患，但上级要求隐瞒",
    "你的创业项目获得了巨额融资，但投资人要求完全控制",
    "公司推行'透明薪酬'制度，所有人薪资公开",
    "你作为中层管理者，夹在高层战略和基层抱怨之间",
    "公司要求所有员工接受'忠诚度测试'",
    "你被派往海外开拓市场，完全陌生的文化环境",

    # 社会与伦理
    "你发现最好的朋友正在传播虚假信息",
    "社区投票决定是否接纳难民家庭入住",
    "你目睹了一起街头冲突，作为旁观者",
    "你的邻居把房子改成了 Airbnb，影响了整个社区",
    "学校推行'学生自由选课'制度，取消所有必修课",
    "城市推行'碳积分'制度，限制个人碳排放",
    "你发现伴侣在社交媒体上使用了完全不同的身份",
    "你的家乡被规划为大型科技园区，需要整体搬迁",
    "社区要求所有居民共享花园和公共空间",
    "你作为陪审团成员，案件证据存在重大疑点",

    # 个人与关系
    "你在 30 岁时突然继承了巨额遗产",
    "你发现父母一直对你隐瞒了一个重大家庭秘密",
    "你的挚友向你出柜，但你所在的社群对此不友好",
    "你被诊断出患有慢性疾病，需要改变整个人生规划",
    "你的双胞胎兄弟姐妹犯了罪，警察找上门来",
    "你在相亲时发现对方是你曾经的竞争对手",
    "你的孩子在学校的价值观教育与你完全相悖",
    "你必须在照顾年迈父母和追求事业之间做出选择",
    "你的前任突然联系你，说你们有一个孩子",
    "你发现自己在网上的匿名账号被人肉了",

    # 危机与极端情境
    "全球性瘟疫再次爆发，你被困在陌生的城市",
    "你乘坐的飞机遭遇严重故障，机长失去意识",
    "你发现所在的国家正在秘密监控所有公民",
    "你参与的探险队在极地失联，物资有限",
    "城市突发大规模停电，持续数周",
    "你作为少数族裔，在歧视性政策下生活",
    "你目睹了一场不公正的执法，需要决定是否作证",
    "你的国家卷入战争，你被征召入伍",
    "你发现一种能治愈所有疾病的药，但成本极高",
    "你拥有预知未来的能力，但无法改变任何事",
]


def main():
    print("=" * 60)
    print("问卷生成器 - 批量生成 50 份心理学问卷")
    print("=" * 60)

    # 初始化 LLM 客户端（model 从 configs/default.yaml 读取）
    llm = LLMClient.from_config("llm.qgenerator_model")
    generator = QGenerator(llm)

    # 测试阶段：只生成前 5 份
    test_contexts = BRIEF_CONTEXTS[:5]
    print(f"\n【测试阶段】开始生成 {len(test_contexts)} 份问卷...\n")
    questionnaires = generator.batch_generate(test_contexts)

    print(f"\n✅ 成功生成 {len(questionnaires)} 份问卷")

    # 保存
    data_dir = Path(__file__).parent.parent / "data" / "questionnaires"
    QGenerator.save(questionnaires, data_dir / "test_5.json")

    print("\n" + "=" * 60)
    print("测试生成完成!")
    print("=" * 60)

    # 打印每份问卷的摘要
    for i, q in enumerate(questionnaires):
        print(f"\n📋 问卷 {i+1}: {q.brief}")
        print(f"  维度: {q.dimensions}")
        grouped = q.items_by_dimension()
        for dim, items in grouped.items():
            print(f"    - {dim}: {len(items)} 题")


if __name__ == "__main__":
    main()
