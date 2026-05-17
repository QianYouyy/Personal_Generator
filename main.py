"""
Persona Generator — 主入口

完整 pipeline：
  1. 加载问卷（train/val/test）
  2. 初始化 LLM、评估器、变异算子
  3. 启动 Open-Evolve 进化引擎
  4. 输出最优人格生成器代码
  5. 用 test 问卷做最终评估

用法：
  python main.py --generations 100 --questionnaires data/questionnaires/train.json
  python main.py --resume data/results/checkpoint_gen_50.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.utils.llm_client import LLMClient
from src.utils.config import get_config
from src.qgenerator.generator import QGenerator
from src.open_evolve.mutator import Mutator
from src.open_evolve.evaluator import PersonaCodeEvaluator
from src.open_evolve.engine import OpenEvolve, Candidate


def load_questionnaires(path: str):
    """加载问卷."""
    print(f"[Main] 加载问卷: {path}")
    qs = QGenerator.load(path)
    print(f"[Main] 加载了 {len(qs)} 份问卷")
    return qs


def setup_components(questionnaires, llm_model_key: str = "llm.qgenerator_model"):
    """初始化所有组件."""
    print("[Main] 初始化组件...")

    llm = LLMClient.from_config(llm_model_key)
    mutator = Mutator(llm)
    evaluator = PersonaCodeEvaluator(
        questionnaires=questionnaires,
        llm_client=llm,
        num_personas=25,
    )
    return mutator, evaluator


def run_evolution(
    mutator,
    evaluator,
    questionnaires,
    max_generations: int = 100,
    max_hours: float = None,
):
    """运行进化."""
    print("\n" + "=" * 70)
    print("启动 Open-Evolve 进化")
    print("=" * 70)

    engine = OpenEvolve(
        mutator=mutator,
        evaluator=evaluator,
        questionnaires=questionnaires,
    )

    best = engine.run(
        max_generations=max_generations,
        max_hours=max_hours,
    )

    return engine, best


def final_evaluation(best: Candidate, test_questionnaires, llm_model_key: str = "llm.qgenerator_model"):
    """用 test 问卷做最终评估（全程不参与进化）."""
    if not test_questionnaires:
        print("[Main] 无测试问卷，跳过最终评估")
        return

    print("\n" + "=" * 70)
    print("最终评估（Test 问卷集）")
    print("=" * 70)

    llm = LLMClient.from_config(llm_model_key)
    evaluator = PersonaCodeEvaluator(
        questionnaires=test_questionnaires,
        llm_client=llm,
        num_personas=25,
    )

    fitness = evaluator.evaluate(best.code)
    print(f"\n[Main] Test 集适应度:")
    for k, v in fitness.items():
        print(f"  {k:20s}: {v:+.6f}")

    # 保存最优代码
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    code_path = output_dir / "best_persona_generator.py"
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(best.code)
    print(f"\n[Main] 最优代码已保存到: {code_path}")

    result_path = output_dir / "final_evaluation.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "fitness": fitness,
            "generation": best.generation,
            "island_id": best.island_id,
            "seed_name": best.seed_name,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Main] 评估结果已保存到: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Persona Generator — 人格生成器进化")
    parser.add_argument("--train", default="data/questionnaires/train.json",
                        help="训练问卷路径（用于进化）")
    parser.add_argument("--test", default="data/questionnaires/test.json",
                        help="测试问卷路径（用于最终评估）")
    parser.add_argument("--generations", type=int, default=100,
                        help="最大进化轮数")
    parser.add_argument("--hours", type=float, default=None,
                        help="最大运行时间（小时）")
    parser.add_argument("--eval-model", default="llm.qgenerator_model",
                        help="评估时使用的模型配置键")

    args = parser.parse_args()

    print("=" * 70)
    print("Persona Generator — Open-Evolve 进化引擎")
    print("=" * 70)

    # 加载问卷
    train_qs = load_questionnaires(args.train)
    test_qs = None
    if args.test:
        try:
            test_qs = load_questionnaires(args.test)
        except FileNotFoundError:
            print(f"[Main] 测试问卷不存在: {args.test}")

    if not train_qs:
        print("[Main] 错误: 没有加载到训练问卷")
        sys.exit(1)

    # 初始化组件
    mutator, evaluator = setup_components(train_qs, args.eval_model)

    # 运行进化
    engine, best = run_evolution(
        mutator=mutator,
        evaluator=evaluator,
        questionnaires=train_qs,
        max_generations=args.generations,
        max_hours=args.hours,
    )

    # 最终评估
    if best:
        final_evaluation(best, test_qs, args.eval_model)

    print("\n" + "=" * 70)
    print("全部完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
