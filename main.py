"""
Persona Generator — 主入口

完整 pipeline：
  1. 加载问卷（train/val/test）
  2. 初始化 LLM、评估器、变异算子
  3. 启动 Open-Evolve 进化引擎
  4. 输出最优人格生成器代码
  5. 用 test 问卷做最终评估
  6. 生成可视化图表

用法：
  python main.py
  python main.py --generations 50 --train data/questionnaires/train.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.utils.llm_client import LLMClient
from src.utils.config import get_config
from src.utils.logger import logger
from src.utils.output_manager import output_manager
from src.qgenerator.generator import QGenerator
from src.open_evolve.mutator import Mutator
from src.open_evolve.evaluator import PersonaCodeEvaluator
from src.open_evolve.engine import OpenEvolve, Candidate
from src.utils.visualization import generate_all_visualizations


def load_questionnaires(path: str):
    """加载问卷."""
    logger.info(f"加载问卷: {path}")
    qs = QGenerator.load(path)
    logger.success(f"加载了 {len(qs)} 份问卷")
    for i, q in enumerate(qs):
        logger.debug(f"  问卷 {i+1}: {q.brief[:50]}... | 维度: {q.dimensions} | 题项: {len(q.items)}")
    return qs


def setup_components(questionnaires):
    """初始化所有组件.
    
    各组件使用独立模型：
    - Mutator: mutator_model（代码改写，需要强模型）
    - Evaluator.persona_llm: persona_model（人格生成）
    - Evaluator.simulator: simulator_model（模拟回答）
    """
    logger.step("初始化组件", 1, 4)

    # 变异算子使用 mutator_model
    logger.info("创建变异算子 (Mutator)...")
    mutator_llm = LLMClient.from_config("llm.mutator_model")
    mutator = Mutator(mutator_llm)
    logger.success(f"变异算子就绪: {mutator_llm.model}")

    # 评估器使用 persona_model + simulator_model
    logger.info("创建评估器 (Evaluator)...")
    cfg = get_config()
    num_personas = cfg.get("open_evolve.personas_per_evaluation", 5)
    simulator_llm = LLMClient.from_config("llm.simulator_model")
    evaluator = PersonaCodeEvaluator(
        questionnaires=questionnaires,
        llm_client=simulator_llm,
        num_personas=num_personas,
    )
    logger.success(f"评估器就绪 | 人格模型: {evaluator.persona_llm.model} | 模拟模型: {simulator_llm.model}")
    logger.info(f"每份问卷生成 {num_personas} 个人格")

    return mutator, evaluator


def run_evolution(
    mutator,
    evaluator,
    questionnaires,
    max_generations: int = 20,
    max_hours: float = None,
):
    """运行进化."""
    logger.step("启动 Open-Evolve 进化", 2, 4)

    logger.info(f"配置: {len(questionnaires)} 份问卷, 最大 {max_generations} 轮")
    logger.info(f"岛屿数: 10 | 每岛精英位: 6 | 灭绝间隔: 100 轮/8h")

    engine = OpenEvolve(
        mutator=mutator,
        evaluator=evaluator,
        questionnaires=questionnaires,
    )

    logger.info("开始进化...")
    best = engine.run(
        max_generations=max_generations,
        max_hours=max_hours,
    )

    return engine, best


def final_evaluation(best: Candidate, test_questionnaires):
    """用 test 问卷做最终评估.
    
    使用独立的 persona_model + simulator_model 进行评估。
    """
    if not test_questionnaires:
        logger.warn("无测试问卷，跳过最终评估")
        return

    logger.step("最终评估（Test 问卷集）", 3, 4)
    logger.info(f"使用 {len(test_questionnaires)} 份测试问卷评估最优代码...")

    # 最终评估使用 simulator_model（模拟器）
    simulator_llm = LLMClient.from_config("llm.simulator_model")
    cfg = get_config()
    num_personas = cfg.get("open_evolve.personas_per_evaluation", 5)
    evaluator = PersonaCodeEvaluator(
        questionnaires=test_questionnaires,
        llm_client=simulator_llm,
        num_personas=num_personas,
    )

    logger.info("执行评估...")
    fitness = evaluator.evaluate(best.code)

    logger.success("Test 集评估结果:")
    for k, v in fitness.items():
        logger.metric(k, v)

    # 保存最优代码到统一输出目录
    code_path = output_manager.get_output_path("best_persona_generator.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(best.code)
    logger.success(f"最优代码已保存: {code_path}")

    result_path = output_manager.get_output_path("final_evaluation.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "fitness": fitness,
            "generation": best.generation,
            "island_id": best.island_id,
            "seed_name": best.seed_name,
        }, f, ensure_ascii=False, indent=2)
    logger.success(f"评估结果已保存: {result_path}")


def generate_viz(engine, test_questionnaires):
    """生成可视化图表."""
    logger.step("生成可视化图表", 4, 4)

    try:
        if not engine or not engine.history:
            logger.warn("无进化历史数据，跳过可视化")
            return

        logger.info(f"进化历史: {len(engine.history)} 轮")

        # 构造 islands_data
        islands_data = {}
        for island in engine.islands:
            data = {}
            for metric, candidate in island.elites.items():
                data[metric] = candidate.fitness.get(metric, 0)
            islands_data[island.id] = data
            logger.debug(f"  Island {island.id}: { {k: f'{v:.3f}' for k, v in data.items()} }")

        # 用占位 Z（实际运行时会从评估器获取真实 Z）
        import numpy as np
        n_dims = len(test_questionnaires[0].dimensions) if test_questionnaires else 2
        Z_placeholder = np.random.rand(25, n_dims)

        viz_dir = output_manager.viz_dir
        
        # 获取最优候选的 fitness 用于雷达图
        best_fitness = None
        if best := engine.get_global_best():
            best_fitness = best.fitness
        
        # 推断灭绝轮数（从 engine 配置）
        extinction_gens = getattr(engine, '_extinction_log', None)
        
        generate_all_visualizations(
            Z=Z_placeholder,
            dimensions=test_questionnaires[0].dimensions if test_questionnaires else ["dim1", "dim2"],
            history=engine.history,
            islands_data=islands_data,
            output_dir=viz_dir,
            extinction_generations=extinction_gens,
            best_fitness=best_fitness,
        )
        logger.success(f"可视化图表已生成: {viz_dir}")

    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Persona Generator — 人格生成器进化")
    parser.add_argument("--train", default="data/questionnaires/train.json",
                        help="训练问卷路径（用于进化，默认30份训练问卷）")
    parser.add_argument("--test", default="data/questionnaires/test.json",
                        help="测试问卷路径（用于最终评估，默认10份测试问卷）")
    parser.add_argument("--generations", type=int, default=20,
                        help="最大进化轮数（默认20）")
    parser.add_argument("--hours", type=float, default=None,
                        help="最大运行时间（小时）")
    parser.add_argument("--eval-model", default="llm.qgenerator_model",
                        help="评估时使用的模型配置键")
    parser.add_argument("--name", default="default",
                        help="运行名称，用于命名输出目录")

    args = parser.parse_args()
    
    # 初始化输出管理器（统一目录结构）
    output_manager.setup(args.name)
    
    # 设置日志记录器
    logger.setup(args.name)
    
    # 设置 API 记录器
    from src.utils.llm_client import api_recorder
    api_recorder.setup(args.name)

    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"开始时间: {start_time_str}")

    logger.section("Persona Generator — Open-Evolve 进化引擎")

    # 读取模型配置
    cfg = get_config()
    eval_model_name = cfg.get(args.eval_model, "unknown")
    logger.info(f"评估模型: {eval_model_name} (配置键: {args.eval_model})")
    logger.info(f"配置: generations={args.generations}")
    logger.info(f"训练问卷: {args.train}")
    logger.info(f"测试问卷: {args.test}")

    # 加载问卷
    train_qs = load_questionnaires(args.train)
    test_qs = None
    if args.test and args.test != args.train:
        try:
            test_qs = load_questionnaires(args.test)
        except FileNotFoundError:
            logger.error(f"测试问卷不存在: {args.test}")
    elif args.test == args.train:
        logger.info("训练集和测试集相同，复用已加载问卷")
        test_qs = train_qs

    if not train_qs:
        logger.error("没有加载到训练问卷，退出")
        sys.exit(1)

    # 初始化组件
    mutator, evaluator = setup_components(train_qs)

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
        final_evaluation(best, test_qs)
        generate_viz(engine, test_qs)
    else:
        logger.error("进化未产生最优解")

    logger.section("全部完成!")
    logger.info(f"开始时间: {start_time_str}")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总运行时间: {logger._timestamp()}")
    logger.info(f"输出目录: {output_manager.base_dir}")


if __name__ == "__main__":
    main()
