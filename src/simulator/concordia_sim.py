"""Concordia 模拟器 — 将文本人格转化为数值化回答嵌入.

基于 Concordia 框架核心概念，但简化了配置流程，直接通过 LLM 调用实现"适当性逻辑"。
支持多线程并行加速。
"""

import re
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from src.qgenerator.fewshot_data import AGREEMENT_SCALE


@dataclass
class ConcordiaAgent:
    """简化版 Concordia Agent."""

    name: str
    persona_description: str
    memory: List[str]

    def reset(self):
        self.memory = []

    def observe(self, observation: str):
        self.memory.append(observation)

    def answer_question(
        self,
        question_statement: str,
        choices: List[str],
        llm_client,
        context: str = "",
    ) -> int:
        """让 Agent 回答一道问卷题项."""
        memory_context = "\n".join(
            f"- {m}" for m in self.memory[-5:]
        ) if self.memory else "（暂无相关记忆）"

        choices_str = "\n".join(
            f"  {i}. {choice}" for i, choice in enumerate(choices)
        )

        prompt = f"""【情境】
{context}

【你的人格描述】
{self.persona_description}

【你最近的记忆】
{memory_context}

【问卷题项】
{question_statement}

【可选回答】
{choices_str}

【任务】
请根据上述情境、你的人格描述和记忆，选择最符合你这个人的回答。
只输出选项编号（0-{len(choices)-1}），不要任何解释。
"""

        resp = llm_client.generate(
            prompt,
            temperature=0.3,
            max_tokens=10,
        )

        match = re.search(r'\b(\d+)\b', resp.strip())
        if match:
            choice_idx = int(match.group(1))
            choice_idx = max(0, min(choice_idx, len(choices) - 1))
        else:
            choice_idx = len(choices) // 2

        self.observe(f"回答了问题：'{question_statement[:50]}...' → 选择 '{choices[choice_idx]}'")
        return choice_idx


class ConcordiaSimulator:
    """Concordia 模拟器 — 支持多线程并行加速."""

    def __init__(self, llm_client, max_workers: int = 5):
        self.llm = llm_client
        self.max_workers = max_workers

    def _answer_single(
        self,
        agent_idx: int,
        agent: ConcordiaAgent,
        item_idx: int,
        statement: str,
        choices: List[str],
        context: str,
    ) -> tuple:
        """回答单道题（用于多线程）."""
        agent.reset()
        choice_idx = agent.answer_question(
            question_statement=statement,
            choices=choices,
            llm_client=self.llm,
            context=context,
        )
        return (agent_idx, item_idx, choice_idx)

    def simulate(
        self,
        personas: List[Dict],
        questionnaire: Dict,
    ) -> np.ndarray:
        """模拟人格群体回答问卷（多线程加速）."""
        context = questionnaire.get("context", "")
        dimensions = questionnaire.get("dimensions", [])
        items = questionnaire.get("items", [])
        k = len(dimensions)
        n = len(personas)

        logger_msg = f"[Simulator] 模拟 {n} 个人格回答 {len(items)} 道题 (并行 {self.max_workers} 线程)..."
        print(f"  {logger_msg}")

        # 创建 Agent
        agents = []
        for i, p in enumerate(personas):
            desc = p.get("description", p) if isinstance(p, dict) else str(p)
            agents.append(ConcordiaAgent(
                name=f"Persona_{i+1}",
                persona_description=desc,
                memory=[],
            ))

        answers = np.zeros((n, len(items)), dtype=int)

        # 多线程并行：所有 (人格, 题项) 组合同时处理
        total_tasks = n * len(items)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for item_idx, item in enumerate(items):
                statement = item.get("statement", item.get("text", ""))
                choices = item.get("choices", AGREEMENT_SCALE)
                for agent_idx, agent in enumerate(agents):
                    future = executor.submit(
                        self._answer_single,
                        agent_idx, agent, item_idx,
                        statement, choices, context
                    )
                    futures[future] = (agent_idx, item_idx)

            for future in as_completed(futures):
                agent_idx, item_idx = futures[future]
                try:
                    _, _, choice_idx = future.result()
                    answers[agent_idx, item_idx] = choice_idx
                except Exception as e:
                    print(f"    [Simulator] 回答失败 (agent={agent_idx}, item={item_idx}): {e}")

                completed += 1
                # 进度打印已禁用（减少日志噪音）
                # if completed % 10 == 0 or completed == total_tasks:
                #     print(f"    [Simulator] 进度: {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%)")

        # 按维度取平均
        dim_to_items = {dim: [] for dim in dimensions}
        for item_idx, item in enumerate(items):
            dim = item.get("dimension", "")
            if dim in dim_to_items:
                dim_to_items[dim].append(item_idx)

        Z = np.zeros((n, k))
        for dim_idx, dim in enumerate(dimensions):
            item_indices = dim_to_items.get(dim, [])
            if item_indices:
                max_choice = len(items[0].get("choices", AGREEMENT_SCALE)) - 1
                dim_scores = answers[:, item_indices] / max_choice
                Z[:, dim_idx] = dim_scores.mean(axis=1)

        print(f"  [Simulator] 完成: Z.shape = {Z.shape}")
        return Z
