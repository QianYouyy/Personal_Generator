"""Concordia 模拟器 — 将文本人格转化为数值化回答嵌入.

本实现基于 Concordia 框架的核心概念：
  - AssociativeMemoryBank: Agent 的记忆系统
  - EntityAgentWithLogging: 带日志的 Agent
  - QuestionOfRecentMemories: 让 Agent 回答问题

但简化了配置流程，直接通过 LLM 调用实现"适当性逻辑"：
  1. "这是什么样的情境？"
  2. "我是什么样的人？"
  3. "像我这样的人会怎么做？"

每道题后重置 Agent 记忆，防止顺序效应。
"""

import re
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

from src.qgenerator.fewshot_data import AGREEMENT_SCALE


@dataclass
class ConcordiaAgent:
    """简化版 Concordia Agent.

    封装了人格描述和记忆状态，核心行为是让 LLM 根据人格回答问卷题项。
    """

    name: str
    persona_description: str  # p_i: 完整人格描述
    memory: List[str]         # 模拟 Concordia 的 AssociativeMemory

    def reset(self):
        """重置记忆（防止顺序效应）."""
        self.memory = []

    def observe(self, observation: str):
        """观察/记忆一条信息."""
        self.memory.append(observation)

    def answer_question(
        self,
        question_statement: str,
        choices: List[str],
        llm_client,
        context: str = "",
    ) -> int:
        """让 Agent 回答一道问卷题项.

        使用"适当性逻辑"(Logic of Appropriateness)：
          1. 识别情境
          2. 回忆自己是谁
          3. 做出符合人格的选择

        Args:
            question_statement: 题项陈述
            choices: 选项列表（如 Likert 5 点量表）
            llm_client: LLM 客户端
            context: 问卷情境描述

        Returns:
            int: 选择的选项索引（0-based）
        """
        # 构建适当性逻辑的 prompt
        memory_context = "\n".join(
            f"- {m}" for m in self.memory[-5:]  # 最近 5 条记忆
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
            temperature=0.3,  # 低温度确保一致性
            max_tokens=10,
        )

        # 解析选项编号
        match = re.search(r'\b(\d+)\b', resp.strip())
        if match:
            choice_idx = int(match.group(1))
            choice_idx = max(0, min(choice_idx, len(choices) - 1))
        else:
            # 回退：默认选中间值
            choice_idx = len(choices) // 2

        # 记录这次回答到记忆
        self.observe(f"回答了问题：'{question_statement[:50]}...' → 选择 '{choices[choice_idx]}'")

        return choice_idx


class ConcordiaSimulator:
    """Concordia 模拟器 — 将人格群体转化为数值化回答嵌入.

    输入：P（人格群体），I（问卷题项）
    输出：Z = {z₁...z_N} ⊂ ℝ^K
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def simulate(
        self,
        personas: List[Dict],  # 人格描述列表，每个含 "description" 字段
        questionnaire: Dict,   # 问卷数据，含 "context", "dimensions", "items"
    ) -> np.ndarray:
        """模拟人格群体回答问卷.

        Args:
            personas: 人格描述列表，每项为 {"description": str}
            questionnaire: 问卷数据结构
                - context: str
                - dimensions: List[str]
                - items: List[Question]（含 statement, choices, dimension）

        Returns:
            np.ndarray: Z = {z₁...z_N} ⊂ ℝ^K，每行是一个人格在各维度上的平均得分
        """
        context = questionnaire.get("context", "")
        dimensions = questionnaire.get("dimensions", [])
        items = questionnaire.get("items", [])
        k = len(dimensions)
        n = len(personas)

        print(f"  [Simulator] 模拟 {n} 个人格回答 {len(items)} 道题...")

        # 为每个人格创建 Agent
        agents = []
        for i, p in enumerate(personas):
            desc = p.get("description", p) if isinstance(p, dict) else str(p)
            agent = ConcordiaAgent(
                name=f"Persona_{i+1}",
                persona_description=desc,
                memory=[],
            )
            agents.append(agent)

        # 收集回答
        # answers[persona_idx][item_idx] = choice_idx
        answers = np.zeros((n, len(items)), dtype=int)

        for item_idx, item in enumerate(items):
            statement = item.get("statement", item.get("text", ""))
            choices = item.get("choices", AGREEMENT_SCALE)
            dim = item.get("dimension", "")

            for agent_idx, agent in enumerate(agents):
                # 每道题前重置记忆（防止顺序效应）
                agent.reset()

                # 让 Agent 回答
                choice_idx = agent.answer_question(
                    question_statement=statement,
                    choices=choices,
                    llm_client=self.llm,
                    context=context,
                )
                answers[agent_idx, item_idx] = choice_idx

        # 按维度取平均 → z_i
        # 构建维度到题项索引的映射
        dim_to_items = {dim: [] for dim in dimensions}
        for item_idx, item in enumerate(items):
            dim = item.get("dimension", "")
            if dim in dim_to_items:
                dim_to_items[dim].append(item_idx)

        Z = np.zeros((n, k))
        for dim_idx, dim in enumerate(dimensions):
            item_indices = dim_to_items.get(dim, [])
            if item_indices:
                # 将选项索引归一化到 [0, 1]
                # 假设选项是等距的 Likert 量表
                max_choice = len(items[0].get("choices", AGREEMENT_SCALE)) - 1
                dim_scores = answers[:, item_indices] / max_choice
                Z[:, dim_idx] = dim_scores.mean(axis=1)

        print(f"  [Simulator] 完成: Z.shape = {Z.shape}")
        return Z
