"""LLM 调用封装，支持 OpenAI 和本地模型."""

import os
from typing import Optional
import openai


def load_env():
    """尝试加载 .env 文件."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


class LLMClient:
    """通用 LLM 客户端，支持同步和异步调用."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        load_env()
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        api_base = api_base or os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise ValueError(
                "API key 未设置。请设置 OPENAI_API_KEY 环境变量，"
                "或在 .env 文件中配置，或传入 api_key 参数。"
            )
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """同步调用 LLM 生成文本."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """异步调用 LLM 生成文本."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content
