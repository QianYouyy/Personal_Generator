"""LLM 调用封装，支持 OpenAI 和本地模型."""

import os
from typing import Optional
import openai

from src.utils.config import get_config


def load_env():
    """加载环境变量文件.
    
    优先级: .env.development > .env
    .env.development 已加入 .gitignore，用于存放真实 API Key。
    """
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        env_dev = project_root / ".env.development"
        env_default = project_root / ".env"
        
        if env_dev.exists():
            load_dotenv(env_dev, override=True)
        elif env_default.exists():
            load_dotenv(env_default, override=True)
    except ImportError:
        pass


class LLMClient:
    """通用 LLM 客户端，支持同步和异步调用.

    使用方式：
      1. 从配置文件创建: LLMClient.from_config("llm.qgenerator_model")
      2. 直接指定 model: LLMClient(model="gpt-4o")
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        load_env()
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        api_base = api_base or os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise ValueError(
                "API key 未设置。请将 API Key 写入 .env.development 文件\n"
                "（已加入 .gitignore，不会上传到仓库），"
                "或设置 OPENAI_API_KEY 环境变量，"
                "或传入 api_key 参数。"
            )
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)

    @classmethod
    def from_config(
        cls,
        model_key: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> "LLMClient":
        """从配置文件创建 LLMClient.

        Args:
            model_key: 配置中的 model 键名，如 "llm.qgenerator_model"
            api_key: 可选，覆盖配置
            api_base: 可选，覆盖配置
        """
        cfg = get_config()
        model = cfg.get(model_key)
        if not model:
            raise ValueError(
                f"配置中未找到 '{model_key}'，请检查 configs/default.yaml"
            )
        api_base = api_base or cfg.get("llm.api_base")
        return cls(model=model, api_key=api_key, api_base=api_base)

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
