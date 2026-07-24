"""LLM 客户端 — 统一 API 调用 + 通信记录.

支持两种创建方式:
  1. 从配置文件创建: LLMClient.from_config("llm.persona_model")
  2. 直接指定 model: LLMClient(model="gpt-5.4")

API 记录由 OutputManager 统一管理:
data/results/{run_name}_{timestamp}/api_logs/api_calls.jsonl
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import openai
from dotenv import load_dotenv

from src.utils.output_manager import output_manager
from src.utils.logger import logger
from src.utils.config import get_config

# 加载环境变量
load_dotenv('.env.development')
load_dotenv('.env')


OPENAI_COMPATIBLE_PROVIDERS = {
    "openai": {
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
        "mutator_model": "gpt-5.4-mini",
        "persona_model": "gpt-4o-mini",
        "simulator_model": "gpt-4o-mini",
    },
    "deepseek": {
        "api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "mutator_model": "deepseek-v4-flash",
        "persona_model": "deepseek-v4-flash",
        "simulator_model": "deepseek-v4-flash",
    },
}


class APIRecorder:
    """记录 API 调用（单例模式）."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self._log_file = None
        self._log_path = None
    
    def setup(self, run_name: str):
        """设置记录文件路径（通过 OutputManager）."""
        # 确保 OutputManager 已初始化
        if output_manager.base_dir is None:
            output_manager.setup(run_name)
        
        self._log_path = output_manager.get_api_log_path()
        self._log_file = open(self._log_path, 'w', encoding='utf-8')
        
        # 写入文件头
        header = {
            "type": "header",
            "run_name": run_name,
            "created_at": datetime.now().isoformat(),
        }
        self._log_file.write(json.dumps(header, ensure_ascii=False) + '\n')
        self._log_file.flush()
        
        logger.info(f"API 记录文件: {self._log_path}")
    
    def record(self, model: str, messages: list, response: str, elapsed: float, tokens: int = 0, error: str = None):
        """记录一次 API 调用."""
        if self._log_file is None:
            return
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "elapsed": elapsed,
            "tokens": tokens,
            "messages": messages,
            "response": response,
        }
        if error:
            record["error"] = error
        
        self._log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._log_file.flush()
    
    def close(self):
        """关闭记录文件."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# 全局 API 记录器实例
api_recorder = APIRecorder()


class LLMClient:
    """统一 LLM 调用封装（纯同步）."""
    
    def __init__(
        self,
        model: str,
        api_key: str = None,
        base_url: str = None,
        api_key_env: str = "OPENAI_API_KEY",
        allow_env_base_url: bool = True,
        timeout_seconds: Optional[float] = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        self.base_url = base_url or (self._env_base_url() if allow_env_base_url else None)
        self.timeout_seconds = _resolve_timeout_seconds(timeout_seconds)
        if not self.api_key:
            raise ValueError(
                "API key 未设置。请将 API Key 写入 .env.development 文件\n"
                "（已加入 .gitignore，不会上传到仓库），"
                f"或设置 {api_key_env} 环境变量，"
                "或传入 api_key 参数。"
            )
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
        self.recorder = APIRecorder()
    
    @classmethod
    def from_config(cls, model_key: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> "LLMClient":
        """从配置文件创建 LLMClient.

        Args:
            model_key: 配置中的 model 键名，如 "llm.persona_model"
            api_key: 可选，覆盖配置
            base_url: 可选，覆盖配置
        """
        cfg = get_config()
        model = cfg.get(model_key)
        if not model:
            raise ValueError(f"配置中未找到 '{model_key}'，请检查 configs/default.yaml")
        base_url = base_url or cfg.get("llm.api_base") or cls._env_base_url()
        api_key = api_key or cfg.get("llm.api_key") or os.getenv("OPENAI_API_KEY")
        timeout_seconds = cfg.get("llm.timeout_seconds")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def from_provider(
        cls,
        provider: str,
        role: str = "persona",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ) -> "LLMClient":
        """Create an OpenAI-compatible client from a named provider preset.

        Args:
            provider: Provider preset name, e.g. ``openai`` or ``deepseek``.
            role: ``mutator``, ``persona``, or ``simulator``. Selects the
                role-specific model from config unless ``model`` is provided.
            model: Optional direct model override.
            api_key: Optional direct API key override.
            base_url: Optional OpenAI-compatible API base URL override.
            api_key_env: Optional environment variable name override.
        """
        if role not in {"mutator", "persona", "simulator"}:
            raise ValueError("role must be 'mutator', 'persona', or 'simulator'")

        cfg = get_config()
        provider_cfg = {
            **OPENAI_COMPATIBLE_PROVIDERS.get(provider, {}),
            **(cfg.get(f"llm.providers.{provider}") or {}),
        }
        if not provider_cfg:
            known = sorted(
                set(OPENAI_COMPATIBLE_PROVIDERS)
                | set((cfg.get("llm.providers") or {}).keys())
            )
            raise ValueError(f"Unknown LLM provider '{provider}'. Known providers: {known}")

        model = model or provider_cfg.get(f"{role}_model")
        if not model:
            raise ValueError(f"Provider '{provider}' does not define a {role}_model")
        key_env = api_key_env or provider_cfg.get("api_key_env") or "OPENAI_API_KEY"
        api_key = api_key or os.getenv(key_env)
        base_url = base_url if base_url is not None else provider_cfg.get("api_base")
        timeout_seconds = provider_cfg.get("timeout_seconds", cfg.get("llm.timeout_seconds"))
        _validate_provider_credentials(
            provider=provider,
            api_key=api_key,
            api_key_env=key_env,
            base_url=base_url,
        )
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env=key_env,
            allow_env_base_url=False,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _env_base_url() -> Optional[str]:
        """兼容 OpenAI SDK 常用命名和旧 README 中的 OPENAI_API_BASE."""
        return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> str:
        """同步调用 LLM 生成文本."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 4000, **kwargs) -> str:
        """发送聊天请求."""
        start = time.time()
        try:
            response = self._create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            elapsed = time.time() - start
            content = _extract_response_content(response)
            tokens = _response_completion_tokens(response)
            
            self.recorder.record(
                model=self.model,
                messages=messages,
                response=content,
                elapsed=elapsed,
                tokens=tokens
            )
            return content
        except Exception as e:
            elapsed = time.time() - start
            self.recorder.record(
                model=self.model,
                messages=messages,
                response="",
                elapsed=elapsed,
                error=str(e)
            )
            raise

    def _create_chat_completion(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ):
        try:
            # GPT-5/o-series prefer max_completion_tokens; some
            # OpenAI-compatible providers still expect max_tokens.
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            if not _looks_like_token_param_error(exc):
                raise
            fallback_kwargs = dict(kwargs)
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **fallback_kwargs,
            )


def get_llm_client(model_name: str) -> LLMClient:
    """根据配置名称创建 LLM 客户端."""
    return LLMClient(model=model_name)


def _looks_like_token_param_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "max_completion_tokens" in text
        and (
            "unsupported" in text
            or "unknown" in text
            or "invalid" in text
            or "unrecognized" in text
            or "extra" in text
        )
    )


def _extract_response_content(response: Any) -> str:
    """Extract text from an OpenAI-compatible chat response.

    Some compatible providers can return a successful HTTP response with an
    empty or malformed ``choices`` payload under high concurrency. Treat that
    as a retryable provider-side empty response instead of leaking a vague
    ``NoneType`` error into the persona pipeline.
    """
    if response is None:
        raise RuntimeError("LLM server returned an empty response: response is None")

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM server returned an empty response: missing choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise RuntimeError("LLM server returned an empty response: missing message")

    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
                if text:
                    parts.append(str(text))
        content = "\n".join(part for part in parts if part)

    if content is None:
        raise RuntimeError("LLM server returned an empty response: missing message content")
    if not isinstance(content, str):
        content = str(content)
    if not content.strip():
        raise RuntimeError("LLM server returned an empty response: blank content")
    return content


def _response_completion_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    try:
        return int(getattr(usage, "completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _resolve_timeout_seconds(value: Optional[float]) -> float:
    env_value = os.getenv("LLM_TIMEOUT_SECONDS")
    raw = env_value if env_value not in {None, ""} else value
    if raw in {None, ""}:
        return 180.0
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LLM timeout must be a positive number of seconds. "
            "Set `llm.timeout_seconds` in configs/default.yaml or "
            "`LLM_TIMEOUT_SECONDS` in the environment."
        ) from exc
    if timeout <= 0:
        raise ValueError("LLM timeout must be greater than 0 seconds.")
    return timeout


def _validate_provider_credentials(
    provider: str,
    api_key: Optional[str],
    api_key_env: str,
    base_url: Optional[str],
) -> None:
    if provider != "deepseek":
        return
    if not api_key:
        raise ValueError(
            "DeepSeek provider requires an API key. Set the environment variable "
            f"`{api_key_env}` or pass `api_key` explicitly."
        )
    if not base_url:
        raise ValueError(
            "DeepSeek provider requires an explicit base URL. Configure "
            "`llm.providers.deepseek.api_base` in `configs/default.yaml`, or pass "
            "`--persona-api-base` / `--simulator-api-base`."
        )
