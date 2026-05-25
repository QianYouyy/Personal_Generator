"""LLM 客户端 — 统一 API 调用 + 通信记录.

支持两种创建方式:
  1. 从配置文件创建: LLMClient.from_config("llm.qgenerator_model")
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
    
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError(
                "API key 未设置。请将 API Key 写入 .env.development 文件\n"
                "（已加入 .gitignore，不会上传到仓库），"
                "或设置 OPENAI_API_KEY 环境变量，"
                "或传入 api_key 参数。"
            )
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.recorder = APIRecorder()
    
    @classmethod
    def from_config(cls, model_key: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> "LLMClient":
        """从配置文件创建 LLMClient.

        Args:
            model_key: 配置中的 model 键名，如 "llm.qgenerator_model"
            api_key: 可选，覆盖配置
            base_url: 可选，覆盖配置
        """
        cfg = get_config()
        model = cfg.get(model_key)
        if not model:
            raise ValueError(f"配置中未找到 '{model_key}'，请检查 configs/default.yaml")
        base_url = base_url or cfg.get("llm.api_base") or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        return cls(model=model, api_key=api_key, base_url=base_url)
    
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
            # 统一使用 max_completion_tokens（兼容 o1/o3/gpt-5 系列）
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            elapsed = time.time() - start
            content = response.choices[0].message.content
            tokens = response.usage.completion_tokens if response.usage else 0
            
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


def get_llm_client(model_name: str) -> LLMClient:
    """根据配置名称创建 LLM 客户端."""
    return LLMClient(model=model_name)
