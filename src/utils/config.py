"""配置加载器 - 统一管理所有模块的 model 和参数."""

from pathlib import Path
from typing import Any, Optional
import yaml


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class Config:
    """配置管理器，支持点号访问嵌套配置."""

    _instance = None
    _config_data: dict = {}

    def __new__(cls, config_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path or CONFIG_PATH)
        return cls._instance

    def _load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._config_data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """通过点号路径获取配置值，如 'llm.persona_model'."""
        keys = key.split(".")
        value = self._config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def raw(self) -> dict:
        return self._config_data


# 全局配置单例
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """获取全局配置单例."""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None):
    """重新加载配置."""
    global _config
    _config = Config(config_path)
    return _config
