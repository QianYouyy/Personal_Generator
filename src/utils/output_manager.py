"""输出管理器 — 统一管理单次任务的所有输出.

目录结构:
data/results/{run_name}_{timestamp}/
  ├── logs/
  │   └── run.log                    # 主日志
  ├── api_logs/
  │   └── api_calls.jsonl            # API 通信记录
  ├── outputs/
  │   └── checkpoint_gen_*.json      # 每轮 checkpoint
  └── visualizations/
      ├── persona_distribution.png
      ├── evolution_curves.png
      └── island_heatmap.png
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class OutputManager:
    """管理单次任务的所有输出目录和文件."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.run_name = None
        self.base_dir = None
        self.logs_dir = None
        self.api_logs_dir = None
        self.outputs_dir = None
        self.viz_dir = None
        
    def setup(self, run_name: str) -> Path:
        """设置输出目录.
        
        Args:
            run_name: 运行名称
            
        Returns:
            Path: 基础输出目录
        """
        self.run_name = run_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 基础目录: data/results/{run_name}_{timestamp}/
        self.base_dir = Path("data/results") / f"{run_name}_{timestamp}"
        
        # 子目录
        self.logs_dir = self.base_dir / "logs"
        self.api_logs_dir = self.base_dir / "api_logs"
        self.outputs_dir = self.base_dir / "outputs"
        self.viz_dir = self.base_dir / "visualizations"
        
        # 创建目录
        for d in [self.logs_dir, self.api_logs_dir, self.outputs_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"[OutputManager] 输出目录: {self.base_dir}")
        return self.base_dir
    
    def get_log_path(self) -> Path:
        """获取日志文件路径."""
        return self.logs_dir / "run.log"
    
    def get_api_log_path(self) -> Path:
        """获取 API 日志路径."""
        return self.api_logs_dir / "api_calls.jsonl"
    
    def get_output_path(self, filename: str) -> Path:
        """获取产出文件路径."""
        return self.outputs_dir / filename
    
    def get_viz_path(self, filename: str) -> Path:
        """获取可视化文件路径."""
        return self.viz_dir / filename
    
    def get_stats(self) -> dict:
        """获取输出统计."""
        return {
            "run_name": self.run_name,
            "base_dir": str(self.base_dir),
            "logs_dir": str(self.logs_dir),
            "api_logs_dir": str(self.api_logs_dir),
            "outputs_dir": str(self.outputs_dir),
            "viz_dir": str(self.viz_dir),
        }


# 全局实例
output_manager = OutputManager()
