"""日志工具 — 彩色格式化输出 + 文件持久化.

日志文件路径由 OutputManager 统一管理:
data/results/{run_name}_{timestamp}/logs/run.log
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.output_manager import output_manager


class Colors:
    """ANSI 颜色码."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Logger:
    """带时间戳和颜色的日志器，同时写入文件."""

    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "PG"):
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        
        self.name = name
        self.start_time = time.time()
        self._log_file = None
        self._log_path = None

    def setup(self, run_name: str):
        """设置日志文件路径（通过 OutputManager）."""
        # 获取输出目录
        base_dir = output_manager.setup(run_name)
        self._log_path = output_manager.get_log_path()
        self._log_file = open(self._log_path, 'w', encoding='utf-8')
        
        # 写入日志头
        self._write_to_file(f"=" * 70)
        self._write_to_file(f"Persona Generator 运行日志")
        self._write_to_file(f"运行名称: {run_name}")
        self._write_to_file(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_to_file(f"=" * 70)
        self._write_to_file("")
        
        print(f"[Logger] 日志文件: {self._log_path}")

    def _timestamp(self) -> str:
        """返回已运行时间."""
        elapsed = time.time() - self.start_time
        return f"[{elapsed:>7.1f}s]"

    def _prefix(self, level: str, color: str) -> str:
        return f"{color}{self._timestamp()} [{level:>4}] [{self.name}]{Colors.END}"

    def _plain_prefix(self, level: str) -> str:
        """无颜色的前缀（用于文件）."""
        return f"{self._timestamp()} [{level:>4}] [{self.name}]"

    def _write_to_file(self, msg: str):
        """写入日志文件（去除ANSI颜色码）."""
        if self._log_file:
            # 去除ANSI颜色码
            plain = msg
            for code in vars(Colors).values():
                if isinstance(code, str):
                    plain = plain.replace(code, '')
            self._log_file.write(plain + '\n')
            self._log_file.flush()

    def _log(self, level: str, color: str, msg: str):
        """同时输出到控制台和文件."""
        console_msg = f"{self._prefix(level, color)} {msg}"
        file_msg = f"{self._plain_prefix(level)} {msg}"
        print(console_msg)
        self._write_to_file(file_msg)

    def info(self, msg: str):
        self._log('INFO', Colors.CYAN, msg)

    def success(self, msg: str):
        self._log('OK', Colors.GREEN, msg)

    def warn(self, msg: str):
        self._log('WARN', Colors.YELLOW, msg)

    def error(self, msg: str):
        self._log('ERR', Colors.RED, msg)

    def debug(self, msg: str):
        self._log('DBG', Colors.BLUE, msg)

    def section(self, title: str):
        """打印分隔线标题."""
        line = "=" * 70
        console_output = f"\n{Colors.BOLD}{Colors.HEADER}{line}{Colors.END}\n{Colors.BOLD}{Colors.HEADER}  {title}{Colors.END}\n{Colors.BOLD}{Colors.HEADER}{line}{Colors.END}"
        file_output = f"\n{line}\n  {title}\n{line}"
        print(console_output)
        self._write_to_file(file_output)

    def progress(self, current: int, total: int, msg: str = ""):
        """打印进度条."""
        pct = current / total * 100
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        console_msg = f"\r{Colors.CYAN}{self._timestamp()} [PROG] [{self.name}]{Colors.END} [{bar}] {pct:>5.1f}% {msg}"
        file_msg = f"{self._timestamp()} [PROG] [{self.name}] [{bar}] {pct:>5.1f}% {msg}"
        print(console_msg, end="", flush=True)
        self._write_to_file(file_msg)
        if current >= total:
            print()  # 换行

    def metric(self, name: str, value: float, unit: str = ""):
        """打印指标."""
        console_msg = f"{self._prefix('METR', Colors.GREEN)} {name:20s}: {value:+.6f} {unit}"
        file_msg = f"{self._plain_prefix('METR')} {name:20s}: {value:+.6f} {unit}"
        print(console_msg)
        self._write_to_file(file_msg)

    def step(self, step_name: str, step_num: int, total_steps: int):
        """打印步骤标题."""
        console_msg = f"\n{Colors.BOLD}{Colors.YELLOW}[{step_num}/{total_steps}] {step_name}{Colors.END}"
        file_msg = f"\n[{step_num}/{total_steps}] {step_name}"
        print(console_msg)
        self._write_to_file(file_msg)

    def close(self):
        """关闭日志文件."""
        if self._log_file:
            self._write_to_file("")
            self._write_to_file(f"日志结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write_to_file(f"总运行时间: {self._timestamp()}")
            self._log_file.close()
            self._log_file = None

    def __del__(self):
        """析构时确保文件关闭."""
        try:
            self.close()
        except Exception:
            pass


# 全局日志器
logger = Logger("PG")
