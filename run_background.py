"""
后台运行进化任务 + 轮询状态

用法:
  # 启动后台任务
  python run_background.py start --generations 5

  # 查看状态（轮询）
  python run_background.py status

  # 停止任务
  python run_background.py stop

  # 查看日志
  python run_background.py logs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PID_FILE = Path("data/results/.background_pid")
STATUS_FILE = Path("data/results/.background_status")


def start_background(args_list):
    """启动后台进化任务."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        if _is_running(pid):
            print(f"[!] 后台任务已在运行 (PID: {pid})")
            print(f"    使用: python run_background.py status 查看状态")
            return

    # 构建命令
    cmd = [sys.executable, "main.py"] + args_list

    # 启动后台进程
    if sys.platform == "darwin":  # macOS
        # macOS 使用 nohup
        log_file = open("data/results/background.log", "a")
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"后台任务启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"命令: {' '.join(cmd)}\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        # Linux/其他
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    PID_FILE.write_text(str(process.pid))

    # 初始化状态文件
    status = {
        "pid": process.pid,
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "current_generation": 0,
        "total_generations": None,
        "current_eval": 0,
        "best_fitness": {},
        "last_update": datetime.now().isoformat(),
    }
    _write_status(status)

    print(f"[✓] 后台任务已启动 (PID: {process.pid})")
    print(f"    命令: {' '.join(cmd)}")
    print(f"    日志: data/results/background.log")
    print(f"    使用: python run_background.py status 查看状态")


def status_loop():
    """轮询显示状态."""
    if not PID_FILE.exists():
        print("[!] 没有正在运行的后台任务")
        print("    使用: python run_background.py start --generations 5")
        return

    pid = int(PID_FILE.read_text().strip())

    print("=" * 70)
    print("  进化状态监控")
    print("=" * 70)
    print(f"PID: {pid}")
    print(f"按 Ctrl+C 退出监控（不会停止任务）")
    print(f"使用 python run_background.py stop 停止任务")
    print("=" * 70)
    print()

    try:
        while True:
            if not _is_running(pid):
                print("\n[!] 任务已结束")
                _show_final_status()
                break

            _show_current_status()
            time.sleep(5)  # 每5秒刷新

    except KeyboardInterrupt:
        print("\n\n[!] 监控已退出（任务仍在后台运行）")
        print(f"    使用: python run_background.py status 重新连接")
        print(f"    使用: python run_background.py stop 停止任务")


def stop_background():
    """停止后台任务."""
    if not PID_FILE.exists():
        print("[!] 没有正在运行的后台任务")
        return

    pid = int(PID_FILE.read_text().strip())

    if not _is_running(pid):
        print(f"[!] 任务 (PID: {pid}) 已不在运行")
        PID_FILE.unlink(missing_ok=True)
        return

    # 发送终止信号
    try:
        os.kill(pid, 15)  # SIGTERM
        print(f"[✓] 已发送终止信号 (PID: {pid})")

        # 等待进程结束
        for _ in range(10):
            if not _is_running(pid):
                break
            time.sleep(0.5)

        if _is_running(pid):
            os.kill(pid, 9)  # SIGKILL
            print(f"[!] 强制终止 (PID: {pid})")

    except ProcessLookupError:
        print(f"[!] 进程已不存在")

    PID_FILE.unlink(missing_ok=True)
    print("[✓] 后台任务已停止")


def show_logs():
    """显示日志."""
    log_file = Path("data/results/background.log")
    if not log_file.exists():
        print("[!] 暂无日志文件")
        return

    # 显示最后50行
    lines = log_file.read_text().splitlines()
    print("\n".join(lines[-100:]))


def _is_running(pid):
    """检查进程是否在运行."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _write_status(status):
    """写入状态文件."""
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def _read_status():
    """读取状态文件."""
    if not STATUS_FILE.exists():
        return {}
    return json.loads(STATUS_FILE.read_text())


def _show_current_status():
    """显示当前状态."""
    status = _read_status()
    if not status:
        print("  等待状态更新...")
        return

    # 清屏（macOS/Linux）
    print("\033[2J\033[H", end="")

    print("=" * 70)
    print("  🧬 Open-Evolve 进化状态")
    print("=" * 70)

    gen = status.get("current_generation", 0)
    total = status.get("total_generations", "?")
    print(f"  进度: Gen {gen} / {total}")

    # 进度条
    if total and total != "?":
        pct = gen / total * 100
        bar_len = 30
        filled = int(bar_len * gen / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}] {pct:.1f}%")

    print()

    # 最佳适应度
    best = status.get("best_fitness", {})
    if best:
        print("  🏆 当前最优:")
        for k, v in best.items():
            print(f"    {k:20s}: {v:+.6f}")

    print()

    # 时间信息
    start = status.get("start_time")
    if start:
        start_dt = datetime.fromisoformat(start)
        elapsed = datetime.now() - start_dt
        print(f"  ⏱  已运行: {elapsed}")

    last = status.get("last_update")
    if last:
        print(f"  📝 最后更新: {last}")

    print("=" * 70)
    print("  按 Ctrl+C 退出监控 | python run_background.py stop 停止")
    print("=" * 70)


def _show_final_status():
    """显示最终状态."""
    status = _read_status()
    if not status:
        return

    print("\n" + "=" * 70)
    print("  ✅ 任务完成")
    print("=" * 70)

    gen = status.get("current_generation", 0)
    print(f"  完成轮数: {gen}")

    best = status.get("best_fitness", {})
    if best:
        print("\n  最终最优适应度:")
        for k, v in best.items():
            print(f"    {k:20s}: {v:+.6f}")

    start = status.get("start_time")
    if start:
        start_dt = datetime.fromisoformat(start)
        elapsed = datetime.now() - start_dt
        print(f"\n  总运行时间: {elapsed}")

    print("\n  输出目录: data/results/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="后台运行进化任务")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # start 命令
    start_parser = subparsers.add_parser("start", help="启动后台任务")
    start_parser.add_argument("--train", default="data/questionnaires/train.json")
    start_parser.add_argument("--test", default="data/questionnaires/test.json")
    start_parser.add_argument("--generations", type=int, default=5)
    start_parser.add_argument("--hours", type=float, default=None)
    start_parser.add_argument("--eval-model", default="llm.qgenerator_model")

    # status 命令
    subparsers.add_parser("status", help="查看状态（轮询）")

    # stop 命令
    subparsers.add_parser("stop", help="停止后台任务")

    # logs 命令
    subparsers.add_parser("logs", help="查看日志")

    args = parser.parse_args()

    if args.command == "start":
        args_list = []
        if args.train:
            args_list.extend(["--train", args.train])
        if args.test:
            args_list.extend(["--test", args.test])
        if args.generations:
            args_list.extend(["--generations", str(args.generations)])
        if args.hours:
            args_list.extend(["--hours", str(args.hours)])
        if args.eval_model:
            args_list.extend(["--eval-model", args.eval_model])
        start_background(args_list)

    elif args.command == "status":
        status_loop()

    elif args.command == "stop":
        stop_background()

    elif args.command == "logs":
        show_logs()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
