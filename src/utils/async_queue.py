"""异步队列处理器 — 已弃用.

此模块保留用于向后兼容，但不再推荐使用。
项目已统一使用 ThreadPoolExecutor 进行并发，
asyncio 模式已被移除以避免与 ThreadPoolExecutor 嵌套调用时的冲突。

如需并发调用 LLM，请直接使用 concurrent.futures.ThreadPoolExecutor。
"""

# 保留空模块，避免 import 错误
# 所有功能已迁移到 ThreadPoolExecutor 模式
