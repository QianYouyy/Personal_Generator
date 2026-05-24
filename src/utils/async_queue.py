"""
异步队列处理器 — 真正的后台请求模式

生产者把请求放入队列，消费者异步处理，
生产者可以立即继续执行其他逻辑，轮询获取结果。
"""

import asyncio
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import time


@dataclass
class LLMRequest:
    """LLM 请求任务."""
    id: int
    prompt: str
    system_prompt: str = None
    temperature: float = 0.7
    max_tokens: int = 512
    result: str = None
    error: str = None
    done: bool = False
    start_time: float = 0
    end_time: float = 0


class BackgroundLLMProcessor:
    """后台 LLM 处理器 — 使用队列实现真正的异步.
    
    用法:
        processor = BackgroundLLMProcessor(llm_client, max_workers=5)
        
        # 提交请求（非阻塞）
        req_id = processor.submit("生成人格描述...")
        
        # 继续做其他事...
        
        # 轮询获取结果
        while not processor.is_done(req_id):
            await asyncio.sleep(0.1)
        result = processor.get_result(req_id)
        
        # 或者批量提交，批量获取
        req_ids = processor.submit_batch([...])
        results = await processor.wait_all(req_ids)
    """

    def __init__(self, llm_client, max_workers: int = 5):
        self.llm = llm_client
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.results: Dict[int, LLMRequest] = {}
        self._counter = 0
        self._workers = []
        self._running = False

    async def start(self):
        """启动后台工作线程."""
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.max_workers)
        ]

    async def stop(self):
        """停止后台工作线程."""
        self._running = False
        # 发送结束信号
        for _ in self._workers:
            await self.queue.put(None)
        # 等待所有工作线程结束
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def _worker(self):
        """工作线程：从队列取请求并处理."""
        while self._running:
            try:
                req = await self.queue.get()
                if req is None:  # 结束信号
                    break
                
                req.start_time = time.time()
                try:
                    resp = await self.llm.generate_async(
                        req.prompt,
                        system_prompt=req.system_prompt,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                    )
                    req.result = resp
                except Exception as e:
                    req.error = str(e)
                
                req.end_time = time.time()
                req.done = True
                self.results[req.id] = req
                
            except Exception:
                pass

    def submit(self, prompt: str, **kwargs) -> int:
        """提交请求到后台队列（非阻塞）.
        
        Returns:
            req_id: 请求ID，用于后续查询结果
        """
        self._counter += 1
        req = LLMRequest(
            id=self._counter,
            prompt=prompt,
            **kwargs
        )
        self.results[req.id] = req
        
        # 使用 create_task 异步放入队列，避免阻塞
        asyncio.create_task(self.queue.put(req))
        
        return req.id

    def submit_batch(self, prompts: List[str], **kwargs) -> List[int]:
        """批量提交请求.
        
        Returns:
            List[int]: 请求ID列表
        """
        return [self.submit(p, **kwargs) for p in prompts]

    def is_done(self, req_id: int) -> bool:
        """检查请求是否完成."""
        req = self.results.get(req_id)
        return req.done if req else False

    def get_result(self, req_id: int) -> str:
        """获取请求结果.
        
        Returns:
            str: 结果文本，如果出错则返回错误信息
        """
        req = self.results.get(req_id)
        if not req:
            return None
        if req.error:
            return f"ERROR: {req.error}"
        return req.result

    def get_elapsed(self, req_id: int) -> float:
        """获取请求耗时."""
        req = self.results.get(req_id)
        if not req or not req.done:
            return 0
        return req.end_time - req.start_time

    async def wait(self, req_id: int, timeout: float = None) -> str:
        """等待单个请求完成.
        
        Args:
            req_id: 请求ID
            timeout: 超时时间（秒）
            
        Returns:
            str: 结果文本
        """
        start = time.time()
        while not self.is_done(req_id):
            if timeout and (time.time() - start) > timeout:
                return "TIMEOUT"
            await asyncio.sleep(0.05)
        return self.get_result(req_id)

    async def wait_all(self, req_ids: List[int], timeout: float = None) -> List[str]:
        """等待所有请求完成.
        
        Args:
            req_ids: 请求ID列表
            timeout: 超时时间（秒）
            
        Returns:
            List[str]: 结果列表
        """
        start = time.time()
        pending = set(req_ids)
        
        while pending:
            done_ids = {rid for rid in pending if self.is_done(rid)}
            pending -= done_ids
            
            if timeout and (time.time() - start) > timeout:
                break
            
            if pending:
                await asyncio.sleep(0.05)
        
        return [self.get_result(rid) for rid in req_ids]

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息."""
        total = len(self.results)
        done = sum(1 for r in self.results.values() if r.done)
        pending = total - done
        
        elapsed_times = [
            r.end_time - r.start_time 
            for r in self.results.values() 
            if r.done
        ]
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
        
        return {
            "total_requests": total,
            "completed": done,
            "pending": pending,
            "avg_request_time": avg_time,
            "queue_size": self.queue.qsize(),
        }
