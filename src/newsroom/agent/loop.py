"""手写的 agent 循环（不用任何框架）。

核心就是一个 while 循环：
    调模型 → 模型要调工具吗？
        不要 → 它说的就是最终答案，结束
        要   → 执行工具，把结果追加进对话，再调模型

生产里真正难的是循环外面这一圈「护栏」，也是面试追问最多的地方：
- 步数上限：防止 ReAct 死循环（max_steps）；
- 重复调用检测：同一个工具、同样参数反复调，结果不会变，直接拦下并提醒模型换思路；
- 错误回传：工具报错、参数不对，都作为工具结果发回去，让模型自己修正，而不是让程序崩掉；
- token 预算：超预算就收尾，成本可控；
- 并行工具调用：模型一次要调多个工具时并发执行，结果按原顺序一次性发回；
- 收尾：到上限后再给模型一次「不带工具」的机会，让它基于已有信息给出答案。
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..gateway.gateway import ModelGateway
from ..gateway.types import ChatRequest, Message, ToolCall, Usage
from ..tools.registry import ToolRegistry, ToolResult

WRAP_UP_PROMPT = "已经到了步数或预算上限。不要再调用工具，根据目前掌握的信息直接给出最终答案。"


@dataclass
class LoopResult:
    final_text: str
    messages: list[Message]
    steps: int
    usage: Usage
    stopped_reason: str  # final | max_steps | budget | repeat_limit
    trace: list[dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    def __init__(self, gateway: ModelGateway, tools: ToolRegistry | None, *, task: str,
                 max_steps: int = 8, max_repeat_calls: int = 2, token_budget: int = 60000,
                 max_parallel: int = 4, max_tokens: int = 4000):
        self.gateway, self.tools, self.task = gateway, tools, task
        self.max_steps, self.max_repeat_calls = max_steps, max_repeat_calls
        self.token_budget, self.max_parallel, self.max_tokens = token_budget, max_parallel, max_tokens

    def run(self, messages: list[Message]) -> LoopResult:
        msgs = list(messages)
        usage = Usage()
        trace: list[dict[str, Any]] = []
        call_counts: Counter[str] = Counter()
        blocked_streak = 0
        specs = self.tools.specs() if self.tools else []
        stopped = "max_steps"

        for step in range(1, self.max_steps + 1):
            if usage.total >= self.token_budget:
                stopped = "budget"
                break
            # 传一份拷贝：msgs 后面还会追加，别让记账、日志里存下的请求跟着变
            resp = self.gateway.chat(ChatRequest(messages=list(msgs), tools=specs, task=self.task,
                                                 max_tokens=self.max_tokens))
            usage = usage + resp.usage
            msgs.append(resp.message)
            trace.append({"step": step, "type": "llm", "tokens": resp.usage.total,
                          "tool_calls": [c.name for c in resp.message.tool_calls],
                          "latency_ms": round(resp.latency_ms, 1)})

            if not resp.message.tool_calls:
                return LoopResult(resp.message.content, msgs, step, usage, "final", trace)

            results, n_blocked = self._execute_all(resp.message.tool_calls, call_counts)
            for call, res in zip(resp.message.tool_calls, results):
                msgs.append(Message(role="tool", content=res.content, tool_call_id=call.id,
                                    name=call.name, is_error=res.is_error))
                trace.append({"step": step, "type": "tool", "name": call.name,
                              "args": call.arguments, "is_error": res.is_error,
                              "result_chars": len(res.content)})

            # 连续两轮所有调用都是被拦下的重复调用 → 模型卡住了，强制收尾
            blocked_streak = blocked_streak + 1 if n_blocked == len(results) else 0
            if blocked_streak >= 2:
                stopped = "repeat_limit"
                break

        # 收尾：不给工具，逼模型基于现有信息作答
        msgs.append(Message(role="user", content=WRAP_UP_PROMPT))
        resp = self.gateway.chat(ChatRequest(messages=list(msgs), tools=[], task=self.task,
                                             max_tokens=self.max_tokens))
        usage = usage + resp.usage
        msgs.append(resp.message)
        trace.append({"step": "wrap_up", "type": "llm", "reason": stopped, "tokens": resp.usage.total})
        return LoopResult(resp.message.content, msgs, self.max_steps, usage, stopped, trace)

    def _execute_all(self, calls: list[ToolCall], counts: Counter[str]) -> tuple[list[ToolResult], int]:
        planned: list[ToolResult | None] = []
        blocked = 0
        for call in calls:
            key = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
            counts[key] += 1
            if counts[key] > self.max_repeat_calls:
                blocked += 1
                planned.append(ToolResult(
                    f"已拦截：你已经用同样的参数调用 {call.name} {counts[key] - 1} 次了，结果不会变。"
                    "请换个思路，或者直接给出最终答案。", is_error=True))
            else:
                planned.append(None)

        todo = [(i, c) for i, c in enumerate(calls) if planned[i] is None]
        if self.tools and todo:
            with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(todo))) as pool:
                for (i, _), res in zip(todo, pool.map(lambda ic: self.tools.execute(ic[1]), todo)):
                    planned[i] = res
        elif todo:
            for i, c in todo:
                planned[i] = ToolResult(f"错误：当前没有可用的工具（{c.name}）", True)
        return [r for r in planned if r is not None], blocked
