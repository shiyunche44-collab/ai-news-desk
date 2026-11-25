"""工具注册表：把 Python 函数变成模型能调用的工具。

每个工具 = 名字 + 说明 + 参数的 pydantic 模型 + 执行函数。
参数的 JSON Schema 直接由 pydantic 生成，执行前再用同一个模型校验一遍：
模型给的参数不对，不抛异常，而是把错误「作为工具结果」回传，让模型自己改（错误回传）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ..gateway.types import ToolCall, ToolSpec

MAX_RESULT_CHARS = 8000


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], Any]

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(f"错误：没有叫 {call.name} 的工具。可用的工具：{', '.join(self.tools)}", True)
        if call.parse_error:
            return ToolResult(f"错误：参数不是合法的 JSON（{call.parse_error}）。请重新调用并给出 JSON 对象。", True)
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return ToolResult(f"错误：参数校验没通过（{problems}）。请按工具说明修正参数。", True)
        try:
            result = tool.fn(args)
        except Exception as e:  # 工具自己出错也回传给模型，让它决定换个办法还是放弃
            return ToolResult(f"错误：工具执行失败（{type(e).__name__}: {e}）", True)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n……（结果太长，已截断到 {MAX_RESULT_CHARS} 字）"
        return ToolResult(text)
