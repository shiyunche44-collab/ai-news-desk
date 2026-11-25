"""结构化输出：让模型按 JSON Schema 输出，再用 pydantic 校验，不合格就回传错误让它改。

三层保障（从弱到强）：
1. 提示词里给出 JSON Schema；
2. 接口支持的话打开 JSON 模式（保证是合法 JSON，但不保证字段对）；
3. 本地用 pydantic 校验，失败就把具体错误发回去，最多修 N 次。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..gateway.gateway import ModelGateway
from ..gateway.types import ChatRequest, Message, Usage

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(Exception):
    pass


def schema_instruction(schema: type[BaseModel]) -> str:
    return ("只输出一个 JSON 对象，不要输出任何其他文字，符合下面的 JSON Schema：\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False))


def task_input(payload: dict[str, Any]) -> str:
    """把任务输入打包成 JSON。结构化的输入比一段长文本更不容易被模型误读。"""
    return "<task_input>\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n</task_input>"


def extract_json(text: str) -> Any:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("输出里没有 JSON 对象")
    # 从第一个 { 开始找配对的 }，容忍模型在 JSON 前后多说了几句话
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:  # 字符串里的括号不算数，还要处理转义的引号
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 对象没有闭合")


def parse_as(text: str, schema: type[T]) -> tuple[T | None, str | None]:
    try:
        return schema.model_validate(extract_json(text)), None
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"不是合法的 JSON：{e}"
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or '(根)'}: {err['msg']}" for err in e.errors())
        return None, f"字段校验没通过：{problems}"


@dataclass
class StructuredResult:
    value: Any
    usage: Usage
    attempts: int
    errors: list[str] = field(default_factory=list)


def repair_messages(messages: list[Message], bad_output: str, error: str) -> list[Message]:
    return messages + [
        Message(role="assistant", content=bad_output),
        Message(role="user", content=f"上面的输出有问题：{error}\n请只输出修正后的 JSON 对象。"),
    ]


def ask_structured(gateway: ModelGateway, system: str, payload: dict[str, Any], schema: type[T],
                   task: str, max_repairs: int = 1, max_tokens: int = 4000,
                   model: str | None = None) -> StructuredResult:
    """一次性调用（不带工具）+ 校验 + 修复。写手、核查、改写都用它。

    model 不填就按 task 路由；评审（judge）会显式指定一个和被评模型不同的模型。"""
    messages = [Message(role="system", content=system + "\n\n" + schema_instruction(schema)),
                Message(role="user", content=task_input(payload))]
    usage, errors = Usage(), []
    for attempt in range(1, max_repairs + 2):
        resp = gateway.chat(ChatRequest(messages=messages, task=task, json_mode=True,
                                        max_tokens=max_tokens), model=model)
        usage = usage + resp.usage
        value, err = parse_as(resp.message.content, schema)
        if value is not None:
            return StructuredResult(value, usage, attempt, errors)
        errors.append(err or "未知错误")
        messages = repair_messages(messages, resp.message.content, err or "")
    raise StructuredOutputError(f"{task}：修了 {max_repairs} 次还是不合格：{errors[-1]}")
