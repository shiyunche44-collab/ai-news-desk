"""网关内部统一的数据结构。

各家接口格式不同（OpenAI 格式、Anthropic 格式），agent 代码只认这里的结构，
由各个 provider 负责来回翻译。这是「适配器模式」，也是面试里讲「怎么做到换模型不改代码」的核心。

内部格式选的是 OpenAI 风格（role = system/user/assistant/tool），
因为 DeepSeek、通义千问都兼容它，翻译成本最低。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """模型说「我要调用某个工具」。注意：模型本身不执行函数，它只输出函数名和参数，
    真正执行的是我们的代码（面试高频题：function calling 时模型会直接执行函数吗？不会）。"""

    id: str
    name: str
    arguments: dict[str, Any]
    # 参数 JSON 解析失败时保留原文，让 agent 循环把错误回传给模型
    raw_arguments: str | None = None
    parse_error: str | None = None


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # 仅 assistant
    tool_call_id: str | None = None  # 仅 tool
    name: str | None = None  # 仅 tool：工具名
    is_error: bool = False  # 仅 tool：这次工具调用失败了
    # 有些厂商的回复里带了必须原样回传的内容（比如 Claude 的 thinking 块），存在这里
    provider_raw: Any = None


@dataclass
class ToolSpec:
    """给模型看的工具说明：名字、用途、参数的 JSON Schema。"""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # 命中上下文缓存的输入 token，单价通常便宜很多

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ChatRequest:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    # task 是「哪个环节在调用」，用来分级选模型、按环节记账，mock 也靠它决定怎么回答
    task: str = "default"
    json_mode: bool = False
    max_tokens: int = 4000
    temperature: float | None = None
    stop: list[str] | None = None


@dataclass
class ChatResponse:
    message: Message
    usage: Usage
    model: str
    provider: str
    finish_reason: str = "stop"  # stop | tool_calls | length | refusal
    latency_ms: float = 0.0
    cost: float = 0.0
    currency: str = "USD"
