"""Provider 接口和错误分类。

错误要分清「能重试」和「不能重试」：限流（429）、超时、5xx 可以退避重试；
key 错了（401）、参数错了（400）重试一百次也没用，应该立刻报出来。
"""

from __future__ import annotations

from typing import Protocol

from .types import ChatRequest, ChatResponse


class GatewayError(Exception):
    """网关层错误的基类。"""


class RetryableError(GatewayError):
    """限流、超时、连接失败、服务端 5xx：可以退避后重试。"""


class ConfigError(GatewayError):
    """没设 key、模型名为空、key 无效：要人去改配置，重试没用。"""


class ProviderError(GatewayError):
    """其他不可重试的接口错误（比如 400 参数错误）。"""


class RefusalError(GatewayError):
    """模型出于安全原因拒绝回答。"""


class LLMProvider(Protocol):
    name: str

    def chat(self, request: ChatRequest) -> ChatResponse:
        """发一次请求，返回统一格式的回复。不负责重试，重试在网关层统一做。"""
        ...
