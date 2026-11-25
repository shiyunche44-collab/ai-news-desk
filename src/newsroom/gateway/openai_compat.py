"""OpenAI 兼容接口的 provider：DeepSeek、通义千问（阿里云百炼的 compatible-mode）都走这里。

同一套代码，只要换 base_url、model、key 就能接不同厂商。
这里手动做了消息格式转换，方便你看清楚 function calling 在接口层长什么样：
- 请求里的 tools 是一组 JSON Schema；
- 模型回复里的 tool_calls 只是「函数名 + JSON 字符串参数」，模型不会自己执行；
- 我们执行完，把结果作为 role="tool" 的消息、带上 tool_call_id 发回去。
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import ModelConfig
from .base import ConfigError, ProviderError, RetryableError
from .types import ChatRequest, ChatResponse, Message, ToolCall, Usage


class OpenAICompatProvider:
    def __init__(self, cfg: ModelConfig, client: Any = None):
        self.cfg = cfg
        self.name = cfg.name
        self._client = client  # 测试时可以注入假的 client

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self.cfg.api_key()
            if not key:
                raise ConfigError(
                    f"没有找到环境变量 {self.cfg.api_key_env}。请在你自己的电脑上 export 它，"
                    "或者先用 --model mock 跑通流程。"
                )
            try:
                import openai
            except ImportError as e:  # pragma: no cover
                raise ConfigError("没装 openai SDK：pip install -e '.[llm]'") from e
            # 重试交给网关统一做，所以这里关掉 SDK 自带的重试
            self._client = openai.OpenAI(
                api_key=key, base_url=self.cfg.base_url, max_retries=0, timeout=90
            )
        return self._client

    # ---------- 格式转换：内部格式 → OpenAI 格式 ----------
    def _to_wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments
                                or json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                # DeepSeek 的思考模式在工具调用的同一轮里要求把 reasoning_content 原样带回去
                if self.cfg.echo_reasoning and isinstance(m.provider_raw, dict):
                    rc = m.provider_raw.get("reasoning_content")
                    if rc:
                        item["reasoning_content"] = rc
                wire.append(item)
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                wire.append({"role": m.role, "content": m.content})
        return wire

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self.cfg.resolved_model()
        if not model:
            raise ConfigError(f"模型 {self.name} 没有配置模型名")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._to_wire_messages(request.messages),
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        elif request.json_mode and self.cfg.json_mode:
            # JSON 模式：保证输出是合法 JSON（DeepSeek 要求提示词里出现 "json" 这个词）
            kwargs["response_format"] = {"type": "json_object"}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop"] = request.stop

        client = self.client  # 先拿 client：没设 key 的 ConfigError 不能被当成接口错误
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # 把 SDK 的异常翻译成网关的三类错误
            raise _classify(e) from e
        latency = (time.perf_counter() - t0) * 1000

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [_parse_tool_call(tc) for tc in (msg.tool_calls or [])]
        raw = {}
        rc = getattr(msg, "reasoning_content", None)
        if rc:
            raw["reasoning_content"] = rc

        return ChatResponse(
            message=Message(
                role="assistant",
                content=msg.content or "",
                tool_calls=tool_calls,
                provider_raw=raw or None,
            ),
            usage=_usage(resp.usage),
            model=model,
            provider=self.name,
            finish_reason="tool_calls" if tool_calls else (choice.finish_reason or "stop"),
            latency_ms=latency,
        )


def _parse_tool_call(tc: Any) -> ToolCall:
    raw = tc.function.arguments or "{}"
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("参数不是 JSON 对象")
        return ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw)
    except (json.JSONDecodeError, ValueError) as e:
        # 模型偶尔会输出坏掉的 JSON。不要崩，留给 agent 循环把错误回传给模型让它重来
        return ToolCall(
            id=tc.id, name=tc.function.name, arguments={}, raw_arguments=raw, parse_error=str(e)
        )


def _usage(u: Any) -> Usage:
    if u is None:
        return Usage()
    cached = getattr(u, "prompt_cache_hit_tokens", None)  # DeepSeek 的字段名
    if cached is None:
        details = getattr(u, "prompt_tokens_details", None)  # OpenAI / 百炼的字段名
        cached = getattr(details, "cached_tokens", 0) if details else 0
    return Usage(
        input_tokens=u.prompt_tokens or 0,
        output_tokens=u.completion_tokens or 0,
        cached_input_tokens=cached or 0,
    )


def _classify(e: Exception) -> Exception:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return ProviderError(str(e))
    if isinstance(e, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                      openai.InternalServerError)):
        return RetryableError(f"{type(e).__name__}: {e}")
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return ConfigError(f"key 无效或没有权限：{e}")
    return ProviderError(f"{type(e).__name__}: {e}")
