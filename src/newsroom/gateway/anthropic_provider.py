"""Claude provider：只在评测里和国产模型做质量、成本对照（中国大陆用不了 Claude）。

和 OpenAI 格式的几个关键差别，面试时可以拿来讲「适配层到底在适配什么」：
1. system 提示是请求的顶层字段，不是一条消息；
2. 工具结果放在 role="user" 的消息里，类型是 tool_result；同一轮的多个结果必须放进同一条消息；
3. 回复是一组「内容块」（text、tool_use、thinking……），要原样回传，所以存在 provider_raw 里；
4. stop_reason 可能是 "refusal"（安全拒答），要先检查再读内容。
"""

from __future__ import annotations

import time
from typing import Any

from ..config import ModelConfig
from .base import ConfigError, ProviderError, RefusalError, RetryableError
from .types import ChatRequest, ChatResponse, Message, ToolCall, Usage

_STOP_MAP = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length",
             "stop_sequence": "stop", "refusal": "refusal"}


class AnthropicProvider:
    def __init__(self, cfg: ModelConfig, client: Any = None):
        self.cfg = cfg
        self.name = cfg.name
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self.cfg.api_key()
            if not key:
                raise ConfigError(f"没有找到环境变量 {self.cfg.api_key_env}")
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ConfigError("没装 anthropic SDK：pip install -e '.[llm]'") from e
            self._client = anthropic.Anthropic(api_key=key, max_retries=0)
        return self._client

    def _to_wire(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        wire: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            # 同一轮的所有工具结果合并成一条 user 消息
            if pending_results:
                wire.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            if m.role == "tool":
                pending_results.append({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                    **({"is_error": True} if m.is_error else {}),
                })
                continue
            if m.role == "user" and pending_results:
                # 工具结果后面紧跟一句用户的话（比如收尾提示）：放进同一条 user 消息，结果块在前
                pending_results.append({"type": "text", "text": m.content})
                flush_results()
                continue
            flush_results()
            if m.role == "assistant":
                if m.provider_raw is not None:
                    # Claude 自己的回复：原样回传（里面可能有 thinking 块，改了会失效）
                    wire.append({"role": "assistant", "content": m.provider_raw})
                else:
                    blocks: list[dict[str, Any]] = []
                    if m.content:
                        blocks.append({"type": "text", "text": m.content})
                    for tc in m.tool_calls:
                        blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                                       "input": tc.arguments})
                    wire.append({"role": "assistant", "content": blocks or m.content})
            else:
                wire.append({"role": "user", "content": m.content})
        flush_results()
        return "\n\n".join(system_parts), wire

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self.cfg.resolved_model()
        if not model:
            raise ConfigError("Claude 对照模型名为空：在 config.toml 填 models.claude.model，或设 CLAUDE_MODEL")
        system, messages = self._to_wire(request.messages)
        kwargs: dict[str, Any] = {
            "model": model,
            # 新一代 Claude 默认会先思考，思考也占输出 token，所以上限给宽一些
            "max_tokens": max(request.max_tokens, 16000),
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
        if request.stop:
            kwargs["stop_sequences"] = request.stop
        # 不传 temperature：新一代 Claude 模型不接受采样参数

        client = self.client  # 先拿 client：没设 key 的 ConfigError 不能被当成接口错误
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(**kwargs)
        except Exception as e:
            raise _classify(e) from e
        latency = (time.perf_counter() - t0) * 1000

        if resp.stop_reason == "refusal":
            raise RefusalError(f"Claude 拒绝了这次请求：{getattr(resp, 'stop_details', None)}")

        texts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        u = resp.usage
        return ChatResponse(
            message=Message(role="assistant", content="".join(texts), tool_calls=calls,
                            provider_raw=resp.content),
            usage=Usage(
                input_tokens=(u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0),
                output_tokens=u.output_tokens or 0,
                cached_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            ),
            model=model,
            provider=self.name,
            finish_reason=_STOP_MAP.get(resp.stop_reason, resp.stop_reason or "stop"),
            latency_ms=latency,
        )


def _classify(e: Exception) -> Exception:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return ProviderError(str(e))
    if isinstance(e, (anthropic.RateLimitError, anthropic.APITimeoutError,
                      anthropic.APIConnectionError, anthropic.InternalServerError)):
        return RetryableError(f"{type(e).__name__}: {e}")
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return ConfigError(f"key 无效或没有权限：{e}")
    return ProviderError(f"{type(e).__name__}: {e}")
