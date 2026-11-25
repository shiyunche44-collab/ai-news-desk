"""模型网关：所有模型调用的唯一入口。

它负责四件事，都是面试里讲「生产环境怎么用大模型」的素材：
1. 路由：按环节（task）选模型，实现模型分级；
2. 重试：只对限流、超时、5xx 做指数退避，其他错误立刻抛出；
3. 记账：每次调用的 token、耗时、费用都记下来，按环节汇总就是「单篇成本」；
4. 屏蔽差异：agent 代码只看统一的 ChatRequest / ChatResponse。
"""

from __future__ import annotations

import random
import time
from typing import Callable

from ..config import ModelConfig, Settings
from .base import ConfigError, LLMProvider, RetryableError
from .types import ChatRequest, ChatResponse, Usage

# 记账回调：网关不关心账记在哪（SQLite、日志、Langfuse 都行），只管把每次调用交出去
Ledger = Callable[[ChatRequest, ChatResponse | None, str, Exception | None], None]


def cost_of(cfg: ModelConfig, usage: Usage) -> float:
    cached_price = (cfg.price_cached_input_per_m
                    if cfg.price_cached_input_per_m is not None else cfg.price_input_per_m)
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (uncached * cfg.price_input_per_m
            + usage.cached_input_tokens * cached_price
            + usage.output_tokens * cfg.price_output_per_m) / 1_000_000


def build_provider(cfg: ModelConfig) -> LLMProvider:
    if cfg.provider == "mock":
        from .mock import MockProvider
        return MockProvider(cfg)
    if cfg.provider == "openai_compat":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(cfg)
    if cfg.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(cfg)
    raise ConfigError(f"不认识的 provider：{cfg.provider}")


class ModelGateway:
    def __init__(self, settings: Settings, model_override: str | None = None,
                 ledger: Ledger | None = None, sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.model_override = model_override
        self.ledger = ledger
        self._providers: dict[str, LLMProvider] = {}
        self._sleep = sleep  # 测试里换成不睡觉的函数
        self.total_usage = Usage()

    def provider(self, name: str) -> tuple[ModelConfig, LLMProvider]:
        if name not in self.settings.models:
            raise ConfigError(f"config.toml 里没有模型 {name}，可选：{', '.join(self.settings.models)}")
        cfg = self.settings.models[name]
        if name not in self._providers:
            self._providers[name] = build_provider(cfg)
        return cfg, self._providers[name]

    def register(self, name: str, provider: LLMProvider) -> None:
        """测试或实验时直接塞一个 provider 进来。"""
        self._providers[name] = provider

    def model_name_for(self, task: str) -> str:
        return self.settings.model_for(task, self.model_override)

    def chat(self, request: ChatRequest, model: str | None = None) -> ChatResponse:
        name = model or self.model_name_for(request.task)
        cfg, provider = self.provider(name)
        attempts = self.settings.agent.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = provider.chat(request)
                resp.cost = cost_of(cfg, resp.usage)
                resp.currency = cfg.currency
                self.total_usage = self.total_usage + resp.usage
                if self.ledger:
                    self.ledger(request, resp, name, None)
                return resp
            except RetryableError as e:
                last_error = e
                if attempt + 1 < attempts:
                    # 指数退避 + 随机抖动，避免一堆请求同时重试又一起撞上限流
                    self._sleep(min(2 ** attempt, 20) + random.random())
            except Exception as e:
                if self.ledger:
                    self.ledger(request, None, name, e)
                raise

        if self.ledger:
            self.ledger(request, None, name, last_error)
        assert last_error is not None
        raise last_error
