"""网关：路由、重试、记账，以及两种接口格式的翻译。用假的 SDK client，不联网。"""

from types import SimpleNamespace as NS

import pytest

from newsroom.gateway.anthropic_provider import AnthropicProvider
from newsroom.gateway.base import ConfigError, RefusalError, RetryableError
from newsroom.gateway.gateway import ModelGateway, cost_of
from newsroom.gateway.openai_compat import OpenAICompatProvider
from newsroom.gateway.types import ChatRequest, ChatResponse, Message, ToolCall, ToolSpec, Usage


class Flaky:
    """前两次抛限流错误，第三次成功。"""

    name = "flaky"

    def __init__(self, error=RetryableError("429")):
        self.n, self.error = 0, error

    def chat(self, req):
        self.n += 1
        if self.n < 3:
            raise self.error
        return ChatResponse(Message("assistant", "ok"), Usage(1000, 500), "m", "flaky")


def test_routing_by_task(settings):
    settings.routing["judge"] = "qwen"
    gw = ModelGateway(settings, model_override="deepseek")
    assert gw.model_name_for("judge") == "qwen"  # 评审固定用另一个模型
    assert gw.model_name_for("writer") == "deepseek"  # 其他环节跟随 --model
    assert ModelGateway(settings).model_name_for("writer") == "mock"


def test_retry_then_success_and_ledger(settings):
    sleeps, ledger = [], []
    gw = ModelGateway(settings, sleep=sleeps.append, ledger=lambda *a: ledger.append(a))
    flaky = Flaky()
    gw.register("deepseek", flaky)
    resp = gw.chat(ChatRequest([Message("user", "hi")]), model="deepseek")
    assert resp.message.content == "ok" and flaky.n == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] - 1  # 指数退避
    assert resp.cost == pytest.approx(cost_of(settings.models["deepseek"], Usage(1000, 500)))
    assert len(ledger) == 1 and ledger[0][1] is resp


def test_no_retry_on_config_error(settings):
    gw = ModelGateway(settings, sleep=lambda _: pytest.fail("不应该重试"))
    flaky = Flaky(ConfigError("no key"))
    gw.register("deepseek", flaky)
    with pytest.raises(ConfigError):
        gw.chat(ChatRequest([Message("user", "hi")]), model="deepseek")
    assert flaky.n == 1


def test_gives_up_after_max_retries(settings):
    gw = ModelGateway(settings, sleep=lambda _: None)

    class Always429:
        name = "x"
        calls = 0

        def chat(self, req):
            Always429.calls += 1
            raise RetryableError("429")
    gw.register("deepseek", Always429())
    with pytest.raises(RetryableError):
        gw.chat(ChatRequest([Message("user", "hi")]), model="deepseek")
    assert Always429.calls == settings.agent.max_retries + 1


def test_cost_with_cache(settings):
    cfg = settings.models["deepseek"]
    full = cost_of(cfg, Usage(1_000_000, 0, 0))
    assert full == pytest.approx(cfg.price_input_per_m)
    cfg2 = type(cfg)(**{**cfg.__dict__, "price_cached_input_per_m": 0.03})
    assert cost_of(cfg2, Usage(1_000_000, 0, 1_000_000)) == pytest.approx(0.03)


def test_missing_key_is_config_error(settings):
    gw = ModelGateway(settings)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        gw.chat(ChatRequest([Message("user", "hi")]), model="deepseek")


# ---------------- OpenAI 兼容格式（DeepSeek / 通义千问） ----------------

class FakeCompletions:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_openai(message, usage=None):
    comp = FakeCompletions(NS(choices=[NS(message=message, finish_reason="stop")],
                              usage=usage or NS(prompt_tokens=10, completion_tokens=5,
                                                prompt_cache_hit_tokens=4)))
    return NS(chat=NS(completions=comp)), comp


def test_openai_compat_parses_tool_calls_and_bad_json(settings):
    msg = NS(content=None, reasoning_content="先想一想", tool_calls=[
        NS(id="c1", function=NS(name="search_items", arguments='{"query": "agent"}')),
        NS(id="c2", function=NS(name="fetch_url", arguments='{"url": ')),  # 坏 JSON
    ])
    client, comp = fake_openai(msg)
    p = OpenAICompatProvider(settings.models["deepseek"], client=client)
    tools = [ToolSpec("search_items", "检索", {"type": "object", "properties": {}})]
    resp = p.chat(ChatRequest([Message("user", "找新闻")], tools=tools, json_mode=True))
    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls[0].arguments == {"query": "agent"}
    assert resp.message.tool_calls[1].parse_error  # 不崩，留给 agent 循环回传错误
    assert resp.usage.cached_input_tokens == 4
    assert "response_format" not in comp.kwargs  # 带工具时不开 JSON 模式
    assert comp.kwargs["tools"][0]["function"]["name"] == "search_items"
    assert resp.message.provider_raw == {"reasoning_content": "先想一想"}


def test_openai_compat_wire_format_round_trip(settings):
    client, comp = fake_openai(NS(content='{"a": 1}', tool_calls=None))
    p = OpenAICompatProvider(settings.models["deepseek"], client=client)
    history = [
        Message("system", "你是研究员"),
        Message("user", "开始"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "search_items", {"query": ""})],
                provider_raw={"reasoning_content": "想法"}),
        Message("tool", '[{"id": "1"}]', tool_call_id="c1", name="search_items"),
    ]
    p.chat(ChatRequest(history, json_mode=True))
    wire = comp.kwargs["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool"]
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"query": ""}'
    assert wire[2]["reasoning_content"] == "想法"  # DeepSeek 思考模式要求带回
    assert wire[3]["tool_call_id"] == "c1"
    assert comp.kwargs["response_format"] == {"type": "json_object"}


# ---------------- Anthropic 格式（Claude 对照） ----------------

class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def claude_cfg(settings):
    cfg = settings.models["claude"]
    return type(cfg)(**{**cfg.__dict__, "model": "claude-test-model"})


def test_anthropic_wire_merges_tool_results(settings):
    resp = NS(stop_reason="end_turn", content=[NS(type="text", text="好")],
              usage=NS(input_tokens=10, output_tokens=2, cache_read_input_tokens=0))
    fake = FakeMessages(resp)
    p = AnthropicProvider(claude_cfg(settings), client=NS(messages=fake))
    history = [
        Message("system", "你是研究员"),
        Message("user", "开始"),
        Message("assistant", "", tool_calls=[ToolCall("t1", "a", {}), ToolCall("t2", "b", {})]),
        Message("tool", "结果1", tool_call_id="t1"),
        Message("tool", "出错了", tool_call_id="t2", is_error=True),
        Message("user", "已经到了步数上限，直接回答"),
    ]
    p.chat(ChatRequest(history, temperature=0.2))
    k = fake.kwargs
    assert k["system"] == "你是研究员"
    assert "temperature" not in k
    msgs = k["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    last = msgs[-1]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]  # 合并成一条，结果在前
    assert last[1]["is_error"] is True


def test_anthropic_refusal_raises(settings):
    resp = NS(stop_reason="refusal", content=[], usage=NS(input_tokens=1, output_tokens=0))
    p = AnthropicProvider(claude_cfg(settings), client=NS(messages=FakeMessages(resp)))
    with pytest.raises(RefusalError):
        p.chat(ChatRequest([Message("user", "hi")]))


def test_anthropic_empty_model_name_is_config_error(settings):
    p = AnthropicProvider(settings.models["claude"], client=NS(messages=None))
    with pytest.raises(ConfigError, match="CLAUDE_MODEL"):
        p.chat(ChatRequest([Message("user", "hi")]))
