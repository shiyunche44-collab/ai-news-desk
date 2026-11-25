from pydantic import BaseModel

from newsroom.agent.react import react
from newsroom.gateway.mock import ScriptedProvider
from newsroom.gateway.types import Message
from newsroom.tools.registry import Tool, ToolRegistry


class Args(BaseModel):
    city: str


def reg(calls):
    def weather(a: Args):
        calls.append(a.city)
        return {"city": a.city, "weather": "晴"}
    return ToolRegistry([Tool("weather", "查天气", Args, weather)])


def test_react_parses_action_and_final(gateway):
    calls = []
    scripted = ScriptedProvider([
        Message("assistant", 'Thought: 要查天气\nAction: weather\nAction Input: {"city": "台北"}\n'
                             "Observation: 下雨（模型自己编的，应该被截掉）"),
        Message("assistant", "Thought: 我已经知道答案了\nFinal Answer: 台北是晴天"),
    ])
    gateway.register("mock", scripted)
    r = react(gateway, reg(calls), "台北天气怎么样")
    assert r.answer == "台北是晴天" and r.stopped_reason == "final"
    assert calls == ["台北"]
    assert "下雨" not in r.transcript  # 模型编的 Observation 被丢掉，用的是真实工具结果
    assert scripted.calls[0].stop == ["Observation:"]


def test_react_format_error_and_repeat_detection(gateway):
    same = 'Thought: 查\nAction: weather\nAction Input: {"city": "台北"}'
    gateway.register("mock", ScriptedProvider([
        Message("assistant", "我随便说点什么"),  # 格式不对
        Message("assistant", same), Message("assistant", same),  # 重复动作
        Message("assistant", "Final Answer: 晴"),
    ]))
    calls = []
    r = react(gateway, reg(calls), "台北天气")
    assert "格式错误" in r.transcript
    assert "完全一样的动作" in r.transcript
    assert calls == ["台北"] and r.answer == "晴"


def test_react_max_steps(gateway):
    gateway.register("mock", ScriptedProvider([Message("assistant", "Thought: 嗯……")]))
    r = react(gateway, reg([]), "q", max_steps=3)
    assert r.answer is None and r.stopped_reason == "max_steps" and r.steps == 3


def test_react_with_mock_model(gateway, store, settings):
    # 内置模拟模型也会按 ReAct 格式回答
    from newsroom.collect.pipeline import collect
    from newsroom.collect.sources import build_sources
    from newsroom.tools.fetcher import PageFetcher
    from newsroom.tools.news_tools import ToolContext, build_news_tools

    collect(build_sources(settings.sources, settings.safety.primary_domains), store, "r1")
    fetcher = PageFetcher([], [], fixtures_dir=None, live=False)
    r = react(gateway, build_news_tools(ToolContext(store, fetcher)), "今天写哪条？")
    assert r.stopped_reason == "final" and "示例" in r.answer
