"""agent 循环的护栏：步数上限、重复调用拦截、错误回传、并行工具调用、token 预算。"""

import threading
import time

from pydantic import BaseModel

from newsroom.agent.loop import AgentLoop
from newsroom.gateway.mock import ScriptedProvider
from newsroom.gateway.types import Message, ToolCall
from newsroom.tools.registry import Tool, ToolRegistry


class Q(BaseModel):
    q: str


def registry(log=None, sleep=0.0):
    def slow(a: Q):
        if log is not None:
            log.append(threading.current_thread().name)
        time.sleep(sleep)
        return {"echo": a.q}

    def boom(a: Q):
        raise RuntimeError("网站挂了")
    return ToolRegistry([Tool("echo", "回显", Q, slow), Tool("boom", "总是出错", Q, boom)])


def call(name, **args):
    return ToolCall("x", name, args)


def make(gateway, replies, **kw):
    gateway.register("mock", ScriptedProvider(replies))
    return AgentLoop(gateway, registry(), task="t", **kw)


def test_final_answer_without_tools(gateway):
    r = make(gateway, [Message("assistant", "答案")]).run([Message("user", "q")])
    assert r.final_text == "答案" and r.stopped_reason == "final" and r.steps == 1


def test_tool_then_answer(gateway):
    loop = make(gateway, [Message("assistant", "", tool_calls=[call("echo", q="hi")]),
                          Message("assistant", "完成")])
    r = loop.run([Message("user", "q")])
    tool_msgs = [m for m in r.messages if m.role == "tool"]
    assert r.final_text == "完成" and tool_msgs[0].content == '{"echo": "hi"}'


def test_errors_are_fed_back_not_raised(gateway):
    bad = ToolCall("x", "echo", {}, raw_arguments='{"q": ', parse_error="坏 JSON")
    loop = make(gateway, [Message("assistant", "", tool_calls=[
        call("nope", q="1"), call("echo", wrong="1"), call("boom", q="1"), bad]),
        Message("assistant", "换个办法")])
    r = loop.run([Message("user", "q")])
    errors = [m for m in r.messages if m.role == "tool"]
    assert all(m.is_error for m in errors)
    assert "没有叫 nope" in errors[0].content
    assert "参数校验没通过" in errors[1].content
    assert "网站挂了" in errors[2].content
    assert "不是合法的 JSON" in errors[3].content
    assert r.final_text == "换个办法"


def test_repeated_calls_are_blocked(gateway):
    # 模型一直用同样的参数调同一个工具
    loop = make(gateway, [Message("assistant", "", tool_calls=[call("echo", q="same")])],
                max_steps=10, max_repeat_calls=2)
    r = loop.run([Message("user", "q")])
    assert r.stopped_reason == "repeat_limit"
    blocked = [m for m in r.messages if m.role == "tool" and "已拦截" in m.content]
    assert len(blocked) == 2
    assert r.trace[-1]["step"] == "wrap_up"


def test_max_steps_then_wrap_up(gateway):
    replies = [Message("assistant", "", tool_calls=[call("echo", q=str(i))]) for i in range(3)]
    loop = make(gateway, replies + [Message("assistant", "收尾答案")], max_steps=3)
    r = loop.run([Message("user", "q")])
    assert r.stopped_reason == "max_steps"
    assert r.messages[-2].role == "user" and r.messages[-2].content.startswith("已经到了")
    assert r.final_text == "收尾答案"


def test_wrap_up_request_has_no_tools(gateway):
    scripted = ScriptedProvider([Message("assistant", "", tool_calls=[call("echo", q="a")]),
                                 Message("assistant", "", tool_calls=[call("echo", q="b")]),
                                 Message("assistant", "最终")])
    gateway.register("mock", scripted)
    r = AgentLoop(gateway, registry(), task="t", max_steps=2).run([Message("user", "q")])
    assert scripted.calls[-1].tools == []
    assert r.final_text == "最终"


def test_token_budget(gateway):
    # ScriptedProvider 每次报 120 token，预算 200：第二轮之后就超了
    replies = [Message("assistant", "", tool_calls=[call("echo", q=str(i))]) for i in range(5)]
    r = make(gateway, replies, max_steps=10, token_budget=200).run([Message("user", "q")])
    assert r.stopped_reason == "budget"


def test_parallel_tool_calls_keep_order(gateway):
    log = []
    gateway.register("mock", ScriptedProvider([
        Message("assistant", "", tool_calls=[call("echo", q=str(i)) for i in range(4)]),
        Message("assistant", "完成")]))
    loop = AgentLoop(gateway, registry(log, sleep=0.2), task="t", max_parallel=4)
    t0 = time.perf_counter()
    r = loop.run([Message("user", "q")])
    elapsed = time.perf_counter() - t0
    results = [m.content for m in r.messages if m.role == "tool"]
    assert results == [f'{{"echo": "{i}"}}' for i in range(4)]  # 结果按调用顺序回传
    assert elapsed < 0.6  # 串行要 0.8 秒
    assert len(set(log)) > 1  # 确实在多个线程里跑
