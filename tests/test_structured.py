import pytest

from newsroom.agent.structured import StructuredOutputError, ask_structured, extract_json, parse_as
from newsroom.agents.schemas import JudgeScore
from newsroom.gateway.mock import ScriptedProvider
from newsroom.gateway.types import Message


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('好的，结果如下：\n```json\n{"score": 4, "reason": "ok"}\n```\n希望有帮助') == \
        {"score": 4, "reason": "ok"}
    assert extract_json('前面有话 {"a": {"b": 1}} 后面也有话') == {"a": {"b": 1}}


def test_extract_json_braces_and_quotes_inside_strings():
    text = '{"reason": "原文里写的是 \\"{32K}\\"，不是 }", "score": 2}'
    assert extract_json(text) == {"reason": '原文里写的是 "{32K}"，不是 }', "score": 2}


def test_extract_json_errors():
    with pytest.raises(ValueError):
        extract_json("没有 JSON")
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_parse_as_reports_field_errors():
    value, err = parse_as('{"score": 9, "reason": "太好了"}', JudgeScore)
    assert value is None and "score" in err


def test_repair_loop_fixes_bad_output(gateway):
    scripted = ScriptedProvider([Message("assistant", '{"score": 9, "reason": "x"}'),
                                 Message("assistant", '{"score": 4, "reason": "修好了"}')])
    gateway.register("mock", scripted)
    res = ask_structured(gateway, "评审", {"draft": {}}, JudgeScore, task="judge")
    assert res.value.score == 4 and res.attempts == 2
    # 第二次请求里带着第一次的错误输出和校验错误
    repair = scripted.calls[1].messages[-1].content
    assert "score" in repair and "修正" in repair
    assert scripted.calls[0].json_mode is True


def test_repair_gives_up(gateway):
    gateway.register("mock", ScriptedProvider([Message("assistant", "不是 JSON")]))
    with pytest.raises(StructuredOutputError):
        ask_structured(gateway, "评审", {}, JudgeScore, task="judge", max_repairs=2)
