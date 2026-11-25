import json

from newsroom.agents.schemas import Claim, ClaimVerdict, Draft, FactCheckReport
from newsroom.evals.runner import evaluate, load_cases, to_markdown
from newsroom.evals.scorers import score_factcheck, score_grounding, score_sources, score_style
from newsroom.gateway.gateway import ModelGateway
from newsroom.gateway.mock import HANDLERS, MockProvider
from newsroom.gateway.types import Message

SRC = "https://www.example-lab.cn/blog/example-7b"


def test_golden_cases_load():
    cases = load_cases()
    assert len(cases) >= 8
    assert {c["type"] for c in cases} == {"write", "factcheck"}


def test_mock_passes_golden_set(settings):
    report = evaluate(settings, "mock")
    assert report.avg >= 0.9
    assert report.judge_is_same_model  # 报告里会提醒「评审和被评是同一个模型」
    assert "评审和被评是同一个模型" in to_markdown([report])


def test_eval_catches_a_writer_that_makes_things_up(settings):
    def hallucinating_writer(req):
        good = json.loads(HANDLERS["writer"](req).content)
        good["claims"].append({"text": "这个模型在所有评测上都超过了 GPT 系列", "source_url": "https://made-up.example.com"})
        good["title"] = "震惊！" + good["title"]
        return Message("assistant", json.dumps(good, ensure_ascii=False))
    gw = ModelGateway(settings, model_override="mock", sleep=lambda _: None)
    gw.register("mock", MockProvider(handlers={"writer": hallucinating_writer}))
    bad = evaluate(settings, "mock", gateway=gw)
    good = evaluate(settings, "mock")
    assert bad.avg < good.avg - 0.1
    assert bad.by_scorer["grounding"] < 1 and bad.by_scorer["sources"] < 1 and bad.by_scorer["style"] < 1


def test_scorers():
    draft = Draft(title="t", body="b", claims=[
        Claim(text="Example-7B 支持 32K 上下文长度", source_url=SRC),
        Claim(text="它是世界第一", source_url="https://news.example-media.cn/a")])
    src = "Example-7B 支持 32K 上下文长度，可以在单张显卡上运行。"
    assert score_grounding(draft, src).value == 0.5
    assert score_sources(draft, SRC, ["example-lab.cn"]).value == 0.5
    assert score_style(Draft(title="重磅", body="x", claims=draft.claims[:1]), ["32K"], []).value < 1
    report = FactCheckReport(verdicts=[ClaimVerdict(claim="a", status="supported"),
                                       ClaimVerdict(claim="b", status="supported")])
    s = score_factcheck(report, [{"text": "a", "expect": "supported"}, {"text": "b", "expect": "unsupported"}])
    assert s.value == 0.5 and "漏掉了错误" in s.detail
