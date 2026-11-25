"""四个子 agent 的代码护栏：事实核查的预检查、平台长度限制、选题研究员的输出校验。"""

import json

from newsroom.agents.adapter import platform_problems, x_weighted_length
from newsroom.agents.fact_checker import check_draft, missing_numbers
from newsroom.agents.researcher import research
from newsroom.agents.schemas import Claim, Draft, PlatformVersions, XiaohongshuNote
from newsroom.collect.pipeline import collect
from newsroom.collect.sources import build_sources
from newsroom.gateway.mock import HANDLERS, MockProvider
from newsroom.gateway.types import Message
from newsroom.tools.fetcher import FetchedPage, PageFetcher
from newsroom.tools.news_tools import ToolContext

SRC = "https://www.example-lab.cn/blog/example-7b"
TEXT = "Example-7B 支持 32K 上下文长度，可以在单张 24GB 显存的显卡上运行。"


def test_x_weighted_length():
    assert x_weighted_length("abc") == 3
    assert x_weighted_length("中文") == 4
    assert x_weighted_length("看 https://www.example-lab.cn/blog/a-very-long-path-that-is-long") == 2 + 1 + 23


def test_platform_problems():
    ok = PlatformVersions(x_post="短帖", xiaohongshu=XiaohongshuNote(title="标题", body="正文"),
                          shipinhao_script="字" * 200)
    assert platform_problems(ok) == []
    bad = PlatformVersions(x_post="字" * 150, xiaohongshu=XiaohongshuNote(title="长" * 21, body=""),
                           shipinhao_script="短")
    assert len(platform_problems(bad)) == 3


def test_missing_numbers():
    assert missing_numbers("支持 128K 上下文", TEXT) == ["128"]
    assert missing_numbers("支持 32K 上下文", TEXT) == []


def test_fact_checker_code_checks_before_llm(gateway):
    page = FetchedPage(SRC, TEXT, "primary")
    media = FetchedPage("https://news.example-media.cn/a", "媒体说 Example-7B 很强", "secondary")
    draft = Draft(title="t", body="b", claims=[
        Claim(text="Example-7B 支持 32K 上下文长度", source_url=SRC),
        Claim(text="Example-7B 支持 128K 上下文长度", source_url=SRC),
        Claim(text="Example-7B 很强", source_url=media.url),
        Claim(text="Example-7B 获得了图灵奖", source_url="https://made-up.example.com/x"),
    ])
    mock = MockProvider()
    gateway.register("mock", mock)
    report = check_draft(gateway, draft, {SRC: page, media.url: media})
    assert [v.status for v in report.verdicts] == ["supported", "unsupported", "secondhand", "unsupported"]
    assert "128" in report.verdicts[1].note
    # 只有第一条需要交给模型判断
    sent = json.loads(mock.calls[0].messages[-1].content.split("<task_input>")[1].split("</task_input>")[0])
    assert [c["index"] for c in sent["claims"]] == [0]


def test_fact_checker_treats_missing_verdict_as_unsupported(gateway):
    gateway.register("mock", MockProvider(handlers={
        "fact_checker": lambda req: Message("assistant", '{"verdicts": [], "summary": "偷懒了"}')}))
    draft = Draft(title="t", body="b", claims=[Claim(text="Example-7B 支持 32K 上下文长度", source_url=SRC)])
    report = check_draft(gateway, draft, {SRC: FetchedPage(SRC, TEXT, "primary")})
    assert report.verdicts[0].status == "unsupported" and not report.passed


def _ctx(settings, store):
    collect(build_sources(settings.sources, settings.safety.primary_domains), store, "r1")
    fetcher = PageFetcher(settings.safety.fetch_allowlist, settings.safety.primary_domains,
                          fixtures_dir=settings.root / "data/fixtures/pages", live=False)
    return ToolContext(store, fetcher)


def test_researcher_uses_tools_in_parallel(gateway, settings, store):
    mock = MockProvider()
    gateway.register("mock", mock)
    out = research(gateway, _ctx(settings, store), settings, n=3)
    assert len(out.picks) == 3
    assert all("example" in p.primary_source_url for p in out.picks)
    fetch_rounds = [c for c in mock.calls if c.messages[-1].role == "tool"]
    assert out.loop is not None and out.loop.stopped_reason == "final"
    # 第二轮一次发出了多个 fetch_url
    fetch_step = [t for t in out.loop.trace if t["type"] == "llm" and "fetch_url" in t["tool_calls"]]
    assert len(fetch_step[0]["tool_calls"]) >= 3
    assert fetch_rounds


def test_researcher_guardrails_reject_bad_picks(gateway, settings, store):
    ctx = _ctx(settings, store)
    real = store.search_items("", 50)
    primary = next(i for i in real if i.tier == "primary")
    rumor = next(i for i in real if i.tier == "secondary")

    def liar(req):
        if req.tools and not any(m.role == "tool" for m in req.messages):
            return HANDLERS["researcher"](req)  # 先老实检索一次
        return Message("assistant", json.dumps({"picks": [
            {"item_id": "doesnotexist", "angle": "a", "why": "b", "primary_source_url": primary.url},
            {"item_id": rumor.id, "angle": "a", "why": "b", "primary_source_url": rumor.url},
            {"item_id": primary.id, "angle": "a", "why": "b", "primary_source_url": primary.url},
            {"item_id": primary.id, "angle": "a", "why": "b", "primary_source_url": primary.url},
        ]}, ensure_ascii=False))
    gateway.register("mock", MockProvider(handlers={"researcher": liar}))
    out = research(gateway, ctx, settings, n=3)
    assert [p.item_id for p in out.picks] == [primary.id]
    reasons = [r for _, r in out.rejected]
    assert any("编出来" in r for r in reasons)
    assert any("不是一手" in r for r in reasons)
    assert any("重复" in r for r in reasons)


def test_research_tools_have_no_write_or_publish():
    from newsroom.tools.news_tools import build_news_tools
    names = set(build_news_tools(ToolContext(None, None)).tools)  # type: ignore[arg-type]
    assert names == {"search_items", "fetch_url", "check_seen"}
