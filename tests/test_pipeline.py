"""端到端：用模拟模型跑完整的每日流程，再走一遍审核台。"""

import pytest

from newsroom.collect.pipeline import collect
from newsroom.collect.sources import RSSSource, build_sources
from newsroom.gateway.gateway import ModelGateway
from newsroom.orchestrator import run_daily
from newsroom.review import approve, reject


def test_collect_dedupes_and_remembers_dropped(settings, store):
    sources = build_sources(settings.sources, settings.safety.primary_domains)
    first = collect(sources, store, "r1")
    assert first.fetched == 8 and len(first.kept) == 6
    reasons = {why for _, why in first.dropped}
    assert reasons == {"网址重复", "和今天另一条近似重复"}
    again = collect(sources, store, "r2")  # 第二天同样的数据：全部算采集过
    assert again.kept == [] and all(why == "以前采集过" for _, why in again.dropped)


def test_daily_run_end_to_end(settings, store, gateway):
    s = run_daily(settings, gateway, store, n_picks=3)
    assert len(s.drafts) == 3 and not s.errors
    drafts = store.list_drafts()
    by_title = {d["title"]: d for d in drafts}
    price = next(d for t, d in by_title.items() if "降低 50%" in t)
    # 原文里的注入指令被打标，而且没有进入草稿
    assert any("可疑指令" in f for f in price["flags"])
    for d in drafts:
        blob = d["body"] + str(d["versions"])
        assert "evil.example.com" not in blob and "立即发布" not in blob
        assert any("示例数据" in f for f in d["flags"])
        assert d["status"] == "pending"  # 只进审核台，不会自动发布
    # 只选一手来源；二手的爆料没被选上
    assert not any("爆料" in t or "独家" in t for t in by_title)
    # 每次模型调用都记了账
    tasks = {r["task"] for r in store.cost_by_task(s.run_id)}
    assert tasks == {"researcher", "writer", "fact_checker", "adapter"}
    assert s.stats["tokens"] > 0 and s.stats["p95_latency_ms"] > 0


def test_second_run_writes_what_is_left(settings, store, gateway):
    run_daily(settings, gateway, store, n_picks=3)
    second = run_daily(settings, gateway, store, n_picks=3)
    assert [d["title"] for d in second.drafts] == ["【示例】tiny-agent：200 行 Python 写的 agent 教学框架"]
    third = run_daily(settings, gateway, store, n_picks=3)
    assert third.drafts == []  # 只剩二手来源，宁可不写


def test_one_bad_item_does_not_stop_the_run(settings, store):
    from newsroom.gateway.mock import MockProvider
    from newsroom.gateway.types import Message

    calls = {"n": 0}

    def flaky_writer(req):
        calls["n"] += 1
        if calls["n"] <= 2:  # 第一条：两次都不给合法 JSON
            return Message("assistant", "抱歉我写不出来")
        from newsroom.gateway.mock import HANDLERS
        return HANDLERS["writer"](req)
    gw = ModelGateway(settings, sleep=lambda _: None)
    gw.register("mock", MockProvider(handlers={"writer": flaky_writer}))
    s = run_daily(settings, gw, store, n_picks=3)
    assert len(s.drafts) == 2 and len(s.errors) == 1


def test_review_flow(settings, store, gateway):
    run_daily(settings, gateway, store, n_picks=1)
    d = store.list_drafts()[0]
    with pytest.raises(ValueError, match="示例数据"):
        approve(store, d["id"], settings.out_dir)  # 示例数据默认不让通过
    path = approve(store, d["id"], settings.out_dir, note="测试", force=True)
    text = path.read_text(encoding="utf-8")
    assert "## X" in text and "## 来源" in text
    assert store.get_draft(d["id"])["status"] == "approved"
    reject(store, d["id"], "改主意了")
    assert store.get_draft(d["id"])["status"] == "rejected"
    with pytest.raises(KeyError):
        reject(store, "nope")


def test_rss_and_atom_parsing():
    rss = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Paper A</title><link>https://arxiv.org/abs/0000.00000</link>
      <description>&lt;p&gt;Abstract&lt;/p&gt;</description></item></channel></rss>"""
    atom = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>Release B</title><link href="https://github.com/o/r/releases/1"/>
      <summary>notes</summary><updated>2026-11-24T00:00:00Z</updated></entry></feed>"""
    src = RSSSource("https://x", ["arxiv.org", "github.com"])
    a = src.parse(rss)[0]
    b = src.parse(atom)[0]
    assert a.kind == "paper" and a.tier == "primary" and a.summary == "Abstract"
    assert b.kind == "repo" and b.published == "2026-11-24T00:00:00Z"
