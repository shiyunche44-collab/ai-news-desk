"""编排：每天跑一次的主流程。

    采集（代码） → 选题研究员（agent） → 对每条：写手 → 事实核查 → 平台改写 → 进审核台

第一版的编排是「代码写死顺序」的主从结构：主控是这段代码，四个子 agent 各管一段、上下文互相隔离。
为什么不让一个大 agent 全包？因为顺序是固定的，没必要花 token 让模型决定「下一步该核查了」；
而且出了问题能精确定位是哪一步。第 7 周会用 LangGraph 重写成图，对比两种写法。

最重要的一条：这里没有「发布」。草稿只会进审核台，等你点通过。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents.adapter import adapt
from .agents.fact_checker import check_draft
from .agents.researcher import research
from .agents.writer import DEFAULT_VOICE, write_draft
from .collect.pipeline import collect
from .collect.sources import build_sources
from .config import Settings
from .gateway.gateway import ModelGateway
from .gateway.types import ChatRequest, ChatResponse
from .store import Store, hours_ago
from .tools.fetcher import PageFetcher
from .tools.news_tools import ToolContext


@dataclass
class RunSummary:
    run_id: str
    collected: int
    kept: int
    candidates: int = 0  # 还没写过的候选总数（含之前采集的）
    drafts: list[dict[str, Any]] = field(default_factory=list)
    rejected_picks: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def bind_ledger(gateway: ModelGateway, store: Store, run_id: str | None) -> None:
    """把网关的每次调用记进 SQLite。"""
    def ledger(req: ChatRequest, resp: ChatResponse | None, name: str, err: Exception | None) -> None:
        if resp is not None:
            store.record_call(run_id, req.task, name, resp.model, resp.usage, resp.latency_ms,
                              resp.cost, resp.currency, True)
        else:
            store.record_call(run_id, req.task, name, "", None, 0, 0, "", False, f"{type(err).__name__}: {err}")
    gateway.ledger = ledger


def load_voice(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else DEFAULT_VOICE


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[-1]


def run_daily(settings: Settings, gateway: ModelGateway, store: Store, *, n_picks: int = 3,
              live_fetch: bool = False, window_hours: float = 36) -> RunSummary:
    model_label = gateway.model_override or settings.default_model
    run_id = store.start_run(model_label)
    bind_ledger(gateway, store, run_id)
    trace: list[Any] = []

    # 1. 采集（workflow）
    report = collect(build_sources(settings.sources, settings.safety.primary_domains), store, run_id)
    summary = RunSummary(run_id, report.fetched, len(report.kept), errors=list(report.errors))
    trace.append({"stage": "collect", "fetched": report.fetched, "kept": len(report.kept),
                  "dropped": report.dropped, "errors": report.errors})
    # 候选 = 最近 36 小时采集、还没写过的（包括之前单独跑 collect 存下的）
    since = hours_ago(window_hours)
    summary.candidates = len(store.search_items("", 200, since=since))
    if not summary.candidates:
        summary.errors.append("没有还没写过的候选新闻")
        store.finish_run(run_id, "empty", {}, trace)
        return summary

    fetcher = PageFetcher(settings.safety.fetch_allowlist, settings.safety.primary_domains,
                          settings.safety.fetch_max_chars, settings.root / "data/fixtures/pages",
                          live=live_fetch)
    ctx = ToolContext(store, fetcher, since)

    # 2. 选题（agent）
    outcome = research(gateway, ctx, settings, n=n_picks)
    summary.rejected_picks = outcome.rejected
    trace.append({"stage": "research", "picks": [p.model_dump() for p in outcome.picks],
                  "rejected": outcome.rejected, "loop": outcome.loop.trace if outcome.loop else [],
                  "stopped": outcome.loop.stopped_reason if outcome.loop else None})

    voice = load_voice(settings.voice_profile)
    # 3. 每条：写 → 核查 → 改写（workflow，每步一个隔离上下文的子 agent）
    for pick in outcome.picks:
        item = store.get_item(pick.item_id)
        assert item is not None
        try:
            page = fetcher.fetch(pick.primary_source_url)
            draft = write_draft(gateway, item, pick, page, voice)
            report_fc = check_draft(gateway, draft, {page.url: page})
            versions, problems = adapt(gateway, draft)
        except Exception as e:  # 一条出错不影响其他条
            summary.errors.append(f"{item.title}: {type(e).__name__}: {e}")
            continue

        flags = list(problems)
        bad = [v for v in report_fc.verdicts if v.status != "supported"]
        if bad:
            flags.append(f"事实核查：{len(bad)} 条没通过，审核时重点看")
        if page.injection_flags:
            flags.append(f"原文含可疑指令：{'、'.join(page.injection_flags[:3])}")
        if item.fixture:
            flags.append("示例数据（虚构），不要发布")
        draft_id = store.add_draft(run_id, item.id, draft.model_dump(), report_fc.model_dump(),
                                   versions.model_dump(), flags)
        summary.drafts.append({"id": draft_id, "title": draft.title, "flags": flags,
                               "fact_check_passed": not bad})
        trace.append({"stage": "draft", "item": item.id, "draft": draft_id, "flags": flags})

    # 4. 统计：三个面试要用的数——单篇成本、p95 耗时、token
    costs: dict[str, float] = {}
    tokens = 0
    for row in store.cost_by_task(run_id):
        if not row["currency"]:  # 失败的调用没有币种，也没有费用
            continue
        costs[row["currency"]] = costs.get(row["currency"], 0) + (row["cost"] or 0)
        tokens += (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
    n = max(len(summary.drafts), 1)
    summary.stats = {
        "drafts": len(summary.drafts),
        "tokens": tokens,
        "cost": costs,
        "cost_per_draft": {k: v / n for k, v in costs.items()},
        "p95_latency_ms": round(p95(store.latencies(run_id)), 1),
    }
    store.finish_run(run_id, "ok" if summary.drafts else "no_drafts", summary.stats, trace)
    return summary
