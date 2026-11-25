"""评测：用同一套「金标准」样例跑不同模型，比分数、成本、耗时。

方案文档里的说法是：「每次换提示词或换模型，先跑评测再上线」。这就是那套评测。
- 分环节评：写手 + 改写一组，事实核查员单独一组（故意埋错的草稿，看它抓不抓得出来）；
- 同一个评审模型给所有被评模型打分，换模型时分数才可比；
- 结果写进 out/evals/，CI 里用 --min-score 卡住，分数掉了就不让合并。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents.adapter import adapt
from ..agents.fact_checker import check_draft
from ..agents.schemas import Claim, Draft, ResearchPick
from ..agents.writer import DEFAULT_VOICE, write_draft
from ..collect.sources import make_item
from ..config import Settings
from ..gateway.gateway import ModelGateway
from ..gateway.types import ChatRequest, ChatResponse
from ..tools.fetcher import FetchedPage
from ..tools.safety import scan_injection, source_tier
from .scorers import (Score, judge, score_factcheck, score_format, score_grounding, score_sources,
                      score_style, weighted_total)

GOLDEN = Path(__file__).resolve().parents[3] / "evals" / "golden.jsonl"


@dataclass
class CaseResult:
    case_id: str
    kind: str
    total: float
    scores: list[Score] = field(default_factory=list)
    error: str = ""


@dataclass
class ModelReport:
    model: str
    judge_model: str
    avg: float
    by_scorer: dict[str, float]
    cases: list[CaseResult]
    calls: int
    tokens: int
    cost: dict[str, float]
    p95_latency_ms: float
    judge_is_same_model: bool


def load_cases(path: Path = GOLDEN) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("//")]


def _page(url: str, text: str, settings: Settings) -> FetchedPage:
    return FetchedPage(url=url, text=text, tier=source_tier(url, settings.safety.primary_domains),
                       injection_flags=scan_injection(text))


def run_case(case: dict[str, Any], gateway: ModelGateway, settings: Settings,
             judge_model: str | None, voice: str = DEFAULT_VOICE) -> CaseResult:
    try:
        if case["type"] == "factcheck":
            page = _page(case["source_url"], case["source_text"], settings)
            draft = Draft(title="核查评测", body="（评测用）",
                          claims=[Claim(text=c["text"], source_url=case["source_url"]) for c in case["claims"]])
            report = check_draft(gateway, draft, {page.url: page})
            s = score_factcheck(report, case["claims"])
            return CaseResult(case["id"], "factcheck", s.value, [s])

        it = case["item"]
        item = make_item(it["title"], it["url"], it.get("source", "golden"), settings.safety.primary_domains,
                         published=it.get("published"))
        pick = ResearchPick(item_id=item.id, angle=case.get("angle", "讲清楚这条新闻"), why="评测",
                            primary_source_url=item.url)
        page = _page(item.url, case["source_text"], settings)
        draft = write_draft(gateway, item, pick, page, voice)
        try:
            versions, problems = adapt(gateway, draft)
        except Exception as e:  # 改写失败只扣格式分，不影响其他分
            versions, problems = None, [f"{type(e).__name__}: {e}"]
        scores = [
            score_format(versions, problems),
            score_sources(draft, item.url, settings.safety.primary_domains),
            score_grounding(draft, case["source_text"]),
            score_style(draft, case.get("must_include", []), case.get("must_not_include", [])),
            judge(gateway, draft, case["source_text"], judge_model),
        ]
        return CaseResult(case["id"], "write", weighted_total(scores), scores)
    except Exception as e:  # 一个 case 崩了记 0 分，继续跑下一个
        return CaseResult(case["id"], case.get("type", "?"), 0.0, error=f"{type(e).__name__}: {e}")


def _p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=20, method="inclusive")[-1]


def evaluate(settings: Settings, model: str, judge_model: str | None = None,
             cases: list[dict[str, Any]] | None = None, gateway: ModelGateway | None = None) -> ModelReport:
    cases = cases if cases is not None else load_cases()
    gateway = gateway or ModelGateway(settings, model_override=model)
    if judge_model is None:
        routed = settings.routing.get("judge", "default")
        judge_model = routed if routed != "default" else model

    calls: list[ChatResponse] = []

    def ledger(req: ChatRequest, resp: ChatResponse | None, name: str, err: Exception | None) -> None:
        if resp is not None and req.task != "judge":  # 评审的花费不算在被评模型头上
            calls.append(resp)
    gateway.ledger = ledger

    results = [run_case(c, gateway, settings, judge_model) for c in cases]
    by_scorer: dict[str, list[float]] = {}
    for r in results:
        for s in r.scores:
            by_scorer.setdefault(s.name, []).append(s.value)
    cost: dict[str, float] = {}
    for c in calls:
        cost[c.currency] = cost.get(c.currency, 0) + c.cost
    return ModelReport(
        model=model, judge_model=judge_model,
        avg=round(sum(r.total for r in results) / max(len(results), 1), 3),
        by_scorer={k: round(sum(v) / len(v), 3) for k, v in by_scorer.items()},
        cases=results, calls=len(calls), tokens=sum(c.usage.total for c in calls),
        cost={k: round(v, 6) for k, v in cost.items()},
        p95_latency_ms=round(_p95([c.latency_ms for c in calls]), 1),
        judge_is_same_model=judge_model == model,
    )


def to_markdown(reports: list[ModelReport]) -> str:
    scorers = sorted({k for r in reports for k in r.by_scorer})
    head = "| 模型 | 总分 | " + " | ".join(scorers) + " | 调用 | token | 费用 | p95 耗时 |"
    lines = ["# 评测结果", "", head, "|" + "---|" * (len(scorers) + 6)]
    for r in reports:
        cost = "、".join(f"{v:.4f} {k}" for k, v in r.cost.items()) or "0"
        lines.append(f"| {r.model} | {r.avg:.2f} | " + " | ".join(f"{r.by_scorer.get(s, 0):.2f}" for s in scorers)
                     + f" | {r.calls} | {r.tokens} | {cost} | {r.p95_latency_ms:.0f} ms |")
    lines.append("")
    for r in reports:
        lines.append(f"## {r.model}（评审：{r.judge_model}）")
        if r.judge_is_same_model:
            lines.append("> 评审和被评是同一个模型，judge 分只能参考。")
        for c in r.cases:
            lines.append(f"- {c.case_id}：{c.total:.2f}" + (f"（出错：{c.error}）" if c.error else ""))
            for s in c.scores:
                if s.value < 1:
                    lines.append(f"  - {s.name} {s.value:.2f}：{s.detail}")
        lines.append("")
    return "\n".join(lines)


def save_reports(reports: list[ModelReport], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(
        json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=1), encoding="utf-8")
    md = out_dir / f"eval-{stamp}.md"
    md.write_text(to_markdown(reports), encoding="utf-8")
    return md
