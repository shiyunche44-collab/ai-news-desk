"""命令行入口：newsroom <命令>。

    newsroom summarize <网址或文件>   第 1 周：一篇文章 → 结构化摘要
    newsroom collect                  只跑采集和去重
    newsroom run                      每天的完整流程：采集 → 选题 → 写 → 核查 → 改写 → 进审核台
    newsroom review list|show|approve|reject
    newsroom eval [--compare deepseek,qwen,claude]
    newsroom react "问题"             手撕 ReAct 演示
    newsroom cost                     看每个环节花了多少钱

所有命令都可以加 --model 换模型（默认 mock，不需要 key）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.react import react
from .collect.pipeline import collect
from .collect.sources import build_sources
from .config import Settings, load_settings
from .evals.runner import evaluate, load_cases, save_reports, to_markdown
from .gateway.base import GatewayError
from .gateway.gateway import ModelGateway
from .orchestrator import run_daily
from .review import approve, format_draft, reject
from .store import Store
from .summarize import load_source, summarize
from .tools.fetcher import FetchBlocked, PageFetcher
from .tools.news_tools import ToolContext, build_news_tools


def _gateway(settings: Settings, args: argparse.Namespace) -> ModelGateway:
    return ModelGateway(settings, model_override=args.model)


def cmd_summarize(settings: Settings, args: argparse.Namespace) -> int:
    text, url, tier, flags = load_source(args.target, settings, live=not args.offline)
    s = summarize(_gateway(settings, args), text, url, tier)
    if flags:
        print(f"⚠ 原文含可疑指令（已当作普通文字处理）：{'、'.join(flags)}\n")
    print(json.dumps(s.model_dump(), ensure_ascii=False, indent=2))
    return 0


def cmd_collect(settings: Settings, args: argparse.Namespace) -> int:
    store = Store(settings.db_path)
    run_id = store.start_run("collect-only")
    report = collect(build_sources(settings.sources, settings.safety.primary_domains), store, run_id)
    store.finish_run(run_id, "collected", {"fetched": report.fetched, "kept": len(report.kept)}, [])
    print(f"抓到 {report.fetched} 条，去重后留下 {len(report.kept)} 条：")
    for it in report.kept:
        print(f"  [{it.tier:9}] {it.title}")
    for title, why in report.dropped:
        print(f"  丢掉（{why}）：{title}")
    for e in report.errors:
        print(f"  信息源出错：{e}")
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    store = Store(settings.db_path)
    gw = _gateway(settings, args)
    summary = run_daily(settings, gw, store, n_picks=args.picks, live_fetch=args.live_fetch)
    print(f"运行 {summary.run_id}：抓到 {summary.collected} 条，去重后新增 {summary.kept} 条，"
          f"待写候选共 {summary.candidates} 条，产出草稿 {len(summary.drafts)} 篇。")
    for d in summary.drafts:
        mark = "✓" if d["fact_check_passed"] else "✗"
        print(f"  {mark} [{d['id']}] {d['title']}")
        for f in d["flags"]:
            print(f"      ⚠ {f}")
    for item_id, why in summary.rejected_picks:
        print(f"  选题被代码护栏拦下：{item_id}（{why}）")
    for e in summary.errors:
        print(f"  出错：{e}")
    st = summary.stats
    if st:
        cost = "、".join(f"{v:.4f} {k}" for k, v in st["cost_per_draft"].items()) or "0"
        print(f"\n单篇成本 {cost}，共 {st['tokens']} token，p95 耗时 {st['p95_latency_ms']} ms")
    print("\n下一步：newsroom review list 看草稿。系统不会自动发布任何内容。")
    return 0 if summary.drafts else 1


def cmd_review(settings: Settings, args: argparse.Namespace) -> int:
    store = Store(settings.db_path)
    if args.action == "list":
        drafts = store.list_drafts(None if args.status == "all" else args.status)
        if not drafts:
            print("没有草稿。先跑 newsroom run。")
        for d in drafts:
            print(format_draft(d, full=False))
        return 0
    if not args.draft_id:
        print("要给草稿 id，比如：newsroom review show 1a2b3c4d")
        return 2
    if args.action == "show":
        d = store.get_draft(args.draft_id)
        if d is None:
            print(f"没有这篇草稿：{args.draft_id}")
            return 1
        print(format_draft(d))
        return 0
    try:
        if args.action == "approve":
            path = approve(store, args.draft_id, settings.out_dir / "approved", args.note, args.force)
            print(f"已通过，导出到 {path}。复制去发就行，系统不会替你发。")
        else:
            reject(store, args.draft_id, args.note)
            print("已打回。")
    except (KeyError, ValueError) as e:
        print(str(e).strip("'\""))
        return 1
    return 0


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases)) if args.cases else load_cases(settings.root / "evals/golden.jsonl")
    models = [m.strip() for m in args.compare.split(",")] if args.compare else [args.model or settings.default_model]
    reports = []
    for m in models:
        print(f"评测 {m}（{len(cases)} 个样例）……", file=sys.stderr)
        try:
            reports.append(evaluate(settings, m, judge_model=args.judge, cases=cases))
        except GatewayError as e:
            print(f"  跳过 {m}：{e}", file=sys.stderr)
    if not reports:
        return 2
    print(to_markdown(reports))
    path = save_reports(reports, settings.out_dir / "evals")
    print(f"\n报告已保存：{path}", file=sys.stderr)
    if args.min_score is not None:
        low = [r for r in reports if r.avg < args.min_score]
        if low:
            print(f"分数低于 {args.min_score}：{', '.join(r.model for r in low)}", file=sys.stderr)
            return 1
    return 0


def cmd_react(settings: Settings, args: argparse.Namespace) -> int:
    store = Store(settings.db_path)
    fetcher = PageFetcher(settings.safety.fetch_allowlist, settings.safety.primary_domains,
                          settings.safety.fetch_max_chars, settings.root / "data/fixtures/pages", live=False)
    registry = build_news_tools(ToolContext(store, fetcher, None))
    result = react(_gateway(settings, args), registry, args.question)
    print(result.transcript)
    print(f"\n结束原因：{result.stopped_reason}，用了 {result.steps} 步")
    if result.answer:
        print(f"最终答案：{result.answer}")
    return 0


def cmd_cost(settings: Settings, args: argparse.Namespace) -> int:
    store = Store(settings.db_path)
    run_id = args.run or store.last_run_id(with_calls=True)
    rows = store.cost_by_task(run_id)
    if not rows:
        print("还没有调用记录。")
        return 0
    print(f"运行 {run_id} 的模型调用：")
    print(f"{'环节':14}{'模型':10}{'调用':>5}{'输入':>9}{'输出':>8}{'缓存':>8}{'费用':>12}{'平均耗时':>10}{'失败':>5}")
    for r in rows:
        print(f"{r['task']:14}{r['model_name']:10}{r['calls']:>5}{r['input_tokens'] or 0:>9}"
              f"{r['output_tokens'] or 0:>8}{r['cached_tokens'] or 0:>8}"
              f"{(r['cost'] or 0):>8.4f} {r['currency'] or '-':3}{(r['avg_latency_ms'] or 0):>9.0f}ms{r['errors']:>5}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="newsroom", description="AI 资讯编辑部（第一版）")
    p.add_argument("--config", help="配置文件路径，默认是项目根目录的 config.toml")
    p.add_argument("--model", help="用哪个模型：mock、deepseek、qwen、claude（见 config.toml）")
    # 让 --model / --config 写在子命令前后都行：newsroom run --model deepseek
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--model", default=argparse.SUPPRESS, help="用哪个模型（见 config.toml）")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    s = add("summarize", help="一个网址或文件 → 结构化摘要")
    s.add_argument("target")
    s.add_argument("--offline", action="store_true", help="不联网，只读示例页面")
    s.set_defaults(fn=cmd_summarize)

    add("collect", help="只跑采集和去重").set_defaults(fn=cmd_collect)

    r = add("run", help="每天的完整流程")
    r.add_argument("--picks", type=int, default=3, help="选几条写（默认 3）")
    r.add_argument("--live-fetch", action="store_true", help="允许读真实网页（默认只读示例页面）")
    r.set_defaults(fn=cmd_run)

    v = add("review", help="审核台")
    v.add_argument("action", choices=["list", "show", "approve", "reject"])
    v.add_argument("draft_id", nargs="?")
    v.add_argument("--status", default="pending", choices=["pending", "approved", "rejected", "all"])
    v.add_argument("--note", default="", help="审核意见")
    v.add_argument("--force", action="store_true", help="有警告也通过")
    v.set_defaults(fn=cmd_review)

    e = add("eval", help="跑评测")
    e.add_argument("--compare", help="逗号分隔的多个模型，比如 deepseek,qwen,claude")
    e.add_argument("--judge", help="评审用哪个模型（建议和被评模型不同）")
    e.add_argument("--cases", help="样例文件，默认 evals/golden.jsonl")
    e.add_argument("--min-score", type=float, help="平均分低于它就返回非 0（给 CI 用）")
    e.set_defaults(fn=cmd_eval)

    a = add("react", help="手撕 ReAct 演示")
    a.add_argument("question")
    a.set_defaults(fn=cmd_react)

    c = add("cost", help="看每个环节的花费")
    c.add_argument("--run", help="运行 id，默认最近一次")
    c.set_defaults(fn=cmd_cost)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    try:
        return args.fn(settings, args)
    except GatewayError as e:
        print(f"模型调用出错：{e}", file=sys.stderr)
        return 2
    except FetchBlocked as e:
        print(f"不能读这个网址：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
