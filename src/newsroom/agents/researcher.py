"""选题研究员：整个系统里唯一一个真正「自己决定下一步」的 agent。

选题需要判断（哪条重要、要不要再读原文、要不要换个关键词再搜），所以交给 agent 循环；
但它的输出还要过一道代码写的护栏：id 必须是真实候选、来源必须是一手的。
模型说什么不重要，代码验过才算数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent.loop import AgentLoop, LoopResult
from ..agent.structured import parse_as, repair_messages, schema_instruction, task_input
from ..config import Settings
from ..gateway.gateway import ModelGateway
from ..gateway.types import ChatRequest, Message
from ..tools.news_tools import ToolContext, build_news_tools
from ..tools.safety import source_tier
from .prompts import RESEARCHER_SYSTEM
from .schemas import ResearchPick, ResearchResult


@dataclass
class ResearchOutcome:
    picks: list[ResearchPick]
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (item_id, 原因)
    loop: LoopResult | None = None


def research(gateway: ModelGateway, ctx: ToolContext, settings: Settings, n: int = 3) -> ResearchOutcome:
    registry = build_news_tools(ctx)
    a = settings.agent
    loop = AgentLoop(gateway, registry, task="researcher", max_steps=a.max_steps,
                     max_repeat_calls=a.max_repeat_calls, token_budget=a.token_budget)
    messages = [
        Message(role="system", content=RESEARCHER_SYSTEM.format(n=n) + "\n\n" + schema_instruction(ResearchResult)),
        Message(role="user", content=task_input({"need": n, "note": "今天的候选已经采集入库，请开始选题。"})),
    ]
    result = loop.run(messages)

    parsed, err = parse_as(result.final_text, ResearchResult)
    if parsed is None:  # 修一次：把校验错误发回去
        fix = gateway.chat(ChatRequest(messages=repair_messages(result.messages, result.final_text, err or ""),
                                       task="researcher", json_mode=True))
        parsed, err = parse_as(fix.message.content, ResearchResult)
    if parsed is None:
        raise ValueError(f"选题结果不合格：{err}")

    # ---------- 代码护栏 ----------
    candidates = {it.id: it for it in ctx.store.search_items("", 200, since=ctx.since)}
    picks, rejected, used = [], [], set()
    for p in parsed.picks:
        if p.item_id not in candidates:
            rejected.append((p.item_id, "不是今天的候选（模型编出来的 id）"))
        elif p.item_id in used:
            rejected.append((p.item_id, "重复选了同一条"))
        elif source_tier(p.primary_source_url, settings.safety.primary_domains) != "primary":
            rejected.append((p.item_id, f"来源不是一手的：{p.primary_source_url}"))
        else:
            picks.append(p)
            used.add(p.item_id)
    return ResearchOutcome(picks[:n], rejected, result)
