"""写手：一次调用，不给工具。

写作不需要「自己决定去查什么」——素材已经由研究员找好了，所以这里是 workflow 的一步，
不是 agent。子 agent 只拿到它需要的东西（这条新闻 + 原文 + 语气档案），上下文是隔离的：
不会看到研究员读过的其他十几篇网页，既省 token 也不容易串台。
"""

from __future__ import annotations

from ..agent.structured import ask_structured
from ..collect.models import NewsItem
from ..gateway.gateway import ModelGateway
from ..tools.fetcher import FetchedPage
from ..tools.safety import wrap_untrusted
from .prompts import WRITER_SYSTEM
from .schemas import Draft, ResearchPick

DEFAULT_VOICE = "用大白话讲 AI，像跟懂点技术的朋友聊天；少用形容词，多给具体信息；不夸张。"


def write_draft(gateway: ModelGateway, item: NewsItem, pick: ResearchPick, page: FetchedPage,
                voice: str = DEFAULT_VOICE) -> Draft:
    payload = {
        "item": {"title": item.title, "url": item.url, "source": item.source, "published": item.published},
        "angle": pick.angle,
        "source_url": page.url,
        "source_text": wrap_untrusted(page.text, page.url),
    }
    return ask_structured(gateway, WRITER_SYSTEM.format(voice=voice), payload, Draft, task="writer").value
