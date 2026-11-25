"""事实核查：代码先查一遍，再让模型逐条核对。

能用代码判断的，绝不交给模型：
- 草稿里某条 claim 引用的网址不在资料里 → 直接判 unsupported（模型可能编了出处）；
- 引用的网址是二手来源 → 直接判 secondhand；
- claim 里有原文没出现过的数字 → 直接判 unsupported。资讯最常见的错就是数字写错
  （32K 写成 128K），而模型核查时恰恰容易被「整句话意思差不多」糊弄过去。
  这条规则偏严（原文写 32000、草稿写 32K 也会被拦），宁可多让人看一眼。
剩下的「这句话和原文意思一致吗」才需要模型判断。
"""

from __future__ import annotations

import re

from ..agent.structured import ask_structured
from ..gateway.gateway import ModelGateway
from ..tools.fetcher import FetchedPage
from ..tools.safety import canonical_url, wrap_untrusted
from .prompts import FACT_CHECK_SYSTEM
from .schemas import ClaimVerdict, Draft, FactCheckReport

_NUM = re.compile(r"\d+(?:\.\d+)?")


def missing_numbers(claim: str, source: str) -> list[str]:
    """claim 里出现、但原文里没有的数字。"""
    have = set(_NUM.findall(source))
    return [n for n in _NUM.findall(claim) if n not in have]


def check_draft(gateway: ModelGateway, draft: Draft, sources: dict[str, FetchedPage]) -> FactCheckReport:
    by_url = {canonical_url(u): p for u, p in sources.items()}
    pre: dict[int, ClaimVerdict] = {}
    to_llm: list[int] = []
    for i, c in enumerate(draft.claims):
        page = by_url.get(canonical_url(c.source_url))
        if page is None:
            pre[i] = ClaimVerdict(claim=c.text, status="unsupported", note="引用的网址不在给写手的资料里")
        elif page.tier != "primary":
            pre[i] = ClaimVerdict(claim=c.text, status="secondhand", note="只有二手来源")
        elif missing := missing_numbers(c.text, page.text):
            pre[i] = ClaimVerdict(claim=c.text, status="unsupported",
                                  note=f"数字 {'、'.join(missing)} 在原文里找不到")
        else:
            to_llm.append(i)

    llm_verdicts: list[ClaimVerdict] = []
    summary = ""
    if to_llm:
        payload = {
            "claims": [{"index": i, "text": draft.claims[i].text, "source_url": draft.claims[i].source_url}
                       for i in to_llm],
            "sources": {p.url: wrap_untrusted(p.text, p.url) for p in sources.values()},
        }
        report = ask_structured(gateway, FACT_CHECK_SYSTEM, payload, FactCheckReport, task="fact_checker").value
        llm_verdicts, summary = report.verdicts, report.summary

    # 按原顺序合并：先按原句匹配，匹配不上再按顺序；模型漏判的条目按 unsupported 处理（宁严勿松）
    verdicts: list[ClaimVerdict] = []
    by_text = {v.claim.strip(): v for v in llm_verdicts}
    llm_iter = iter(llm_verdicts)
    for i, c in enumerate(draft.claims):
        if i in pre:
            verdicts.append(pre[i])
        else:
            v = by_text.get(c.text.strip()) or next(llm_iter, None)
            verdicts.append(v if v is not None else ClaimVerdict(claim=c.text, status="unsupported",
                                                                 note="核查模型漏掉了这一条"))
    return FactCheckReport(verdicts=verdicts, summary=summary)
