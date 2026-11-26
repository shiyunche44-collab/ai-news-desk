"""平台改写：一篇草稿 → 一条 X 帖子，再用代码检查 X 的长度限制。"""

from __future__ import annotations

import re

from ..agent.structured import ask_structured
from ..gateway.gateway import ModelGateway
from ..tools.safety import canonical_url
from .prompts import ADAPTER_SYSTEM
from .schemas import Draft, PlatformVersions

_URL = re.compile(r"https?://\S+")


def x_weighted_length(text: str) -> int:
    """X 的计数规则：中日韩文字算 2，网址不管多长都算 23，其他字符算 1。"""
    n = 0
    for url in _URL.findall(text):
        n += 23
    rest = _URL.sub("", text)
    for ch in rest:
        cp = ord(ch)
        wide = (0x1100 <= cp <= 0x115F or 0x2E80 <= cp <= 0xA4CF or 0xAC00 <= cp <= 0xD7A3
                or 0xF900 <= cp <= 0xFAFF or 0xFE30 <= cp <= 0xFE4F or 0xFF00 <= cp <= 0xFF60
                or 0x20000 <= cp <= 0x3FFFD)
        n += 2 if wide else 1
    return n


def platform_problems(v: PlatformVersions) -> list[str]:
    problems = []
    if x_weighted_length(v.x_post) > 280:
        problems.append(f"X 帖子超长（计数 {x_weighted_length(v.x_post)}，上限 280）")
    return problems


def adapt(gateway: ModelGateway, draft: Draft) -> tuple[PlatformVersions, list[str]]:
    payload = {"title": draft.title, "body": draft.body,
               "source_urls": sorted({canonical_url(c.source_url) for c in draft.claims})}  # 去掉追踪参数
    versions = ask_structured(gateway, ADAPTER_SYSTEM, payload, PlatformVersions, task="adapter").value
    return versions, platform_problems(versions)
