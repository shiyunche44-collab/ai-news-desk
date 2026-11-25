from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel

Kind = Literal["official", "paper", "repo", "media", "social", "unknown"]


class NewsItem(BaseModel):
    id: str
    title: str
    url: str
    source: str
    published: str | None = None
    summary: str = ""
    kind: Kind = "unknown"
    tier: Literal["primary", "secondary"] = "secondary"
    fixture: bool = False  # True 表示是示例数据（虚构的）


def item_id(canonical: str) -> str:
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]
