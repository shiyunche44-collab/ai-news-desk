"""去重：同一条新闻会被官方博客、媒体、X 反复转，必须先合并再交给 agent。

两层：
1. 网址规范化后完全相同 → 重复；
2. 标题的字符 3-gram 相似度（Jaccard）超过阈值 → 近似重复。
   用字符而不是分词，是因为中英混排的标题分词不稳定，字符 n-gram 对两种语言都管用。

数据量大了以后可以换 SimHash 或 MinHash + LSH，复杂度从 O(n²) 降下来（第 8 周用开源小模型做语义去重）。
"""

from __future__ import annotations

import re

from ..tools.safety import canonical_url
from .models import NewsItem

_NOISE = re.compile(r"[\s\W_]+", re.UNICODE)


def shingles(text: str, n: int = 3) -> set[str]:
    t = _NOISE.sub("", text.lower())
    if len(t) <= n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe(items: list[NewsItem], seen_titles: list[str] | None = None,
           threshold: float = 0.6) -> tuple[list[NewsItem], list[tuple[NewsItem, str]]]:
    """返回 (保留的, [(丢掉的, 原因)])。

    同一事件有一手来源和二手来源时，优先保留一手来源那条。
    """
    # 一手来源排前面，这样近似重复时留下的是官方那条
    ordered = sorted(items, key=lambda it: 0 if it.tier == "primary" else 1)
    kept: list[NewsItem] = []
    kept_shingles: list[set[str]] = [shingles(t) for t in (seen_titles or [])]
    n_seen = len(kept_shingles)
    urls: set[str] = set()
    dropped: list[tuple[NewsItem, str]] = []

    for it in ordered:
        cu = canonical_url(it.url)
        if cu in urls:
            dropped.append((it, "网址重复"))
            continue
        sh = shingles(it.title)
        best = max((jaccard(sh, k) for k in kept_shingles), default=0.0)
        if best >= threshold:
            idx = max(range(len(kept_shingles)), key=lambda i: jaccard(sh, kept_shingles[i]))
            dropped.append((it, "和以前的内容近似重复" if idx < n_seen else "和今天另一条近似重复"))
            continue
        urls.add(cu)
        kept.append(it)
        kept_shingles.append(sh)
    return kept, dropped
