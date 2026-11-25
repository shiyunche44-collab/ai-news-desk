"""采集工作流：抓取 → 规范化 → 去重 → 入库。全是确定性代码。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..store import Store
from ..tools.safety import canonical_url
from .dedupe import dedupe
from .models import NewsItem
from .sources import Source


@dataclass
class CollectReport:
    fetched: int = 0
    kept: list[NewsItem] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (标题, 原因)
    errors: list[str] = field(default_factory=list)


def collect(sources: list[Source], store: Store, run_id: str) -> CollectReport:
    report = CollectReport()
    raw: list[NewsItem] = []
    for src in sources:
        try:
            got = src.fetch()
            raw.extend(got)
            report.fetched += len(got)
        except Exception as e:  # 一个源挂了不影响其他源
            report.errors.append(f"{src.name}: {type(e).__name__}: {e}")

    known = store.known_urls()
    fresh = []
    for it in raw:
        if canonical_url(it.url) in known:
            report.dropped.append((it.title, "以前采集过"))
        else:
            fresh.append(it)

    kept, dropped = dedupe(fresh, seen_titles=store.recent_titles() + store.recent_item_titles())
    report.kept = kept
    report.dropped += [(it.title, why) for it, why in dropped]
    store.add_items(kept, run_id, {it.id: canonical_url(it.url) for it in kept})
    # 丢掉的也入库（标成 dropped），记住「这个网址处理过了」
    dropped_items = {it.id: it for it, _ in dropped}
    store.add_items(list(dropped_items.values()), run_id,
                    {i: canonical_url(it.url) for i, it in dropped_items.items()},
                    status="dropped", reasons={it.id: why for it, why in dropped})
    return report
