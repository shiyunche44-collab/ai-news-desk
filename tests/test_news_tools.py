"""研究员的工具：结果太长时整条去掉，保证返回的 JSON 能解析。"""

import json

from newsroom.collect.sources import make_item
from newsroom.gateway.types import ToolCall
from newsroom.tools.fetcher import PageFetcher
from newsroom.tools.news_tools import ToolContext, build_news_tools
from newsroom.tools.registry import MAX_RESULT_CHARS


def test_search_items_stays_valid_json_when_long(store):
    # 真实 arXiv 数据的摘要很长：20 条一起返回会超过注册表的截断长度
    items = [make_item(f"Paper {i}: a long title about agents and planning", f"https://arxiv.org/abs/2611.{10000 + i}",
                       "arXiv", ["arxiv.org"], summary="abstract " * 80) for i in range(30)]
    store.add_items(items, "r1", {it.id: it.url for it in items})
    reg = build_news_tools(ToolContext(store, PageFetcher([], [], live=False)))
    res = reg.execute(ToolCall("c1", "search_items", {"query": "", "limit": 20}))
    rows = json.loads(res.content)  # 不能被从中间截断
    assert not res.is_error and 1 <= len(rows) < 20 and len(res.content) <= MAX_RESULT_CHARS
