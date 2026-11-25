"""信息源。采集是固定流程，用普通代码写，不交给 agent：
抓什么、怎么解析、怎么去重都是确定的，交给模型只会更慢、更贵、更不稳定。
（面试题：什么时候用 workflow，什么时候用 agent？）
"""

from __future__ import annotations

import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from ..tools.safety import canonical_url, html_to_text, matches_domain, host_of, source_tier
from .models import Kind, NewsItem, item_id

USER_AGENT = "ai-news-desk/0.1 (+personal research project)"


class Source(Protocol):
    name: str

    def fetch(self) -> list[NewsItem]: ...


def guess_kind(url: str) -> Kind:
    host = host_of(url)
    if matches_domain(host, ["arxiv.org"]):
        return "paper"
    if matches_domain(host, ["github.com", "huggingface.co", "modelscope.cn"]):
        return "repo"
    if matches_domain(host, ["x.com", "twitter.com", "weibo.com"]):
        return "social"
    return "unknown"


def make_item(title: str, url: str, source: str, primary_domains: list[str], *,
              published: str | None = None, summary: str = "", kind: Kind | None = None,
              fixture: bool = False) -> NewsItem:
    tier = source_tier(url, primary_domains)
    k = kind or guess_kind(url)
    if k == "unknown":
        k = "official" if tier == "primary" else "media"
    return NewsItem(id=item_id(canonical_url(url)), title=title.strip(), url=url, source=source,
                    published=published, summary=summary.strip()[:600], kind=k, tier=tier,
                    fixture=fixture)


def _http_get(url: str, timeout: float = 15, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (网址来自配置)
        return resp.read(2_000_000)


class FixtureSource:
    """示例数据：离线、可复现，测试和评测都用它。"""

    def __init__(self, path: str, primary_domains: list[str], name: str = "fixtures"):
        self.path, self.name, self.primary_domains = path, name, primary_domains

    def fetch(self) -> list[NewsItem]:
        with open(self.path, encoding="utf-8") as f:
            rows = json.load(f)["items"]
        return [make_item(r["title"], r["url"], r.get("source", self.name), self.primary_domains,
                          published=r.get("published"), summary=r.get("summary", ""),
                          kind=r.get("kind"), fixture=True) for r in rows]


class RSSSource:
    """RSS 2.0 和 Atom 都支持，只用标准库。"""

    def __init__(self, url: str, primary_domains: list[str], name: str = "rss", limit: int = 30):
        self.url, self.name, self.primary_domains, self.limit = url, name, primary_domains, limit

    def fetch(self) -> list[NewsItem]:
        return self.parse(_http_get(self.url))

    def parse(self, data: bytes) -> list[NewsItem]:
        root = ET.fromstring(data)
        atom = "{http://www.w3.org/2005/Atom}"
        out: list[NewsItem] = []
        entries = root.findall(".//item") or root.findall(f".//{atom}entry")
        for e in entries[: self.limit]:
            if e.tag == "item":
                title = e.findtext("title") or ""
                link = e.findtext("link") or ""
                summary = e.findtext("description") or ""
                published = e.findtext("pubDate")
            else:
                title = e.findtext(f"{atom}title") or ""
                link_el = e.find(f"{atom}link")
                link = link_el.get("href", "") if link_el is not None else ""
                summary = e.findtext(f"{atom}summary") or e.findtext(f"{atom}content") or ""
                published = e.findtext(f"{atom}updated") or e.findtext(f"{atom}published")
            if title and link:
                out.append(make_item(title, link, self.name, self.primary_domains,
                                     published=published, summary=html_to_text(summary)))
        return out


class GitHubSearchSource:
    """GitHub 搜索：最近几天新建、星数最多的仓库。可选环境变量 GITHUB_TOKEN 提高限额。"""

    def __init__(self, query: str, primary_domains: list[str], name: str = "github",
                 days: int = 7, limit: int = 10):
        self.query, self.name, self.primary_domains = query, name, primary_domains
        self.days, self.limit = days, limit

    def fetch(self) -> list[NewsItem]:
        since = (datetime.now(timezone.utc) - timedelta(days=self.days)).date().isoformat()
        q = urllib.request.quote(f"{self.query} created:>{since}")
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page={self.limit}"
        headers = {"Accept": "application/vnd.github+json"}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        data: dict[str, Any] = json.loads(_http_get(url, headers=headers))
        return [
            make_item(f"{r['full_name']}：{(r.get('description') or '').strip()}"[:200], r["html_url"],
                      self.name, self.primary_domains, published=r.get("created_at"),
                      summary=f"{r.get('description') or ''}（{r.get('stargazers_count', 0)} 星，"
                              f"语言 {r.get('language') or '未知'}）", kind="repo")
            for r in data.get("items", [])
        ]


def build_sources(source_cfgs: list[dict[str, Any]], primary_domains: list[str]) -> list[Source]:
    out: list[Source] = []
    for s in source_cfgs:
        t = s["type"]
        if t == "fixture":
            out.append(FixtureSource(s["path"], primary_domains, s.get("name", "fixtures")))
        elif t == "rss":
            out.append(RSSSource(s["url"], primary_domains, s.get("name", "rss")))
        elif t == "github":
            out.append(GitHubSearchSource(s["query"], primary_domains, s.get("name", "github"),
                                          days=s.get("days", 7)))
        else:
            raise ValueError(f"不认识的信息源类型：{t}")
    return out
