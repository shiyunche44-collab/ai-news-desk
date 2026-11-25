"""抓网页正文。先查白名单，再看是不是示例数据，最后才真的联网。"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from ..collect.sources import USER_AGENT
from .safety import canonical_url, host_of, html_to_text, matches_domain, scan_injection, source_tier


class FetchBlocked(Exception):
    pass


@dataclass
class FetchedPage:
    url: str
    text: str
    tier: str
    injection_flags: list[str] = field(default_factory=list)


class PageFetcher:
    def __init__(self, allowlist: list[str], primary_domains: list[str], max_chars: int = 12000,
                 fixtures_dir: str | Path | None = None, live: bool = True):
        self.allowlist, self.primary_domains, self.max_chars = allowlist, primary_domains, max_chars
        self.live = live
        self.fixture_pages: dict[str, Path] = {}
        if fixtures_dir:
            index = Path(fixtures_dir) / "index.json"
            if index.exists():
                mapping = json.loads(index.read_text(encoding="utf-8"))
                self.fixture_pages = {canonical_url(u): Path(fixtures_dir) / f for u, f in mapping.items()}
        self.cache: dict[str, FetchedPage] = {}

    def fetch(self, url: str) -> FetchedPage:
        if not url.startswith(("http://", "https://")):
            raise FetchBlocked("只支持 http 和 https 网址")
        if not matches_domain(host_of(url), self.allowlist):
            # 白名单外的网址一律不访问。这是防「网页诱导 agent 去访问恶意地址」的硬限制
            raise FetchBlocked(f"{host_of(url)} 不在 fetch_allowlist 里")
        key = canonical_url(url)
        if key in self.cache:
            return self.cache[key]

        if key in self.fixture_pages:
            raw = self.fixture_pages[key].read_text(encoding="utf-8")
        elif self.live:
            raw = self._download(url)
        else:
            raise FetchBlocked("离线模式下只能读示例页面")

        text = html_to_text(raw) if "<" in raw[:500] else raw
        page = FetchedPage(url=url, text=text[: self.max_chars],
                           tier=source_tier(url, self.primary_domains),
                           injection_flags=scan_injection(text))
        self.cache[key] = page
        return page

    def _download(self, url: str) -> str:
        parts = [p for p in urlsplit(url).path.split("/") if p]
        if host_of(url) == "github.com" and len(parts) == 2:
            # GitHub 仓库首页的 HTML 全是导航栏，直接读 README 原文干净得多
            owner, repo = parts
            targets = [(f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md", {}),
                       (f"https://api.github.com/repos/{owner}/{repo}/readme",
                        {"Accept": "application/vnd.github.raw"})]
        else:
            targets = [(url, {})]
        last: Exception | None = None
        for target, extra in targets:
            if not matches_domain(host_of(target), self.allowlist):
                continue
            headers = {"User-Agent": USER_AGENT, **extra}
            if host_of(target) == "api.github.com" and os.environ.get("GITHUB_TOKEN"):
                headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
            try:
                req = urllib.request.Request(target, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (已过白名单)
                    return resp.read(1_000_000).decode("utf-8", errors="replace")
            except Exception as e:  # 换下一个地址试试
                last = e
        raise last or FetchBlocked(f"{host_of(url)} 没有可用的地址")
