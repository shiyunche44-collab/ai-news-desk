"""第 1 周的小工具：一个网址或一个本地文件 → 结构化摘要。

麻雀虽小，第一版就把后面每一周都要用的三件事放进来了：
统一的模型网关、外部内容当「不可信数据」包起来、用 pydantic 校验结构化输出。
"""

from __future__ import annotations

from pathlib import Path

from .agent.structured import ask_structured
from .agents.prompts import SUMMARIZE_SYSTEM
from .agents.schemas import Summary
from .config import Settings
from .gateway.gateway import ModelGateway
from .tools.fetcher import PageFetcher
from .tools.safety import html_to_text, scan_injection, source_tier, wrap_untrusted


def load_source(target: str, settings: Settings, live: bool = True) -> tuple[str, str, str, list[str]]:
    """返回 (正文, 网址或路径, 来源等级, 可疑指令)。"""
    if target.startswith(("http://", "https://")):
        fetcher = PageFetcher(settings.safety.fetch_allowlist, settings.safety.primary_domains,
                              settings.safety.fetch_max_chars, settings.root / "data/fixtures/pages",
                              live=live)
        page = fetcher.fetch(target)
        return page.text, page.url, page.tier, page.injection_flags
    path = Path(target)
    raw = path.read_text(encoding="utf-8")
    text = html_to_text(raw) if path.suffix in {".html", ".htm"} else raw
    text = text[: settings.safety.fetch_max_chars]
    return text, str(path), "unknown", scan_injection(text)


def summarize(gateway: ModelGateway, text: str, url: str = "", tier: str = "unknown") -> Summary:
    payload = {"url": url, "tier": tier, "source_text": wrap_untrusted(text, url or "local")}
    summary = ask_structured(gateway, SUMMARIZE_SYSTEM, payload, Summary, task="summarize").value
    if tier == "primary" and summary.source_type in {"unknown", "media"}:
        summary.source_type = "official"  # 来源等级以代码判断为准，不听模型的
    return summary
