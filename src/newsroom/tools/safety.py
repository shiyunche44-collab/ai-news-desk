"""安全相关的小工具：域名白名单、来源分级、提示注入检测、把外部内容标成「不可信数据」。

对应 OWASP LLM Top 10 里最常考的两条：
- 提示注入：网页里藏着「忽略之前的指令」之类的话。防法是多层的——
  外部内容一律包在 <untrusted_content> 里并在系统提示里说明它只是数据；
  用规则扫一遍可疑句子并打标；最关键的是 agent 根本没有「发布」这种高风险工具。
- 过度授权：agent 能访问的网址用白名单卡住，发布永远要人点。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# 常见注入话术（中英文）。规则只能兜住一部分，所以它只负责「打标提醒」，真正的防线是权限设计。
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
    r"disregard (the |all )?(previous|above|system)",
    r"you are now",
    r"system prompt",
    r"new instructions?:",
    r"忽略(之前|以上|上面|前面)(的)?(所有)?(指令|指示|提示|要求)",
    r"无视(之前|以上|上面)(的)?(指令|规则)",
    r"你现在(是|扮演)",
    r"(请|立即|马上)(直接)?(发布|发帖|转发)",
    r"(把|将)(你的)?(系统提示|提示词|密钥|api ?key)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                    "spm", "from", "ref", "share_token"}


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def matches_domain(host: str, domains: list[str]) -> bool:
    """example.com 匹配 example.com 和 blog.example.com，但不匹配 badexample.com。"""
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def source_tier(url: str, primary_domains: list[str]) -> str:
    return "primary" if matches_domain(host_of(url), primary_domains) else "secondary"


def canonical_url(url: str) -> str:
    """去掉追踪参数、锚点、末尾斜杠，统一大小写，用来去重。"""
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING_PARAMS])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", (parts.hostname or "").lower(), path, query, ""))


def scan_injection(text: str) -> list[str]:
    hits = []
    for rx in _INJECTION_RE:
        m = rx.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def wrap_untrusted(text: str, source: str) -> str:
    """把外部内容包起来。先把内容里伪造的结束标签转义掉，防止它「跳出」包裹。"""
    safe = text.replace("</untrusted_content", "&lt;/untrusted_content")
    return f'<untrusted_content source="{source}">\n{safe}\n</untrusted_content>'


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "div", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
