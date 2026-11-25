"""离线模拟模型：不联网、不要 key、结果固定，用来跑通流程、写测试、在 CI 里跑评测。

它不是真的语言模型，只是按「哪个环节在调用」（request.task）用简单规则给出合法的回答，
包括发起工具调用、并行调用、输出符合 schema 的 JSON。这样整条流水线的每一个接口都能被测到。

另外提供 ScriptedProvider：按顺序吐出事先写好的回复，专门用来测 agent 循环的护栏
（比如模型一直重复调用同一个工具时，循环能不能停下来）。
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Callable

from ..collect.dedupe import shingles
from ..config import ModelConfig
from ..tools.safety import scan_injection
from .types import ChatRequest, ChatResponse, Message, ToolCall, Usage

_ids = itertools.count(1)
_TASK_INPUT = re.compile(r"<task_input>\s*(.*?)\s*</task_input>", re.DOTALL)
_UNTRUSTED = re.compile(r"<untrusted_content[^>]*>\s*(.*?)\s*</untrusted_content>", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；])|\n+")

Handler = Callable[[ChatRequest], Message]


def _new_id() -> str:
    return f"call_mock_{next(_ids)}"


def _task_input(req: ChatRequest) -> dict[str, Any]:
    for m in reversed(req.messages):
        if m.role == "user":
            hit = _TASK_INPUT.search(m.content)
            if hit:
                return json.loads(hit.group(1))
    return {}


def _plain(text: str) -> str:
    hit = _UNTRUSTED.search(text)
    return hit.group(1) if hit else text


def _sentences(text: str) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text):
        s = s.strip(" -*#\t")
        if len(s) >= 12 and not scan_injection(s) and not s.startswith(("【示例", "示例数据")):
            out.append(s)
    return out


def _containment(claim: str, source: str) -> float:
    c = shingles(claim)
    return len(c & shingles(source)) / len(c) if c else 0.0


def _json_msg(obj: Any) -> Message:
    return Message(role="assistant", content=json.dumps(obj, ensure_ascii=False))


# ---------------- 各环节的规则 ----------------

def _researcher(req: ChatRequest) -> Message:
    need = int(_task_input(req).get("need", 3))
    tool_msgs = [m for m in req.messages if m.role == "tool"]
    searched = [m for m in tool_msgs if m.name == "search_items" and not m.is_error]
    fetched = [m for m in tool_msgs if m.name == "fetch_url" and not m.is_error]

    candidates: list[dict[str, Any]] = []
    if searched:
        try:
            candidates = json.loads(searched[-1].content)
        except json.JSONDecodeError:  # 结果太长被截断了
            candidates = []
    if req.tools and not searched:
        return Message(role="assistant", content="先看看今天有哪些候选。", tool_calls=[
            ToolCall(_new_id(), "search_items", {"query": "", "limit": 20})])

    tried = {c.arguments.get("url") for m in req.messages if m.role == "assistant"
             for c in m.tool_calls if c.name == "fetch_url"}
    if req.tools and len(fetched) < need and candidates:
        order = {"official": 0, "paper": 1, "repo": 2}
        primary = [c for c in candidates if c["tier"] == "primary" and c["url"] not in tried]
        primary.sort(key=lambda c: order.get(c["kind"], 9))
        chosen = primary[: need - len(fetched) + 1]  # 多读一条备用；读失败的不再重试，换下一条
        if chosen:
            # 一次发出多个工具调用 → 演示并行工具调用
            return Message(role="assistant", content="读一下这几条的原文。", tool_calls=[
                ToolCall(_new_id(), "fetch_url", {"url": c["url"]}) for c in chosen])

    by_url = {c["url"]: c for c in candidates}
    picks = []
    for m in fetched:
        page = json.loads(m.content)
        cand = by_url.get(page["url"])
        if not cand or page["tier"] != "primary":
            continue
        why = "一手来源，信息量足。"
        if page.get("injection_flags"):
            why += "原文含可疑指令，已忽略其中的要求。"
        picks.append({"item_id": cand["id"], "angle": f"讲清楚「{cand['title']}」对开发者意味着什么",
                      "why": why, "primary_source_url": cand["url"]})
    return _json_msg({"picks": picks[:need]})


def _writer(req: ChatRequest) -> Message:
    inp = _task_input(req)
    text = _plain(inp.get("source_text", ""))
    facts = _sentences(text)[:3] or [text[:60]]
    url = inp.get("source_url", "")
    title = inp.get("item", {}).get("title", "今日 AI 资讯")
    body = facts[0]
    if len(facts) > 1:
        body += "\n\n为什么值得关注：" + "".join(facts[1:])
    body += "\n\n对开发者来说：建议先读原文，再决定要不要上手试。"
    return _json_msg({"title": title, "body": body,
                      "claims": [{"text": f, "source_url": url} for f in facts]})


def _fact_checker(req: ChatRequest) -> Message:
    inp = _task_input(req)
    sources = " ".join(_plain(t) for t in inp.get("sources", {}).values())
    verdicts = []
    for c in inp.get("claims", []):
        score = _containment(c["text"], sources)
        if score >= 0.7:
            best = max(_SENT_SPLIT.split(sources) or [""], key=lambda s: _containment(c["text"], s))
            verdicts.append({"claim": c["text"], "status": "supported", "evidence": best.strip()[:200]})
        else:
            verdicts.append({"claim": c["text"], "status": "unsupported", "evidence": "",
                             "note": f"原文里找不到对应内容（重合度 {score:.0%}）"})
    ok = sum(v["status"] == "supported" for v in verdicts)
    return _json_msg({"verdicts": verdicts, "summary": f"{ok}/{len(verdicts)} 条有来源支撑"})


def _adapter(req: ChatRequest) -> Message:
    from ..agents.adapter import x_weighted_length

    inp = _task_input(req)
    title, body = inp.get("title", ""), inp.get("body", "")
    url = (inp.get("source_urls") or [""])[0]
    first = _sentences(body)[0] if _sentences(body) else body[:60]
    x_post = f"{title}：{first}"
    while x_weighted_length(x_post) > 250:
        x_post = x_post[:-2]
    x_post = f"{x_post} {url}".strip()
    paras = [p for p in body.split("\n\n") if p.strip()]
    script = "今天说一条 AI 新闻。" + "".join(paras)
    while len(script) < 150:
        script += "如果你也在关注 AI，记得点个关注，我每天用大白话讲一条。"
    return _json_msg({
        "x_post": x_post,
        "xiaohongshu": {"title": title[:20], "body": "\n\n".join(paras[:4]),
                        "tags": ["AI", "大模型", "AI资讯"]},
        "shipinhao_script": script[:400],
    })


def _summarize(req: ChatRequest) -> Message:
    inp = _task_input(req)
    text = _plain(inp.get("source_text", ""))
    sents = _sentences(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and sents and sents[0] == lines[0].strip(" -*#\t"):
        sents = sents[1:]  # 第一行是标题，不算要点
    entities = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9\-\.]{2,}\b", text)))[:8]
    url = inp.get("url", "")
    st = ("paper" if "arxiv.org" in url else "repo" if "github.com" in url
          else "official" if inp.get("tier") == "primary" else "media")
    return _json_msg({"title": lines[0][:80] if lines else "无标题",
                      "one_line": sents[0] if sents else text[:80],
                      "key_points": sents[1:5] or sents[:1] or [text[:80]],
                      "entities": entities, "source_type": st})


def _judge(req: ChatRequest) -> Message:
    inp = _task_input(req)
    draft, source = inp.get("draft", {}), _plain(inp.get("source_text", ""))
    claims = draft.get("claims", [])
    supported = sum(_containment(c["text"], source) >= 0.7 for c in claims)
    ratio = supported / len(claims) if claims else 0
    hype = any(w in draft.get("title", "") + draft.get("body", "") for w in ("震惊", "重磅", "炸裂"))
    score = 1 + round(4 * ratio) - (1 if hype else 0)
    return _json_msg({"score": max(1, min(5, score)),
                      "reason": f"{supported}/{len(claims)} 条事实能在来源里找到" + ("；有标题党用词" if hype else "")})


def _react(req: ChatRequest) -> Message:
    user = req.messages[-1].content
    if "Observation:" not in user:
        return Message(role="assistant", content=(
            "Thought: 先看看今天有哪些候选。\nAction: search_items\n"
            'Action Input: {"query": "", "limit": 5}'))
    obs = user.rsplit("Observation:", 1)[1].strip()
    try:
        items = json.loads(obs)
        primary = [it for it in items if it.get("tier") == "primary"]
        if primary:
            answer = primary[0]["title"]
        elif items:
            answer = "剩下的候选都只有二手来源，今天不建议写"
        else:
            answer = "今天没有候选"
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        answer = "没拿到结构化结果"
    return Message(role="assistant", content=f"Thought: 我已经知道答案了\nFinal Answer: {answer}")


HANDLERS: dict[str, Handler] = {
    "researcher": _researcher, "writer": _writer, "fact_checker": _fact_checker,
    "adapter": _adapter, "summarize": _summarize, "judge": _judge, "react": _react,
}


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 1.5))  # 粗略估算：中文大约 1.5 个字一个 token


class MockProvider:
    def __init__(self, cfg: ModelConfig | None = None, handlers: dict[str, Handler] | None = None):
        self.cfg = cfg
        self.name = cfg.name if cfg else "mock"
        self.handlers = {**HANDLERS, **(handlers or {})}
        self.calls: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        handler = self.handlers.get(request.task)
        msg = handler(request) if handler else Message(role="assistant", content="（mock）收到")
        prompt_chars = sum(len(m.content) for m in request.messages)
        out_chars = len(msg.content) + sum(len(json.dumps(c.arguments)) for c in msg.tool_calls)
        return ChatResponse(
            message=msg,
            usage=Usage(_estimate_tokens(" " * prompt_chars), _estimate_tokens(" " * out_chars)),
            model=self.cfg.model if self.cfg else "mock",
            provider=self.name,
            finish_reason="tool_calls" if msg.tool_calls else "stop",
            latency_ms=5 + out_chars / 100,
        )


class ScriptedProvider:
    """按顺序返回写好的回复，用完了就一直返回最后一条。"""

    def __init__(self, replies: list[Message], name: str = "scripted"):
        self.replies, self.name, self.calls = replies, name, []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        msg = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        # 每次都复制一份，并给工具调用换新 id，模拟真实模型
        msg = Message(role="assistant", content=msg.content,
                      tool_calls=[ToolCall(_new_id(), c.name, dict(c.arguments), c.raw_arguments, c.parse_error)
                                  for c in msg.tool_calls])
        return ChatResponse(message=msg, usage=Usage(100, 20), model="scripted", provider=self.name,
                            finish_reason="tool_calls" if msg.tool_calls else "stop", latency_ms=1)
