"""评测打分器。

原则：能用代码判断的用代码（快、免费、结果稳定），只有「写得好不好」这种主观的才交给 LLM 评审，
而且评审模型和被评模型分开，避免「自己给自己打分」（面试题：LLM-as-judge 有什么偏差，怎么缓解？）。

每个打分器返回 0 到 1 的分数和一句说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agent.structured import ask_structured
from ..agents.adapter import platform_problems
from ..agents.prompts import JUDGE_SYSTEM
from ..agents.schemas import Draft, FactCheckReport, JudgeScore, PlatformVersions
from ..collect.dedupe import shingles
from ..gateway.gateway import ModelGateway
from ..tools.safety import canonical_url, source_tier, wrap_untrusted

HYPE_WORDS = ("震惊", "重磅", "炸裂", "颠覆", "史上最", "吊打")
GROUNDED_AT = 0.6  # 一条 claim 的字符 3-gram 有 60% 能在原文里找到，才算「有出处」

# 一个 case 的总分怎么加权。事实性权重最高：资讯号写错一条事实的代价远大于文笔差一点
WEIGHTS = {"format": 0.15, "sources": 0.2, "grounding": 0.3, "style": 0.15, "judge": 0.2}


@dataclass
class Score:
    name: str
    value: float
    detail: str = ""


def containment(claim: str, source: str) -> float:
    c = shingles(claim)
    return len(c & shingles(source)) / len(c) if c else 0.0


def score_format(versions: PlatformVersions | None, problems: list[str]) -> Score:
    if versions is None:
        return Score("format", 0.0, "平台改写没产出合法结构")
    if problems:
        return Score("format", max(0.0, 1 - 0.34 * len(problems)), "；".join(problems))
    return Score("format", 1.0, "三个平台版本都符合长度限制")


def score_sources(draft: Draft, source_url: str, primary_domains: list[str]) -> Score:
    good = [c for c in draft.claims
            if canonical_url(c.source_url) == canonical_url(source_url)
            and source_tier(c.source_url, primary_domains) == "primary"]
    ratio = len(good) / len(draft.claims)
    return Score("sources", ratio, f"{len(good)}/{len(draft.claims)} 条事实引用的是给定的一手来源")


def score_grounding(draft: Draft, source_text: str) -> Score:
    """确定性的事实检查：每条 claim 能不能在原文里找到。比 LLM 核查粗，但不会被说服。"""
    weak = [c.text for c in draft.claims if containment(c.text, source_text) < GROUNDED_AT]
    ratio = 1 - len(weak) / len(draft.claims)
    detail = "每条事实都能在原文里找到" if not weak else "原文里找不到：" + "；".join(w[:30] for w in weak)
    return Score("grounding", ratio, detail)


def score_style(draft: Draft, must_include: list[str], must_not_include: list[str]) -> Score:
    text = draft.title + "\n" + draft.body
    problems = []
    hype = [w for w in HYPE_WORDS if w in text]
    if hype:
        problems.append("标题党用词：" + "、".join(hype))
    if len(draft.body) > 400:
        problems.append(f"正文 {len(draft.body)} 字，超过 300 字的要求太多")
    missing = [w for w in must_include if w not in text]
    if missing:
        problems.append("缺少关键信息：" + "、".join(missing))
    leaked = [w for w in must_not_include if w in text]
    if leaked:
        problems.append("出现了不该出现的内容：" + "、".join(leaked))
    return Score("style", max(0.0, 1 - 0.34 * len(problems)), "；".join(problems) or "符合写作要求")


def judge(gateway: ModelGateway, draft: Draft, source_text: str, model: str | None) -> Score:
    payload = {"draft": draft.model_dump(), "source_text": wrap_untrusted(source_text, "golden")}
    js: JudgeScore = ask_structured(gateway, JUDGE_SYSTEM, payload, JudgeScore,
                                    task="judge", model=model).value
    return Score("judge", (js.score - 1) / 4, f"{js.score}/5：{js.reason}")


def score_factcheck(report: FactCheckReport, expected: list[dict[str, Any]]) -> Score:
    """核查员评测：草稿里故意埋了错的 claim，看核查员能不能抓出来、又不冤枉对的。"""
    hits = 0
    misses = []
    for exp, v in zip(expected, report.verdicts):
        ok_expected = exp["expect"] == "supported"
        if (v.status == "supported") == ok_expected:
            hits += 1
        else:
            misses.append(("漏掉了错误：" if not ok_expected else "冤枉了正确的：") + exp["text"][:30])
    total = max(len(expected), 1)
    return Score("factcheck", hits / total, "全部判对" if not misses else "；".join(misses))


def weighted_total(scores: list[Score]) -> float:
    by_name = {s.name: s.value for s in scores}
    used = {k: w for k, w in WEIGHTS.items() if k in by_name}
    if not used:  # 核查类的 case 只有一个分数
        return sum(by_name.values()) / max(len(by_name), 1)
    return sum(by_name[k] * w for k, w in used.items()) / sum(used.values())
