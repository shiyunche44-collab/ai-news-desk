"""各个 agent 的输入输出结构。

用 pydantic 定义结构化输出：模型返回的 JSON 先过校验，不合格就把错误信息回传让它改，
而不是让下游代码拿着残缺的数据往下跑（面试题：模型返回的格式不对怎么处理？）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPick(BaseModel):
    item_id: str = Field(description="候选新闻的 id")
    angle: str = Field(description="这条打算从什么角度写，一句话")
    why: str = Field(description="为什么今天值得写")
    primary_source_url: str = Field(description="一手来源的网址（官方博客、论文、代码仓库）")


class ResearchResult(BaseModel):
    picks: list[ResearchPick] = Field(max_length=5)


class Claim(BaseModel):
    text: str = Field(description="草稿里的一条事实陈述")
    source_url: str = Field(description="这条事实出自哪个网址")


class Draft(BaseModel):
    title: str
    body: str
    claims: list[Claim] = Field(min_length=1)


ClaimStatus = Literal["supported", "unsupported", "secondhand"]


class ClaimVerdict(BaseModel):
    claim: str
    status: ClaimStatus
    evidence: str = Field(default="", description="来源里能支撑这条的原句，没有就留空")
    note: str = ""


class FactCheckReport(BaseModel):
    verdicts: list[ClaimVerdict]
    summary: str = ""

    @property
    def passed(self) -> bool:
        return all(v.status == "supported" for v in self.verdicts)


class XiaohongshuNote(BaseModel):
    title: str = Field(description="小红书标题，20 字以内")
    body: str
    tags: list[str] = Field(default_factory=list, max_length=10)


class PlatformVersions(BaseModel):
    x_post: str = Field(description="X 帖子，按 X 的计数规则不超过 280（一个汉字算 2）")
    xiaohongshu: XiaohongshuNote
    shipinhao_script: str = Field(description="视频号口播稿，大约 60 秒")


class Summary(BaseModel):
    """第 1 周的小工具：网址 → 结构化摘要。"""

    title: str
    one_line: str = Field(description="一句话说清这是什么")
    key_points: list[str] = Field(min_length=1, max_length=5)
    entities: list[str] = Field(default_factory=list, description="涉及的公司、模型、人名")
    source_type: Literal["official", "paper", "repo", "media", "social", "unknown"] = "unknown"


class JudgeScore(BaseModel):
    """LLM 评审打分。"""

    score: int = Field(ge=1, le=5)
    reason: str
