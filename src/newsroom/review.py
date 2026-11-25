"""审核台（命令行版）：人看草稿，点通过或打回。

这是整个系统的「人在回路」（human-in-the-loop）：系统里没有任何自动发布的代码，
通过的草稿只会导出成 Markdown 文件，由你自己复制去发。第 7 周会做成网页版审核台。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents.adapter import x_weighted_length
from .store import Store

STATUS_LABEL = {"pending": "待审", "approved": "已通过", "rejected": "已打回"}


def format_draft(d: dict[str, Any], full: bool = True) -> str:
    lines = [f"[{d['id']}] {d['title']}  （{STATUS_LABEL.get(d['status'], d['status'])}）"]
    for flag in d.get("flags") or []:
        lines.append(f"  ⚠ {flag}")
    if not full:
        return "\n".join(lines)
    lines += ["", d["body"], "", "事实核查："]
    for v in (d.get("factcheck") or {}).get("verdicts", []):
        mark = {"supported": "✓", "unsupported": "✗", "secondhand": "?"}.get(v["status"], "?")
        lines.append(f"  {mark} {v['claim']}")
        if v.get("note"):
            lines.append(f"      {v['note']}")
    versions = d.get("versions") or {}
    if versions:
        x = versions.get("x_post", "")
        xhs = versions.get("xiaohongshu", {})
        lines += ["", f"X（计数 {x_weighted_length(x)}/280）：", x,
                  "", f"小红书：{xhs.get('title', '')}", xhs.get("body", ""),
                  "标签：" + "、".join(xhs.get("tags", [])),
                  "", "视频号口播稿：", versions.get("shipinhao_script", "")]
    return "\n".join(lines)


def _final_sections(d: dict[str, Any]) -> list[str]:
    """成稿部分：正文、来源和各平台版本。"""
    v = d.get("versions") or {}
    xhs = v.get("xiaohongshu", {})
    sources = sorted({c["source_url"] for c in d.get("claims") or []})
    return [
        d["body"], "",
        "## 来源", *[f"- {u}" for u in sources], "",
        "## X", v.get("x_post", ""), "",
        f"## 小红书：{xhs.get('title', '')}", xhs.get("body", ""), "",
        " ".join("#" + t for t in xhs.get("tags", [])), "",
        "## 视频号口播稿", v.get("shipinhao_script", ""), "",
    ]


def export_markdown(d: dict[str, Any], out_dir: Path) -> Path:
    """把通过的草稿导出成一个 Markdown 文件，方便复制到各平台。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join([f"# {d['title']}", "", *_final_sections(d)])
    path = out_dir / f"{d['created_at'][:10]}-{d['id']}.md"
    path.write_text(text, encoding="utf-8")
    return path


def export_for_review(d: dict[str, Any], out_dir: Path) -> Path:
    """把一篇草稿连同提醒和事实核查结果导出成 Markdown，给不在本机的人审（比如 GitHub Actions 的产物）。

    和 export_markdown 的区别：这里是「待审稿」，所以把状态、提醒、每条 claim 的核查结论都带上；
    export_markdown 导出的是已通过、可以直接复制去发的成稿。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marks = {"supported": "✓", "unsupported": "✗", "secondhand": "?"}
    lines = [f"# {d['title']}", "",
             f"草稿 id：`{d['id']}`　状态：{STATUS_LABEL.get(d['status'], d['status'])}", ""]
    flags = d.get("flags") or []
    if flags:
        lines += ["## 提醒", *[f"- ⚠ {f}" for f in flags], ""]
    lines += ["## 事实核查"]
    for v in (d.get("factcheck") or {}).get("verdicts", []):
        lines.append(f"- {marks.get(v['status'], '?')} {v['claim']}")
        if v.get("note"):
            lines.append(f"  - {v['note']}")
    lines += ["", "## 正文", *_final_sections(d)]
    path = out_dir / f"{d['created_at'][:10]}-{d['id']}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def approve(store: Store, draft_id: str, out_dir: Path, note: str = "", force: bool = False) -> Path:
    d = store.get_draft(draft_id)
    if d is None:
        raise KeyError(f"没有这篇草稿：{draft_id}")
    fixture_flag = any("示例数据" in f for f in d.get("flags") or [])
    failed = [v for v in (d.get("factcheck") or {}).get("verdicts", []) if v["status"] != "supported"]
    if (fixture_flag or failed) and not force:
        why = "这是示例数据（虚构）" if fixture_flag else f"有 {len(failed)} 条事实核查没通过"
        raise ValueError(f"{why}。确认要通过就加 --force。")
    store.set_draft_status(draft_id, "approved", note)
    return export_markdown(store.get_draft(draft_id) or d, out_dir)


def reject(store: Store, draft_id: str, note: str = "") -> None:
    if store.get_draft(draft_id) is None:
        raise KeyError(f"没有这篇草稿：{draft_id}")
    store.set_draft_status(draft_id, "rejected", note)
