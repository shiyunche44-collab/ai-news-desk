"""SQLite 存储：新闻、运行记录、草稿、每次模型调用的账单。

第一版用 SQLite，零配置、一个文件就能跑。表结构按以后迁到 Postgres + pgvector 设计（第 4 周），
到时只换这一层，上面的 agent 代码不动。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .collect.models import NewsItem

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY, title TEXT, url TEXT, canonical_url TEXT UNIQUE, source TEXT,
  published TEXT, summary TEXT, kind TEXT, tier TEXT, fixture INTEGER, run_id TEXT, collected_at TEXT,
  status TEXT, drop_reason TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, model TEXT, status TEXT,
  stats_json TEXT, trace_json TEXT
);
CREATE TABLE IF NOT EXISTS drafts (
  id TEXT PRIMARY KEY, run_id TEXT, item_id TEXT, title TEXT, body TEXT, claims_json TEXT,
  factcheck_json TEXT, versions_json TEXT, flags_json TEXT, status TEXT,
  created_at TEXT, reviewed_at TEXT, review_note TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, task TEXT, model_name TEXT, model TEXT,
  input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER, latency_ms REAL,
  cost REAL, currency TEXT, ok INTEGER, error TEXT, created_at TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---------- items ----------
    def known_urls(self) -> set[str]:
        """采集过的网址，包括被去重丢掉的，这样丢掉的那条明天也不会再冒出来。"""
        return {r[0] for r in self.db.execute("SELECT canonical_url FROM items")}

    def recent_titles(self, limit: int = 500) -> list[str]:
        """写过的草稿标题。"""
        rows = self.db.execute("SELECT title FROM drafts ORDER BY created_at DESC LIMIT ?", (limit,))
        return [r[0] for r in rows]

    def recent_item_titles(self, hours: float = 72) -> list[str]:
        """最近采集到的候选标题：今天的媒体转载要和昨天的官方原文比对。"""
        rows = self.db.execute("SELECT title FROM items WHERE status='candidate' AND collected_at>=?",
                               (hours_ago(hours),))
        return [r[0] for r in rows]

    def add_items(self, items: list[NewsItem], run_id: str, canonical: dict[str, str],
                  status: str = "candidate", reasons: dict[str, str] | None = None) -> None:
        with self.db:
            for it in items:
                self.db.execute(
                    "INSERT OR IGNORE INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (it.id, it.title, it.url, canonical[it.id], it.source, it.published, it.summary,
                     it.kind, it.tier, int(it.fixture), run_id, now(), status,
                     (reasons or {}).get(it.id)),
                )

    def get_item(self, item_id: str) -> NewsItem | None:
        r = self.db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return _row_to_item(r) if r else None

    def search_items(self, query: str = "", limit: int = 20, since: str | None = None) -> list[NewsItem]:
        """在「还没写过的候选」里按关键词检索。第 4 周会换成 BM25 + 向量的混合检索。

        since：只看这个时间之后采集的（ISO 时间字符串）。"""
        sql = ("SELECT * FROM items WHERE status='candidate'"
               " AND id NOT IN (SELECT item_id FROM drafts)")
        args: list[Any] = []
        if since:
            sql += " AND collected_at>=?"
            args.append(since)
        for word in query.split():
            sql += " AND (title LIKE ? OR summary LIKE ?)"
            args += [f"%{word}%", f"%{word}%"]
        sql += " ORDER BY (tier='primary') DESC, published DESC LIMIT ?"
        args.append(limit)
        return [_row_to_item(r) for r in self.db.execute(sql, args)]

    # ---------- runs ----------
    def start_run(self, model: str) -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                            (run_id, now(), None, model, "running", "{}", "[]"))
        return run_id

    def finish_run(self, run_id: str, status: str, stats: dict[str, Any], trace: list[Any]) -> None:
        with self.db:
            self.db.execute(
                "UPDATE runs SET finished_at=?, status=?, stats_json=?, trace_json=? WHERE id=?",
                (now(), status, json.dumps(stats, ensure_ascii=False),
                 json.dumps(trace, ensure_ascii=False, default=str), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(r) if r else None

    def last_run_id(self, with_calls: bool = False) -> str | None:
        """最近一次运行。with_calls=True 时只看调用过模型的（跳过单独跑的 collect）。"""
        if with_calls:
            r = self.db.execute("SELECT run_id FROM llm_calls WHERE run_id IS NOT NULL"
                                " ORDER BY id DESC LIMIT 1").fetchone()
        else:
            r = self.db.execute("SELECT id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        return r[0] if r else None

    # ---------- drafts ----------
    def add_draft(self, run_id: str, item_id: str, draft: dict[str, Any], factcheck: dict[str, Any],
                  versions: dict[str, Any], flags: list[str]) -> str:
        draft_id = uuid.uuid4().hex[:8]
        with self.db:
            self.db.execute(
                "INSERT INTO drafts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (draft_id, run_id, item_id, draft["title"], draft["body"],
                 json.dumps(draft["claims"], ensure_ascii=False),
                 json.dumps(factcheck, ensure_ascii=False), json.dumps(versions, ensure_ascii=False),
                 json.dumps(flags, ensure_ascii=False), "pending", now(), None, None),
            )
        return draft_id

    def list_drafts(self, status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM drafts", []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY created_at DESC"
        return [_draft(r) for r in self.db.execute(sql, args)]

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        return _draft(r) if r else None

    def set_draft_status(self, draft_id: str, status: str, note: str = "") -> None:
        with self.db:
            self.db.execute("UPDATE drafts SET status=?, reviewed_at=?, review_note=? WHERE id=?",
                            (status, now(), note, draft_id))

    # ---------- 账单 ----------
    def record_call(self, run_id: str | None, task: str, model_name: str, model: str, usage: Any,
                    latency_ms: float, cost: float, currency: str, ok: bool, error: str = "") -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO llm_calls (run_id, task, model_name, model, input_tokens, output_tokens,"
                " cached_tokens, latency_ms, cost, currency, ok, error, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, task, model_name, model,
                 getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0),
                 getattr(usage, "cached_input_tokens", 0), latency_ms, cost, currency, int(ok), error, now()),
            )

    def cost_by_task(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT task, model_name, currency, COUNT(*) calls, SUM(input_tokens) input_tokens,"
               " SUM(output_tokens) output_tokens, SUM(cached_tokens) cached_tokens, SUM(cost) cost,"
               " AVG(latency_ms) avg_latency_ms, SUM(1-ok) errors FROM llm_calls")
        args: list[Any] = []
        if run_id:
            sql += " WHERE run_id=?"
            args.append(run_id)
        sql += " GROUP BY task, model_name, currency ORDER BY task"
        return [dict(r) for r in self.db.execute(sql, args)]

    def latencies(self, run_id: str | None = None) -> list[float]:
        sql, args = "SELECT latency_ms FROM llm_calls WHERE ok=1", []
        if run_id:
            sql += " AND run_id=?"
            args.append(run_id)
        return [r[0] for r in self.db.execute(sql, args)]


def _row_to_item(r: sqlite3.Row) -> NewsItem:
    return NewsItem(id=r["id"], title=r["title"], url=r["url"], source=r["source"],
                    published=r["published"], summary=r["summary"], kind=r["kind"], tier=r["tier"],
                    fixture=bool(r["fixture"]))


def _draft(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for key in ("claims_json", "factcheck_json", "versions_json", "flags_json"):
        d[key.removesuffix("_json")] = json.loads(d.pop(key) or "null")
    return d
