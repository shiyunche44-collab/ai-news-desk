"""配置：NEWSROOM_SOURCES 切换信息源（GitHub Actions 定时跑时用真实信息源）。"""

import pytest

from newsroom.config import load_settings, select_sources

CFGS = [
    {"type": "fixture", "path": "data/fixtures/items.json"},
    {"type": "rss", "url": "https://example.com/rss", "enabled": False},
    {"type": "github", "query": "topic:llm-agent", "enabled": False},
]


def test_default_follows_enabled():
    assert [s["type"] for s in select_sources(CFGS)] == ["fixture"]
    assert [s["type"] for s in select_sources(CFGS, "  ")] == ["fixture"]


def test_env_picks_types_and_ignores_enabled():
    assert [s["type"] for s in select_sources(CFGS, "rss, github")] == ["rss", "github"]


def test_unknown_type_is_an_error():
    with pytest.raises(ValueError, match="weibo"):
        select_sources(CFGS, "github,weibo")


def test_does_not_mutate_config():
    cfgs = [dict(c) for c in CFGS]
    select_sources(cfgs, "github")[0]["query"] = "changed"
    assert cfgs[2]["query"] == "topic:llm-agent"


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("NEWSROOM_SOURCES", "github")
    assert [s["type"] for s in load_settings().sources] == ["github"]
