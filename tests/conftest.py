"""测试公共夹具：全部用模拟模型，不联网、不要 key。"""

from __future__ import annotations

import dataclasses

import pytest

from newsroom.config import load_settings
from newsroom.gateway.gateway import ModelGateway
from newsroom.store import Store


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # 防止你本机设过的环境变量影响测试结果
    for name in ("NEWSROOM_MODEL", "NEWSROOM_SOURCES", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    return dataclasses.replace(s, db_path=tmp_path / "test.db", out_dir=tmp_path / "out",
                               routing=dict(s.routing))


@pytest.fixture
def store(settings):
    return Store(settings.db_path)


@pytest.fixture
def gateway(settings):
    return ModelGateway(settings, sleep=lambda _: None)
