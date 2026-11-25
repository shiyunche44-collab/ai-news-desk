"""读取 config.toml。

设计要点：
- 配置文件里只放「环境变量的名字」，key 本身只在运行时从环境变量读，仓库里永远没有密钥。
- 换模型只改配置或命令行参数，代码不动（方案文档里的「模型网关」）。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    name: str
    provider: str  # openai_compat | anthropic | mock
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    model_env: str | None = None
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    currency: str = "USD"
    # 命中上下文缓存的输入 token 单价；不填就按普通输入价算（偏保守）
    price_cached_input_per_m: float | None = None
    json_mode: bool = False
    # 是否把模型返回的 reasoning_content 在下一轮原样带回（DeepSeek 思考模式 + 工具调用需要）
    echo_reasoning: bool = False

    def resolved_model(self) -> str:
        """模型名可以被环境变量覆盖（比如 CLAUDE_MODEL）。"""
        if self.model_env and os.environ.get(self.model_env):
            return os.environ[self.model_env]
        return self.model

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class AgentConfig:
    max_steps: int = 8
    max_repeat_calls: int = 2
    token_budget: int = 60000
    max_retries: int = 2


@dataclass
class SafetyConfig:
    primary_domains: list[str] = field(default_factory=list)
    fetch_allowlist: list[str] = field(default_factory=list)
    fetch_max_chars: int = 12000


@dataclass
class Settings:
    root: Path
    db_path: Path
    out_dir: Path
    default_model: str
    voice_profile: Path
    models: dict[str, ModelConfig]
    routing: dict[str, str]
    agent: AgentConfig
    sources: list[dict[str, Any]]
    safety: SafetyConfig

    def model_for(self, task: str, override: str | None = None) -> str:
        """按「环节」选模型（模型分级）。override 是命令行 --model。"""
        routed = self.routing.get(task, "default")
        if routed != "default":
            return routed
        return override or os.environ.get("NEWSROOM_MODEL") or self.default_model


def select_sources(cfgs: list[dict[str, Any]], wanted: str | None = None) -> list[dict[str, Any]]:
    """选用哪些信息源。

    默认按 config.toml 里的 enabled。设了环境变量 NEWSROOM_SOURCES（逗号分隔的类型，
    比如 "rss,github"）就只用这几类、忽略 enabled：GitHub Actions 每天定时跑时用真实信息源，
    本机和测试仍然默认用示例数据，不用改配置文件。
    """
    if not wanted or not wanted.strip():
        return [dict(s) for s in cfgs if s.get("enabled", True)]
    types = {t.strip() for t in wanted.split(",") if t.strip()}
    known = {s["type"] for s in cfgs}
    unknown = types - known
    if unknown:
        raise ValueError(f"NEWSROOM_SOURCES 里有 config.toml 没配的类型：{'、'.join(sorted(unknown))}")
    return [dict(s) for s in cfgs if s["type"] in types]


def load_settings(path: str | Path | None = None) -> Settings:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.toml"
    root = cfg_path.resolve().parent
    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    app = raw.get("app", {})
    models = {
        name: ModelConfig(name=name, **body) for name, body in raw.get("models", {}).items()
    }
    sources = select_sources(raw.get("sources", []), os.environ.get("NEWSROOM_SOURCES"))
    for s in sources:
        if "path" in s:
            s["path"] = str((root / s["path"]).resolve())

    return Settings(
        root=root,
        db_path=root / app.get("db_path", "data/newsroom.db"),
        out_dir=root / app.get("out_dir", "out"),
        default_model=app.get("default_model", "mock"),
        voice_profile=root / app.get("voice_profile", "data/voice.md"),
        models=models,
        routing=raw.get("routing", {}),
        agent=AgentConfig(**raw.get("agent", {})),
        sources=sources,
        safety=SafetyConfig(**raw.get("safety", {})),
    )
