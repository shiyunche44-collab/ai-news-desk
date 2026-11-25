"""命令行冒烟测试：每个命令都能用模拟模型跑通。"""

import pytest

from newsroom import cli


@pytest.fixture
def run(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda path=None: settings)

    def _run(*argv):
        code = cli.main(list(argv))
        return code, capsys.readouterr().out
    return _run


def test_full_cli_flow(run, settings):
    code, out = run("collect")
    assert code == 0 and "去重后留下 6 条" in out
    code, out = run("run", "--picks", "2")
    assert code == 0 and "产出草稿 2 篇" in out and "不会自动发布" in out
    code, out = run("review", "list")
    draft_id = out.split("[")[1].split("]")[0]
    code, out = run("review", "show", draft_id)
    assert "事实核查" in out and "视频号口播稿" in out
    code, out = run("review", "approve", draft_id)
    assert code == 1 and "--force" in out
    code, out = run("review", "approve", draft_id, "--force")
    assert code == 0 and list((settings.out_dir / "approved").glob("*.md"))
    code, out = run("cost")
    assert "researcher" in out and "writer" in out


def test_model_flag_after_subcommand(run):
    code, out = run("run", "--model", "deepseek")
    assert code == 2  # 没设 DEEPSEEK_API_KEY：给出清楚的提示，不崩


def test_eval_and_min_score(run):
    code, out = run("eval", "--min-score", "0.5")
    assert code == 0 and "| mock |" in out
    code, _ = run("eval", "--min-score", "1.01")
    assert code == 1


def test_summarize_and_react(run):
    code, out = run("summarize", "https://www.example-lab.cn/blog/example-7b", "--offline")
    assert code == 0 and '"source_type": "official"' in out
    code, out = run("summarize", "https://example-model.cn/news/price-cut", "--offline")
    assert "可疑指令" in out
    run("collect")
    code, out = run("react", "今天写哪条？")
    assert code == 0 and "Final Answer" in out


def test_review_export(run, settings, tmp_path):
    run("run", "--picks", "2")
    code, out = run("review", "export", "--out", str(tmp_path / "drafts"))
    files = sorted((tmp_path / "drafts").glob("*.md"))
    assert code == 0 and len(files) == 2
    text = files[0].read_text(encoding="utf-8")
    # 待审稿要带上提醒和逐条核查结论，成稿部分和 approve 导出的一样
    for part in ("## 提醒", "示例数据（虚构）", "## 事实核查", "## 正文", "## 来源", "## X", "## 视频号口播稿"):
        assert part in text
    code, _ = run("review", "export", "--status", "approved", "--out", str(tmp_path / "none"))
    assert code == 0 and not list((tmp_path / "none").glob("*.md"))
    # 再跑一次：--run latest 只导出最近这次运行的草稿，不带上前一次的
    _, out = run("run", "--picks", "1")
    n_new = int(out.split("产出草稿 ")[1].split(" 篇")[0])
    code, _ = run("review", "export", "--run", "latest", "--out", str(tmp_path / "latest"))
    assert code == 0 and n_new == 1 and len(list((tmp_path / "latest").glob("*.md"))) == 1
