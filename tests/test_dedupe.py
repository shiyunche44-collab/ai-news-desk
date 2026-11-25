from newsroom.collect.dedupe import dedupe, jaccard, shingles
from newsroom.collect.sources import make_item

DOMAINS = ["example-lab.cn", "github.com"]


def item(title, url):
    return make_item(title, url, "test", DOMAINS)


def test_same_url_after_normalization_is_duplicate():
    a = item("tiny-agent 发布", "https://github.com/example-org/tiny-agent")
    b = item("另一个标题", "https://github.com/example-org/tiny-agent/?utm_source=x")
    kept, dropped = dedupe([a, b])
    assert len(kept) == 1 and dropped[0][1] == "网址重复"


def test_near_duplicate_keeps_primary_source():
    media = item("示例实验室开源Example-7B模型，支持32K上下文！", "https://news.example-media.cn/a")
    official = item("示例实验室开源 Example-7B 模型，支持 32K 上下文", "https://www.example-lab.cn/blog/x")
    kept, dropped = dedupe([media, official])  # 媒体那条排在前面也没关系
    assert [k.url for k in kept] == [official.url]
    assert dropped[0][0].url == media.url


def test_seen_titles_filter():
    a = item("示例实验室开源 Example-7B 模型", "https://www.example-lab.cn/blog/x")
    kept, dropped = dedupe([a], seen_titles=["示例实验室开源Example-7B模型"])
    assert not kept and dropped[0][1] == "和以前的内容近似重复"


def test_different_news_is_kept():
    a = item("示例实验室开源 Example-7B 模型", "https://www.example-lab.cn/a")
    b = item("tiny-agent：200 行的 agent 教学框架", "https://github.com/example-org/tiny-agent")
    kept, _ = dedupe([a, b])
    assert len(kept) == 2


def test_jaccard_bounds():
    assert jaccard(shingles("abc def"), shingles("abcdef")) == 1.0
    assert jaccard(set(), shingles("abc")) == 0.0
