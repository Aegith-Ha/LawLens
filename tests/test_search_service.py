"""검색 서비스 · API 테스트 (Milvus 와 인코더는 가짜)."""

from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lawlens.api.deps import get_search_service
from lawlens.api.main import app
from lawlens.retrieval.hybrid import SearchHit, SearchMode, build_filter
from lawlens.retrieval.service import SearchService, group_by_article, split_header

HEADER = "「민법」 > 제3편 채권 > 제5장 불법행위 > 제750조(불법행위의 내용)"


def _fields(article_no: str, seq: int = 0, law: str = "민법", body: str = "본문") -> dict[str, Any]:
    vk = "1_20260317"
    return {
        "chunk_id": f"{vk}:{article_no}:{seq}",
        "article_id": f"{vk}:{article_no}",
        "version_key": vk,
        "law_id": "001706",
        "mst": "1",
        "law_name": law,
        "law_type": "법률",
        "article_no": article_no,
        "valid_from": 20260317,
        "valid_to": 99991231,
        "text": f"「{law}」 > 제{article_no}조\n{body}",
    }


class FakeSearcher:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kw: Any) -> list[SearchHit]:
        self.calls.append({"query": query, **kw})
        return self.hits


class FakeClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.filters: list[str] = []

    def query(self, collection: str, filter: str, output_fields: list[str]) -> list[dict[str, Any]]:
        self.filters.append(filter)
        return [r for r in self.rows if f'article_no == "{r["article_no"]}"' in filter]


def test_build_filter_kinds() -> None:
    expr = build_filter(today=date(2026, 11, 13), kinds=["법률", "규칙"])
    assert expr == (
        "valid_from <= 20261113 and valid_to > 20261113"
        ' and ((law_type == "법률") or (law_type like "%규칙"))'
    )
    with pytest.raises(ValueError):
        build_filter(kinds=["없는종류"])


def test_split_header() -> None:
    path, label, body = split_header(f"{HEADER}\n① 첫째 항\n② 둘째 항")
    assert path == ["제3편 채권", "제5장 불법행위"]
    assert label == "제750조(불법행위의 내용)"
    assert body == "① 첫째 항\n② 둘째 항"
    assert split_header("헤더 없는 본문") == ([], "", "헤더 없는 본문")


def test_group_by_article_keeps_best_chunk_and_counts() -> None:
    hits = [
        SearchHit("a:750:1", 0.9, _fields("750", 1, body="뒷부분")),
        SearchHit("a:751:0", 0.8, _fields("751")),
        SearchHit("a:750:0", 0.7, _fields("750", 0, body="앞부분")),
        SearchHit("a:752:0", 0.6, _fields("752")),
    ]
    out = group_by_article(hits, limit=2)
    assert [a.article_no for a in out] == ["750", "751"]
    assert out[0].text == "뒷부분" and out[0].matched_chunks == 2
    assert out[0].valid_to is None  # 99991231 = 현행
    # 바로가기로 이미 보여 준 조문은 관련 조문에서 뺀다
    assert [a.article_no for a in group_by_article(hits, 5, {"1_20260317:750"})] == ["751", "752"]


def test_search_finds_exact_article_and_reports_missing() -> None:
    rows = [_fields("750", 1, body="②항"), _fields("750", 0, body="①항")]
    client = FakeClient(rows)
    searcher = FakeSearcher([SearchHit("x", 0.5, _fields("751"))])
    svc = SearchService(client, searcher)  # type: ignore[arg-type]

    r = svc.search("민법 제750조, 민법 제9999조", limit=5, as_of=date(2026, 11, 13))

    assert [a.label for a in r.exact] == ["제750조"]
    assert r.exact[0].text == "①항\n②항"  # 청크 순서대로 합침
    assert r.missing == ["민법 제9999조"]
    assert 'law_name == "민법"' in client.filters[0]
    assert [a.article_no for a in r.results] == ["751"]
    assert searcher.calls[0]["limit"] == 15  # 조문 단위로 묶기 전 넉넉히


def test_search_api(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = FakeSearcher([SearchHit("x", 0.03, _fields("750"))])
    svc = SearchService(FakeClient([]), searcher)  # type: ignore[arg-type]
    app.dependency_overrides[get_search_service] = lambda: svc
    monkeypatch.setattr("lawlens.api.main._warm_up", lambda: None)
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/search",
                json={
                    "query": "불법행위",
                    "kinds": ["법률"],
                    "mode": "sparse",
                    "as_of": "2020-01-01",
                },
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["results"][0]["law_name"] == "민법"
            assert body["as_of"] == "2020-01-01"
            assert searcher.calls[0]["mode"] is SearchMode.SPARSE
            assert searcher.calls[0]["kinds"] == ["법률"]

            assert c.post("/api/search", json={"query": "x", "kinds": ["없음"]}).status_code == 422
            assert c.post("/api/search", json={"query": ""}).status_code == 422
            assert "법률" in c.get("/api/meta").json()["kinds"]
            assert c.get("/").headers["content-type"].startswith("text/html")
    finally:
        app.dependency_overrides.clear()
