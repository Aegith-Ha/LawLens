"""평가 도구 테스트 (DB · Milvus · Claude 없이)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eval.dataset import EvalItem, Gold, load_jsonl, resolve_law_ids, save_jsonl
from eval.golden.build_expc import build as build_expc
from eval.golden.build_expc import extract_refs, mask
from eval.golden.build_synth import Sample, _leaks
from eval.golden.catalog import LawCatalog
from eval.scoring.metrics import hit_at_k, ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from eval.scoring.runner import run, summarize, to_markdown
from lawlens.retrieval.hybrid import SearchMode
from lawlens.retrieval.service import ArticleResult, SearchResponse


def test_metrics() -> None:
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, {"b", "z"}, k=2) == 0.5
    assert reciprocal_rank(retrieved, {"c"}) == 1 / 3
    assert reciprocal_rank(retrieved, {"c"}, k=2) == 0.0
    assert reciprocal_rank(retrieved, {"z"}) == 0.0
    assert hit_at_k(retrieved, {"a"}, 1) == 1.0
    assert ndcg_at_k(["x", "y"], {"x": 2, "y": 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["y", "x"], {"x": 2, "y": 1}, 10) < 1.0
    assert ndcg_at_k(["x", "x"], {"x": 2}, 10) == pytest.approx(1.0)  # 중복은 한 번만
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4


def test_extract_refs_handles_expc_styles() -> None:
    t = (
        "「주택임대차보호법(이하 “법”이라 한다) 제3조의2」 및 제4조, 「동법 시행령 제10조제1항」, "
        "「민법」 제750조, 같은 법 제751조"
    )
    assert extract_refs(t) == [
        ("주택임대차보호법", "3의2"),
        ("주택임대차보호법", "4"),
        ("주택임대차보호법 시행령", "10"),
        ("민법", "750"),
        ("민법", "751"),
    ]
    assert extract_refs("같은 법 제3조") == []  # 앞에 나온 법령이 없으면 버린다


def test_mask_hides_law_names_and_articles() -> None:
    q = mask("「주택임대차보호법」 제3조의2 및 민법 제750조에 따라 임차인은 보증금을")
    assert "주택임대차보호법" not in q and "민법" not in q and "제750조" not in q
    assert q.endswith("임차인은 보증금을")


def _catalog() -> LawCatalog:
    cat = LawCatalog()
    cat.name_to_id = {"주택임대차보호법": "L1", "민법": "L2"}
    cat.current = {"L1:3의2", "L2:750"}
    return cat


def test_build_expc(tmp_path: Path) -> None:
    raw = {
        "ExpcService": {
            "법령해석례일련번호": "9",
            "해석일자": "20250101",
            "안건명": "주택임대차보호법 제3조의2 관련",
            "질의요지": "「주택임대차보호법」 제3조의2에 따라 임차인이 보증금을 돌려받을 수 있는지? 「민법」 제750조도",
            "회답": "「주택임대차보호법 제3조의2」에 따라 가능합니다.",
        }
    }
    (tmp_path / "9.json").write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
    old = {"ExpcService": {**raw["ExpcService"], "해석일자": "20000101"}}
    (tmp_path / "1.json").write_text(json.dumps(old, ensure_ascii=False), "utf-8")

    items, stats = build_expc(tmp_path, _catalog(), min_date="20150101")

    assert stats["old"] == 1 and stats["items"] == 1
    assert [it.type for it in items] == ["expc_masked", "expc_raw"]
    gains = items[0].gains()
    assert gains == {"L1:3의2": 2, "L2:750": 1}  # 회답에 나온 조문 = 핵심, 질의요지에만 = 관련
    assert "주택임대차보호법" not in items[0].query and "주택임대차보호법" in items[1].query


def test_synth_leak_filter() -> None:
    s = Sample(
        "L1",
        "주택임대차보호법",
        "법률",
        "3의2",
        "제3조의2(보증금의 회수)",
        "임차인이 임차주택에 대하여 보증금반환청구소송의 확정판결에 따라 경매를 신청하는 경우",
    )
    assert _leaks("주택임대차보호법상 보증금은?", s)
    assert _leaks("제3조의2 내용이 궁금해요", s)
    assert _leaks("보증금반환청구소송의 확정판결에 따라 경매를 신청하는 경우 어떻게 되나요", s)
    assert not _leaks(
        "집주인이 전세금을 안 돌려줘서 소송에서 이겼는데 집을 경매에 넘길 수 있나요?", s
    )


def test_dataset_roundtrip_and_resolve(tmp_path: Path) -> None:
    it = EvalItem("q1", "질문", [Gold("750", law_name="민 법", grade=2)], as_of=date(2026, 1, 1))
    p = tmp_path / "d.jsonl"
    assert save_jsonl([it], p) == 1
    [back] = load_jsonl(p)
    assert back.as_of == date(2026, 1, 1) and back.gold[0].law_id is None
    assert resolve_law_ids([back], _catalog().law_id) == []  # 공백 무시하고 "민법" 으로 찾음
    assert back.gains() == {"L2:750": 2}


class FakeService:
    def search(
        self,
        query: str,
        limit: int = 10,
        as_of: Any = None,
        kinds: Any = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> SearchResponse:
        def art(law_id: str, no: str) -> ArticleResult:
            return ArticleResult(
                f"{law_id}:{no}", "v", law_id, "법", "법률", no, "", [], "", None, None
            )

        order = (
            [art("L2", "750"), art("L1", "3의2")]
            if mode is SearchMode.DENSE
            else [art("L1", "3의2")]
        )
        return SearchResponse(query, date(2026, 1, 1), mode, results=order, took_ms=12)


def test_runner_summarizes_by_type_and_mode() -> None:
    items = [
        EvalItem("a", "q", [Gold("3의2", law_id="L1", grade=2)], type="situation"),
        EvalItem("b", "q", [Gold("750", law_id="L2")], type="term"),
        EvalItem("c", "q", [Gold("1", law_name="없는법")]),  # 정답을 못 바꾼 문항은 건너뜀
    ]
    results = run({"ds": items}, FakeService(), [SearchMode.HYBRID, SearchMode.DENSE])
    assert len(results) == 4
    summ = {(s.type, s.mode): s for s in summarize(results)}
    assert summ[("전체", "hybrid")].metrics["recall@5"] == 0.5
    assert summ[("situation", "dense")].metrics["mrr@10"] == 0.5
    assert summ[("term", "dense")].metrics["hit@1"] == 1.0
    assert "| ds | 전체 | hybrid | 2 |" in to_markdown(summarize(results))


def test_embedding_cache_reuses_across_runs(tmp_path: Path) -> None:
    from eval.scoring.embed_cache import EmbeddingCache
    from lawlens.indexing.encoder import Encoded

    calls: list[list[str]] = []

    class Inner:
        dim = 2

        def encode_documents(self, texts: list[str]) -> Encoded:
            raise AssertionError

        def encode_queries(self, texts: list[str]) -> Encoded:
            calls.append(texts)
            return Encoded(dense=[[0.25, -1.5]], sparse=[{7: 0.5}])

    db = tmp_path / "c.sqlite"
    enc = EmbeddingCache(Inner(), "m1", db)
    first = enc.encode_queries(["q1"])
    assert not enc.hit
    enc.encode_queries(["q1"])  # 같은 실행 안에서 다른 방식
    assert enc.hit and len(calls) == 1
    enc.close()

    again = EmbeddingCache(Inner(), "m1", db)  # 다음 평가
    got = again.encode_queries(["q1"])
    assert again.hit and len(calls) == 1
    assert got.dense == first.dense and got.sparse == first.sparse
    again.close()

    other = EmbeddingCache(Inner(), "m2", db)  # 모델이 바뀌면 새로 임베딩
    other.encode_queries(["q1"])
    assert not other.hit and len(calls) == 2
    other.close()

    fresh = EmbeddingCache(Inner(), "m1", db, refresh=True)  # --refresh-cache
    fresh.encode_queries(["q1"])
    assert not fresh.hit and len(calls) == 3
    fresh.encode_queries(["q1"])  # 같은 실행 안에서는 다시 씀
    assert fresh.hit and len(calls) == 3
    fresh.close()

    mem = EmbeddingCache(Inner(), "m1", None)  # --no-cache
    mem.encode_queries(["q1"])
    assert not mem.hit and len(calls) == 4


def test_synth_build_keep_and_types() -> None:
    from eval.golden.build_synth import build

    samples = [Sample(f"L{i}", f"가{i}법", "법률", "1", "제1조", "본문") for i in range(5)]

    def gen(s: Sample) -> dict[str, object] | None:
        if s.law_id == "L0":
            return {"usable": False, "situation": "", "term": ""}
        return {"usable": True, "situation": f"{s.law_id} 상황 질문", "term": f"{s.law_id} 용어"}

    items, stats = build(samples, gen, workers=2, keep=3)
    assert stats["unusable"] == 1 and stats["kept_articles"] == 3 and stats["items"] == 6
    assert {it.type for it in items} == {"synth_situation", "synth_term"}
    assert all(it.gold[0].grade == 2 and it.source == "synth" for it in items)
    assert "L0" not in {it.gold[0].law_id for it in items}
