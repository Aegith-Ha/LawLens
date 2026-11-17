"""검색 서비스: 하이브리드 검색 결과를 조문 단위로 묶고, 조문을 직접 지정한 질의는 바로 찾아 준다.

"민법 제750조"            → 조문 바로가기(exact) + 하이브리드 검색 결과
"보증금을 안 돌려줘요"     → 하이브리드 검색 결과
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from pymilvus import MilvusClient

from lawlens.indexing.milvus_schema import LAW_COLLECTION, LAW_OUTPUT_FIELDS
from lawlens.retrieval.hybrid import SearchHit, SearchMode, build_filter, quote
from lawlens.retrieval.query import parse_query
from lawlens.timeutil import MAX_DATE, parse_yyyymmdd, today_kst

# 하이브리드 결과는 청크 단위라 같은 조문이 여러 번 나올 수 있어 넉넉히 받아 묶는다
_OVERFETCH = 3
_MAX_EXACT = 5


@dataclass
class ArticleResult:
    article_id: str
    version_key: str
    law_id: str
    law_name: str
    law_type: str
    article_no: str
    label: str  # "제750조(불법행위의 내용)"
    path: list[str]  # ["제3편 채권", "제5장 불법행위"]
    text: str  # 헤더를 뺀 본문 (가장 점수가 높은 청크)
    valid_from: date | None
    valid_to: date | None  # None = 현행 (다음 버전 없음)
    score: float = 0.0
    matched_chunks: int = 1


@dataclass
class SearchResponse:
    query: str
    as_of: date
    mode: SearchMode
    exact: list[ArticleResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # 질의에 있지만 그 시점에 없는 조문
    results: list[ArticleResult] = field(default_factory=list)
    took_ms: int = 0


def split_header(text: str) -> tuple[list[str], str, str]:
    """청크 text = "「법령명」 > 편 > 장 > 제N조(제목)\\n본문" → (path, label, 본문)."""
    head, _, body = text.partition("\n")
    parts = head.split(" > ")
    if len(parts) >= 2 and parts[0].startswith("「"):
        return parts[1:-1], parts[-1], body
    return [], "", text


def _date(v: Any) -> date | None:
    d = parse_yyyymmdd(str(v)) if v else None
    return None if d is None or d == MAX_DATE else d


def to_article(fields: dict[str, Any], score: float = 0.0) -> ArticleResult:
    path, label, body = split_header(fields.get("text", ""))
    return ArticleResult(
        article_id=fields["article_id"],
        version_key=fields["version_key"],
        law_id=fields["law_id"],
        law_name=fields["law_name"],
        law_type=fields["law_type"],
        article_no=fields["article_no"],
        label=label,
        path=path,
        text=body,
        valid_from=_date(fields.get("valid_from")),
        valid_to=_date(fields.get("valid_to")),
        score=score,
    )


def group_by_article(
    hits: list[SearchHit], limit: int, exclude: set[str] | None = None
) -> list[ArticleResult]:
    """청크 결과를 조문 단위로 묶는다. 순서는 조문의 가장 높은 청크 점수 기준.

    exclude 의 조문(이미 바로가기로 보여 준 조문)은 뺀다.
    """
    out: dict[str, ArticleResult] = {}
    for h in hits:  # 점수 내림차순으로 들어온다
        aid = h.fields["article_id"]
        if exclude and aid in exclude:
            continue
        if aid in out:
            out[aid].matched_chunks += 1
        elif len(out) < limit:
            out[aid] = to_article(h.fields, h.score)
    return list(out.values())


def _chunk_seq(chunk_id: str) -> int:
    tail = chunk_id.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 0


class Searcher(Protocol):
    def search(
        self,
        query: str,
        limit: int = ...,
        candidates: int = ...,
        as_of: date | None = ...,
        law_types: list[str] | None = ...,
        kinds: list[str] | None = ...,
        mode: SearchMode = ...,
    ) -> list[SearchHit]: ...


class SearchService:
    """searcher 를 바로 주거나, 무거운 경우 searcher_factory 로 처음 쓸 때 만든다."""

    def __init__(
        self,
        client: MilvusClient,
        searcher: Searcher | None = None,
        *,
        searcher_factory: Callable[[], Searcher] | None = None,
        collection: str = LAW_COLLECTION,
    ) -> None:
        if (searcher is None) == (searcher_factory is None):
            raise ValueError("searcher 와 searcher_factory 중 하나만 주어야 합니다")
        self.client = client
        self._searcher = searcher
        self._factory = searcher_factory
        self._lock = threading.Lock()
        self.collection = collection

    @property
    def searcher(self) -> Searcher:
        # 인코더(bge-m3) 로딩이 무거워서 처음 쓸 때 만든다.
        # 서버 시작 시 미리 불러오는 스레드와 첫 요청이 겹쳐도 한 번만 만든다.
        with self._lock:
            if self._searcher is None:
                assert self._factory is not None
                self._searcher = self._factory()
            return self._searcher

    def search(
        self,
        query: str,
        limit: int = 10,
        as_of: date | None = None,
        kinds: list[str] | None = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> SearchResponse:
        started = time.perf_counter()
        d = as_of or today_kst()
        resp = SearchResponse(query=query, as_of=d, mode=mode)
        resp.exact, resp.missing = self.find_articles(query, d, kinds)
        hits = self.searcher.search(
            query,
            limit=limit * _OVERFETCH,
            candidates=max(50, limit * _OVERFETCH),
            as_of=d,
            kinds=kinds,
            mode=mode,
        )
        resp.results = group_by_article(hits, limit, {a.article_id for a in resp.exact})
        resp.took_ms = round((time.perf_counter() - started) * 1000)
        return resp

    def find_articles(
        self, query: str, as_of: date, kinds: list[str] | None = None
    ) -> tuple[list[ArticleResult], list[str]]:
        """질의에 "법령명 제N조" 가 있으면 그 시점의 조문을 바로 찾는다. (찾은 것, 못 찾은 것)"""
        refs = [r for r in parse_query(query).article_refs if r.law_name][:_MAX_EXACT]
        out: list[ArticleResult] = []
        missing: list[str] = []
        base = build_filter(as_of, kinds=kinds)
        for ref in refs:
            assert ref.law_name is not None
            expr = (
                f"{base} and law_name == {quote(ref.law_name)}"
                f" and article_no == {quote(ref.article_no)}"
            )
            rows = self.client.query(self.collection, filter=expr, output_fields=LAW_OUTPUT_FIELDS)
            if not rows:
                missing.append(
                    f"{ref.law_name} 제{ref.article}조" + (f"의{ref.branch}" if ref.branch else "")
                )
                continue
            rows.sort(key=lambda r: _chunk_seq(r["chunk_id"]))
            art = to_article(rows[0])
            art.text = "\n".join(split_header(r["text"])[2] for r in rows)
            art.matched_chunks = len(rows)
            if all(a.article_id != art.article_id for a in out):
                out.append(art)
        return out, missing
