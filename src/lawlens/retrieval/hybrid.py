"""Milvus 검색: dense + sparse 를 가중 RRF 로 결합 (하이브리드). 비교용으로 단일 방식도 지원한다.

결합 설정은 평가셋(dev)으로 정했다 (eval/README.md "검색 결합 설정"):
RRF k=30, dense 비중 0.6. 동일 가중 RRF(k=60) 보다 일반인 질의(C)에서 nDCG@10 이 0.035 높고
해석례(A)에서는 차이가 없다. Milvus 의 RRFRanker 는 가중치를 받지 않아서, dense · sparse 를
따로 검색해 여기서 결합한다.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from pymilvus import MilvusClient

from lawlens.indexing.encoder import Encoder
from lawlens.indexing.milvus_schema import (
    DENSE_FIELD,
    LAW_COLLECTION,
    LAW_OUTPUT_FIELDS,
    SPARSE_FIELD,
)
from lawlens.timeutil import to_yyyymmdd, today_kst


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


# 화면에서 고르는 법령 종류 → Milvus 조건.
# law_type 값은 "법률", "대통령령", "국토교통부령", "대법원규칙" 처럼 다양해서 묶어서 거른다.
LAW_KINDS: dict[str, str] = {
    "헌법": 'law_type == "헌법"',
    "법률": 'law_type == "법률"',
    "대통령령": 'law_type in ["대통령령", "대통령긴급명령"]',
    "총리령·부령": ('law_type like "%령" and law_type not in ["대통령령", "대통령긴급명령"]'),
    "규칙": 'law_type like "%규칙"',
}


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    fields: dict[str, Any]


def quote(value: str) -> str:
    """Milvus 필터 문자열 리터럴."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_filter(
    as_of: date | None = None,
    law_types: list[str] | None = None,
    today: date | None = None,
    kinds: list[str] | None = None,
) -> str:
    """as_of 시점에 유효했던 버전만 남긴다. as_of 가 없으면 오늘(KST) = 현행.

    유효기간 [valid_from, valid_to) 덕분에 과거 시점 검색도 같은 조건 하나로 처리된다.
    """
    d = to_yyyymmdd(as_of or today or today_kst())
    conds = [f"valid_from <= {d}", f"valid_to > {d}"]
    if law_types:
        conds.append(f"law_type in [{', '.join(quote(t) for t in law_types)}]")
    if kinds:
        unknown = set(kinds) - LAW_KINDS.keys()
        if unknown:
            raise ValueError(f"알 수 없는 법령 종류: {sorted(unknown)}")
        conds.append("(" + " or ".join(f"({LAW_KINDS[k]})" for k in kinds) + ")")
    return " and ".join(conds)


RRF_K = 30
DENSE_WEIGHT = 0.6


def fuse_rrf(
    lists: list[list[SearchHit]], weights: list[float], k: int = RRF_K, limit: int = 10
) -> list[SearchHit]:
    """가중 RRF: 점수 = Σ 가중치 / (k + 순위). 같은 청크는 합치고, 결합 점수 내림차순."""
    score: dict[str, float] = defaultdict(float)
    hit: dict[str, SearchHit] = {}
    for hits, w in zip(lists, weights, strict=True):
        for rank, h in enumerate(hits, start=1):
            score[h.chunk_id] += w / (k + rank)
            hit.setdefault(h.chunk_id, h)
    order = sorted(score, key=lambda c: -score[c])[:limit]
    return [SearchHit(chunk_id=c, score=score[c], fields=hit[c].fields) for c in order]


class HybridSearcher:
    def __init__(
        self,
        client: MilvusClient,
        encoder: Encoder,
        collection: str = LAW_COLLECTION,
        rrf_k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT,
    ) -> None:
        self.client = client
        self.encoder = encoder
        self.collection = collection
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 50,
        as_of: date | None = None,
        law_types: list[str] | None = None,
        kinds: list[str] | None = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[SearchHit]:
        enc = self.encoder.encode_queries([query])
        expr = build_filter(as_of, law_types, kinds=kinds)
        dense = {
            "data": [enc.dense[0]],
            "anns_field": DENSE_FIELD,
            "param": {"metric_type": "COSINE", "params": {"ef": max(candidates, 64)}},
        }
        sparse = {
            "data": [enc.sparse[0]],
            "anns_field": SPARSE_FIELD,
            "param": {"metric_type": "IP"},
        }
        if mode is SearchMode.HYBRID:
            lists = [self._search(r, expr, candidates) for r in (dense, sparse)]
            w = self.dense_weight
            return fuse_rrf(lists, [w, 1 - w], self.rrf_k, limit)
        return self._search(dense if mode is SearchMode.DENSE else sparse, expr, limit)

    def _search(self, req: dict[str, Any], expr: str, limit: int) -> list[SearchHit]:
        res = self.client.search(
            collection_name=self.collection,
            data=req["data"],
            anns_field=req["anns_field"],
            search_params=req["param"],
            filter=expr,
            limit=limit,
            output_fields=LAW_OUTPUT_FIELDS,
        )
        # pymilvus 3 의 Hit 은 기본 키를 "id" 가 아니라 필드 이름(chunk_id)으로 돌려준다
        return [
            SearchHit(chunk_id=h["chunk_id"], score=float(h["distance"]), fields=dict(h["entity"]))
            for h in res[0]
        ]
