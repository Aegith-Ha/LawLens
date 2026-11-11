"""Milvus hybrid_search: dense + sparse 를 RRF 로 결합."""

from dataclasses import dataclass
from datetime import date
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from lawlens.indexing.encoder import Encoder
from lawlens.indexing.milvus_schema import (
    DENSE_FIELD,
    LAW_COLLECTION,
    LAW_OUTPUT_FIELDS,
    SPARSE_FIELD,
)
from lawlens.timeutil import to_yyyymmdd, today_kst


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    fields: dict[str, Any]


def build_filter(
    as_of: date | None = None,
    law_types: list[str] | None = None,
    today: date | None = None,
) -> str:
    """as_of 시점에 유효했던 버전만 남긴다. as_of 가 없으면 오늘(KST) = 현행.

    유효기간 [valid_from, valid_to) 덕분에 과거 시점 검색도 같은 조건 하나로 처리된다.
    """
    d = to_yyyymmdd(as_of or today or today_kst())
    conds = [f"valid_from <= {d}", f"valid_to > {d}"]
    if law_types:
        quoted = ", ".join(f'"{t}"' for t in law_types)
        conds.append(f"law_type in [{quoted}]")
    return " and ".join(conds)


class HybridSearcher:
    def __init__(
        self,
        client: MilvusClient,
        encoder: Encoder,
        collection: str = LAW_COLLECTION,
        rrf_k: int = 60,
    ) -> None:
        self.client = client
        self.encoder = encoder
        self.collection = collection
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 50,
        as_of: date | None = None,
        law_types: list[str] | None = None,
    ) -> list[SearchHit]:
        enc = self.encoder.encode_queries([query])
        expr = build_filter(as_of, law_types)
        reqs = [
            AnnSearchRequest(
                data=[enc.dense[0]],
                anns_field=DENSE_FIELD,
                param={"metric_type": "COSINE", "params": {"ef": max(candidates, 64)}},
                limit=candidates,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[enc.sparse[0]],
                anns_field=SPARSE_FIELD,
                param={"metric_type": "IP"},
                limit=candidates,
                expr=expr,
            ),
        ]
        res = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=reqs,
            ranker=RRFRanker(self.rrf_k),
            limit=limit,
            output_fields=LAW_OUTPUT_FIELDS,
        )
        return [
            SearchHit(chunk_id=h["id"], score=h["distance"], fields=h.get("entity", {}))
            for h in res[0]
        ]
