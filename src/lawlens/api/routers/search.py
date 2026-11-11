from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lawlens.retrieval.query import parse_query

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    as_of: date | None = None


@router.post("/parse")
def parse(req: SearchRequest) -> dict[str, object]:
    """쿼리 분석 결과 확인용 (디버그)."""
    pq = parse_query(req.query)
    return {
        "structured": pq.is_structured,
        "case_numbers": pq.case_numbers,
        "article_refs": [r.__dict__ for r in pq.article_refs],
    }


@router.post("/search")
def search(req: SearchRequest) -> dict[str, object]:
    # TODO(Phase 2): HybridSearcher 연결
    raise HTTPException(status_code=501, detail="Phase 2 에서 구현 예정")
