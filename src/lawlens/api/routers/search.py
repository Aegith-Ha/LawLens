from dataclasses import asdict
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from lawlens.api.deps import get_search_service, get_session
from lawlens.retrieval.hybrid import LAW_KINDS, SearchMode
from lawlens.retrieval.query import parse_query
from lawlens.retrieval.service import SearchService
from lawlens.storage.schema import ArticleRow, LawRow, LawVersionRow
from lawlens.timeutil import MAX_DATE, today_kst

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(10, ge=1, le=50)
    as_of: date | None = None  # 기준일. 없으면 오늘(현행)
    kinds: list[str] | None = None  # LAW_KINDS 의 키
    mode: SearchMode = SearchMode.HYBRID


@router.get("/meta")
def meta() -> dict[str, object]:
    """화면 초기값: 법령 종류 목록, 오늘 날짜."""
    return {"kinds": list(LAW_KINDS), "today": today_kst(), "modes": list(SearchMode)}


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
def search(
    req: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, Any]:
    if req.kinds and (unknown := set(req.kinds) - LAW_KINDS.keys()):
        raise HTTPException(422, f"알 수 없는 법령 종류: {sorted(unknown)}")
    resp = service.search(
        req.query.strip(), limit=req.limit, as_of=req.as_of, kinds=req.kinds, mode=req.mode
    )
    return asdict(resp)


@router.get("/articles/{article_id}")
def article(
    article_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    """조문 전체 본문과 버전 정보 (PostgreSQL)."""
    row = session.get(ArticleRow, article_id)
    if row is None:
        raise HTTPException(404, "조문을 찾을 수 없습니다")
    ver = session.get(LawVersionRow, row.version_key)
    law = session.get(LawRow, ver.law_id) if ver else None
    return {
        "article_id": row.article_id,
        "article_no": row.article_no,
        "title": row.title,
        "path": row.path,
        "text": row.text,
        "law_name": law.name if law else None,
        "law_type": law.law_type if law else None,
        "ministry": law.ministry if law else None,
        "effective_date": ver.effective_date if ver else None,
        "promulgation_date": ver.promulgation_date if ver else None,
        "promulgation_no": ver.promulgation_no if ver else None,
        "revision_type": ver.revision_type if ver else None,
        "valid_from": ver.valid_from if ver else None,
        "valid_to": None if ver is None or ver.valid_to == MAX_DATE else ver.valid_to,
    }
