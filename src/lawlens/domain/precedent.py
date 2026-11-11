"""판례 도메인 모델."""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Precedent:
    prec_id: str  # 판례일련번호
    case_no: str  # 사건번호 (예: 2019다12345)
    case_name: str
    court: str
    decision_date: date | None
    case_type: str | None = None
    holdings: str | None = None  # 판시사항
    summary: str | None = None  # 판결요지
    full_text: str | None = None  # 전문
    ref_articles: list[str] = field(default_factory=list)  # 참조조문 원문
    ref_cases: list[str] = field(default_factory=list)  # 참조판례 원문
