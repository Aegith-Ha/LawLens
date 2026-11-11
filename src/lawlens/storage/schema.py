"""PostgreSQL 테이블 정의 (원문 · 버전 · 관계의 기준 저장소).

Milvus 색인은 이 테이블과 raw 저장소에서 언제든 다시 만들 수 있어야 한다.

유효기간: 법령 버전마다 [valid_from, valid_to) 구간을 저장한다.
    valid_from = 시행일자
    valid_to   = 같은 법령의 다음 버전 시행일자 (없으면 9999-12-31, 폐지 버전은 valid_from 과 같음)
    현행 검색:  valid_from <= 오늘 < valid_to
시행예정 버전이 들어오면 직전 버전의 valid_to 가 정해지므로, 시행일이 지나면 별도 작업 없이
검색 결과가 바뀐다.
"""

from datetime import date, datetime

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lawlens.timeutil import MAX_DATE


class Base(DeclarativeBase):
    pass


class LawRow(Base):
    __tablename__ = "law"

    law_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    law_type: Mapped[str] = mapped_column(String(50))
    ministry: Mapped[str | None] = mapped_column(String(100))


class LawVersionRow(Base):
    __tablename__ = "law_version"

    version_key: Mapped[str] = mapped_column(String(48), primary_key=True)  # {MST}_{시행일자}
    mst: Mapped[str] = mapped_column(String(32), index=True)
    law_id: Mapped[str] = mapped_column(ForeignKey("law.law_id"), index=True)
    effective_date: Mapped[date]
    promulgation_date: Mapped[date | None]
    promulgation_no: Mapped[str | None] = mapped_column(String(32))
    revision_type: Mapped[str | None] = mapped_column(String(50))

    valid_from: Mapped[date] = mapped_column(index=True)
    valid_to: Mapped[date] = mapped_column(default=MAX_DATE, index=True)

    content_hash: Mapped[str] = mapped_column(String(64))  # raw 응답 해시
    fetched_at: Mapped[datetime]

    # Milvus 에 반영된 상태. content_hash · valid_* 와 다르면 재색인 대상
    indexed_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_valid_from: Mapped[date | None]
    indexed_valid_to: Mapped[date | None]


class ArticleRow(Base):
    __tablename__ = "article"
    __table_args__ = (UniqueConstraint("version_key", "article_no"),)

    article_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    version_key: Mapped[str] = mapped_column(
        ForeignKey("law_version.version_key", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int]  # 본문 내 순서
    article_no: Mapped[str] = mapped_column(String(24))
    title: Mapped[str | None] = mapped_column(String(300))
    path: Mapped[str | None] = mapped_column(Text)  # "제5편 채권 > 제5장 불법행위"
    text: Mapped[str] = mapped_column(Text)
    is_deleted: Mapped[bool] = mapped_column(default=False)


class SyncStateRow(Base):
    """배치 진행 상태. 일배치는 last_success_date 다음 날(겹침 포함)부터 이어서 처리한다."""

    __tablename__ = "sync_state"

    job: Mapped[str] = mapped_column(String(50), primary_key=True)
    last_success_date: Mapped[date]
    updated_at: Mapped[datetime]


class PrecedentRow(Base):
    __tablename__ = "precedent"

    prec_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    case_no: Mapped[str] = mapped_column(String(100), index=True)
    case_name: Mapped[str] = mapped_column(Text)
    court: Mapped[str] = mapped_column(String(100))
    decision_date: Mapped[date | None] = mapped_column(index=True)
    case_type: Mapped[str | None] = mapped_column(String(50))
    holdings: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    full_text: Mapped[str | None] = mapped_column(Text)


class ReferenceRow(Base):
    """조문↔조문, 판례→참조조문 등 인용 관계."""

    __tablename__ = "reference"
    __table_args__ = (
        Index("ix_reference_from", "from_type", "from_id"),
        Index("ix_reference_to", "to_type", "to_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    from_type: Mapped[str] = mapped_column(String(20))
    from_id: Mapped[str] = mapped_column(String(96))
    to_type: Mapped[str] = mapped_column(String(20))
    to_id: Mapped[str | None] = mapped_column(String(96))  # 해석 못 한 인용은 NULL
    raw_text: Mapped[str] = mapped_column(Text)


class TermRow(Base):
    """법령용어 사전."""

    __tablename__ = "term"

    term: Mapped[str] = mapped_column(String(200), primary_key=True)
    definition: Mapped[str | None] = mapped_column(Text)
    synonyms: Mapped[str | None] = mapped_column(Text)  # 쉼표 구분
