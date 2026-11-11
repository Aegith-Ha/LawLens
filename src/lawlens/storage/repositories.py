"""도메인 모델 ↔ 테이블 변환 및 조회."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from lawlens.domain.statute import LawVersion, is_repeal, make_article_id
from lawlens.storage.schema import (
    ArticleRow,
    LawRow,
    LawVersionRow,
    SyncStateRow,
)
from lawlens.timeutil import MAX_DATE, now_kst


@dataclass(frozen=True)
class VersionOrder:
    version_key: str
    mst: str
    effective_date: date
    promulgation_date: date | None
    revision_type: str | None


def compute_validity(versions: Iterable[VersionOrder]) -> dict[str, tuple[date, date]]:
    """한 법령의 버전 목록 → {version_key: (valid_from, valid_to)}.

    - 시행일자 순으로 정렬하고, 각 버전은 다음 버전의 시행일자까지 유효하다.
    - 시행일자가 같으면 공포일자 → MST 순으로 뒤에 오는 버전이 이긴다 (앞 버전은 빈 구간).
    - 폐지 버전은 빈 구간이다 (폐지 이후에는 현행 버전이 없음).
    """

    def order(v: VersionOrder) -> tuple[date, date, int]:
        mst_num = int(v.mst) if v.mst.isdigit() else 0
        return (v.effective_date, v.promulgation_date or date.min, mst_num)

    ordered = sorted(versions, key=order)
    out: dict[str, tuple[date, date]] = {}
    for i, v in enumerate(ordered):
        nxt = ordered[i + 1].effective_date if i + 1 < len(ordered) else MAX_DATE
        valid_to = v.effective_date if is_repeal(v.revision_type) else nxt
        out[v.version_key] = (v.effective_date, valid_to)
    return out


class LawRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    # ── 조회 ────────────────────────────────────────────
    def get_version(self, version_key: str) -> LawVersionRow | None:
        return self.s.get(LawVersionRow, version_key)

    def get_content_hash(self, version_key: str) -> str | None:
        row = self.get_version(version_key)
        return row.content_hash if row else None

    def all_version_keys(self) -> set[str]:
        return set(self.s.scalars(select(LawVersionRow.version_key)))

    def versions_of_law(self, law_id: str) -> Sequence[LawVersionRow]:
        return self.s.scalars(select(LawVersionRow).where(LawVersionRow.law_id == law_id)).all()

    def articles_of_version(self, version_key: str) -> Sequence[ArticleRow]:
        q = select(ArticleRow).where(ArticleRow.version_key == version_key).order_by(ArticleRow.seq)
        return self.s.scalars(q).all()

    # ── 저장 · 삭제 ─────────────────────────────────────
    def save_version(self, version: LawVersion, content_hash: str) -> None:
        """버전 하나를 통째로 저장(덮어쓰기). 유효기간은 recompute_validity 에서 다시 계산한다."""
        law = self.s.get(LawRow, version.law.law_id) or LawRow(law_id=version.law.law_id)
        law.name = version.law.name
        law.law_type = version.law.law_type
        law.ministry = version.law.ministry
        self.s.add(law)

        key = version.version_key
        row = self.get_version(key)
        if row is None:
            row = LawVersionRow(
                version_key=key,
                valid_from=version.effective_date,
                valid_to=MAX_DATE,
            )
        row.mst = version.mst
        row.law_id = version.law.law_id
        row.effective_date = version.effective_date
        row.promulgation_date = version.promulgation_date
        row.promulgation_no = version.promulgation_no
        row.revision_type = version.revision_type
        row.content_hash = content_hash
        row.fetched_at = now_kst()
        self.s.add(row)

        self.s.execute(delete(ArticleRow).where(ArticleRow.version_key == key))
        self.s.flush()
        for seq, art in enumerate(version.articles):
            self.s.add(
                ArticleRow(
                    article_id=make_article_id(key, art.article_no),
                    version_key=key,
                    seq=seq,
                    article_no=art.article_no,
                    title=art.title,
                    path=" > ".join(art.path) or None,
                    text=art.text,
                    is_deleted=art.is_deleted,
                )
            )
        self.s.flush()

    def delete_by_mst(self, mst: str) -> list[tuple[str, str]]:
        """MST 에 속한 모든 버전(시행일자별)을 지운다. [(version_key, law_id)] 반환."""
        rows = self.s.scalars(select(LawVersionRow).where(LawVersionRow.mst == mst)).all()
        removed = [(r.version_key, r.law_id) for r in rows]
        for key, _ in removed:
            self.s.execute(delete(ArticleRow).where(ArticleRow.version_key == key))
            self.s.execute(delete(LawVersionRow).where(LawVersionRow.version_key == key))
        self.s.flush()
        return removed

    # ── 유효기간 ────────────────────────────────────────
    def recompute_validity(self, law_id: str) -> None:
        rows = self.versions_of_law(law_id)
        validity = compute_validity(
            VersionOrder(
                r.version_key, r.mst, r.effective_date, r.promulgation_date, r.revision_type
            )
            for r in rows
        )
        for r in rows:
            r.valid_from, r.valid_to = validity[r.version_key]
        self.s.flush()

    # ── Milvus 반영 상태 ────────────────────────────────
    def pending_index(self) -> Sequence[LawVersionRow]:
        """Milvus 에 아직 반영되지 않은 버전 (내용 변경 또는 유효기간 변경)."""
        v = LawVersionRow
        q = select(v).where(
            or_(
                v.indexed_hash.is_(None),
                v.indexed_hash != v.content_hash,
                v.indexed_valid_from.is_(None),
                v.indexed_valid_from != v.valid_from,
                v.indexed_valid_to.is_(None),
                v.indexed_valid_to != v.valid_to,
            )
        )
        return self.s.scalars(q.order_by(v.version_key)).all()

    def mark_indexed(self, row: LawVersionRow) -> None:
        row.indexed_hash = row.content_hash
        row.indexed_valid_from = row.valid_from
        row.indexed_valid_to = row.valid_to
        self.s.flush()

    def reset_indexed(self) -> int:
        """전체 재색인이 필요할 때 (스키마 · 임베딩 모델 변경 등)."""
        rows = self.s.scalars(select(LawVersionRow)).all()
        for r in rows:
            r.indexed_hash = None
        self.s.flush()
        return len(rows)


class SyncStateRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, job: str) -> date | None:
        row = self.s.get(SyncStateRow, job)
        return row.last_success_date if row else None

    def set(self, job: str, d: date) -> None:
        row = self.s.get(SyncStateRow, job) or SyncStateRow(job=job)
        row.last_success_date = d
        row.updated_at = now_kst()
        self.s.add(row)
        self.s.flush()
