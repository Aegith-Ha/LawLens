"""평가셋 작성에 쓰는 법령 목록 (PostgreSQL)."""

import re
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from lawlens.timeutil import today_kst

_SPACES = re.compile(r"\s+")


def norm_name(name: str) -> str:
    """법령명 비교용: 공백 제거 ("관한특별법" 과 "관한 특별법" 을 같게)."""
    return _SPACES.sub("", name)


@dataclass
class LawCatalog:
    name_to_id: dict[str, str] = field(default_factory=dict)  # 정규화한 이름 → law_id
    current: set[str] = field(default_factory=set)  # 현행 조문 키 "{law_id}:{article_no}"

    def law_id(self, name: str) -> str | None:
        return self.name_to_id.get(norm_name(name))

    def has_article(self, law_id: str, article_no: str) -> bool:
        return f"{law_id}:{article_no}" in self.current

    @classmethod
    def load(cls, session: Session, as_of: date | None = None) -> LawCatalog:
        d = as_of or today_kst()
        cat = cls()
        for law_id, name in session.execute(text("select law_id, name from law")):
            cat.name_to_id.setdefault(norm_name(name), law_id)
        rows = session.execute(
            text(
                "select v.law_id, a.article_no from article a"
                " join law_version v using (version_key)"
                " where v.valid_from <= :d and v.valid_to > :d and not a.is_deleted"
            ),
            {"d": d},
        )
        cat.current = {f"{law_id}:{no}" for law_id, no in rows}
        return cat


def article_label(article_no: str) -> str:
    """ "3의2" → "제3조의2"."""
    main, _, branch = article_no.partition("의")
    return f"제{main}조의{branch}" if branch else f"제{main}조"


def article_text(
    session: Session, law_id: str, article_no: str, as_of: date | None = None
) -> str | None:
    """현행(또는 as_of 시점) 조문을 "[법령명 제N조(제목)]\n본문" 으로. 없으면 None."""
    row = session.execute(
        text(
            "select l.name, coalesce(a.title, ''), a.text"
            " from article a join law_version v using (version_key) join law l using (law_id)"
            " where v.law_id = :lid and a.article_no = :no"
            " and v.valid_from <= :d and v.valid_to > :d and not a.is_deleted limit 1"
        ),
        {"lid": law_id, "no": article_no, "d": as_of or today_kst()},
    ).first()
    if row is None:
        return None
    name, title, body = row
    return f"[{name} {article_label(article_no)}{f'({title})' if title else ''}]\n{body}"
