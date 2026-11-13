"""eflaw 본문 응답 → domain.statute.LawVersion.

응답 구조 (실제 응답 기준):
    법령
    ├─ 기본정보: 법령ID, 법령명_한글, 법종구분{content}, 소관부처{content}, 시행일자, 공포일자, ...
    └─ 조문
       └─ 조문단위[]
          ├─ 조문여부 "전문": 편·장·절·관 제목 (예: "제1장 총칙")
          │                  조문 없이 본문 한 덩어리로 된 법령(예: 표준시에 관한 법률)은
          │                  본문도 "전문"(조문번호 0)으로 온다 → 조문 하나(SINGLE_ARTICLE_NO)로 저장
          └─ 조문여부 "조문": 조문번호, 조문가지번호, 조문제목, 조문내용,
                              항 (객체 또는 목록) → 호 (객체 또는 목록) → 목 (객체 또는 목록)

본문에는 법령일련번호(MST)가 없으므로 호출하는 쪽에서 넘긴다.
"""

import logging
import re
from datetime import date
from typing import Any

from lawlens.domain.statute import SINGLE_ARTICLE_NO, Article, Law, LawVersion
from lawlens.parsing.text import clean_article_text
from lawlens.timeutil import parse_yyyymmdd

log = logging.getLogger(__name__)

_LEVELS = ("편", "장", "절", "관")
_DIVISION = re.compile(r"^제\s*\d+\s*(?:의\s*\d+\s*)?(편|장|절|관)")
_DELETED = re.compile(r"^제\d+조(?:의\d+)?\s*삭제(?:\s*[<\[][^>\]]*[>\]])*\s*$")
_WS = re.compile(r"\s+")


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _text(v: Any) -> str:
    """문자열 · 문자열 목록 · {content: ...} 를 모두 문자열로."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(t for t in (_text(x) for x in v) if t)
    if isinstance(v, dict):
        return _text(v.get("content"))
    return str(v)


def _sub_lines(unit: dict[str, Any]) -> list[str]:
    """항 → 호 → 목 내용을 순서대로 펼친다."""
    lines: list[str] = []
    for hang in _as_list(unit.get("항")):
        if not isinstance(hang, dict):
            lines.append(_text(hang))
            continue
        lines.append(_text(hang.get("항내용")))
        for ho in _as_list(hang.get("호")):
            if not isinstance(ho, dict):
                lines.append(_text(ho))
                continue
            lines.append(_text(ho.get("호내용")))
            for mok in _as_list(ho.get("목")):
                lines.append(_text(mok.get("목내용")) if isinstance(mok, dict) else _text(mok))
    return [t for t in (clean_article_text(x) for x in lines) if t]


def _article_no(unit: dict[str, Any]) -> str:
    no = str(unit.get("조문번호") or "").strip()
    branch = str(unit.get("조문가지번호") or "").strip()
    if branch.isdigit() and int(branch) > 0:
        return f"{no}의{int(branch)}"
    return no


def parse_law(raw: dict[str, Any], mst: str, effective_date: date | None = None) -> LawVersion:
    """effective_date 는 본문 조회에 쓴 efYd. 주어지면 응답의 시행일자보다 우선한다."""
    root = raw.get("법령")
    if not isinstance(root, dict):
        raise ValueError(f"법령 본문 형식이 아닙니다: {list(raw)[:3]}")
    info = root.get("기본정보") or {}

    eff = effective_date or parse_yyyymmdd(info.get("시행일자"))
    if eff is None:
        raise ValueError("시행일자를 알 수 없습니다")

    law = Law(
        law_id=_text(info.get("법령ID")).strip(),
        name=_text(info.get("법령명_한글")).strip(),
        law_type=_text(info.get("법종구분")).strip(),
        ministry=_text(info.get("소관부처")).strip() or None,
    )
    if not law.law_id:
        raise ValueError("법령ID 가 없습니다")

    articles: list[Article] = []
    seen: dict[str, int] = {}
    divisions: dict[str, str] = {}
    loose: list[str] = []  # 편·장·절·관 제목이 아닌 "전문" 본문

    for unit in _as_list((root.get("조문") or {}).get("조문단위")):
        if not isinstance(unit, dict):
            continue
        body = _WS.sub(" ", _text(unit.get("조문내용"))).strip()
        kind = unit.get("조문여부")

        if kind == "전문":
            m = _DIVISION.match(body)
            if m:
                level = _LEVELS.index(m.group(1))
                for lower in _LEVELS[level:]:
                    divisions.pop(lower, None)
                divisions[m.group(1)] = clean_article_text(body)
            elif body:
                loose.append(clean_article_text(body))
            continue
        if kind != "조문":
            continue

        no = _article_no(unit)
        if no in seen:  # 같은 번호가 두 번 나오면 구분자를 붙여 보존
            seen[no] += 1
            log.warning("중복 조문번호 %s (%s %s)", no, law.name, mst)
            no = f"{no}#{seen[no]}"
        else:
            seen[no] = 1

        lines = [clean_article_text(body), *_sub_lines(unit)]
        articles.append(
            Article(
                article_no=no,
                title=_text(unit.get("조문제목")).strip() or None,
                text="\n".join(t for t in lines if t),
                path=[divisions[lv] for lv in _LEVELS if lv in divisions],
                is_deleted=bool(_DELETED.match(body)) and not unit.get("항"),
            )
        )

    # 조문 없이 본문만 있는 법령: 본문 전체를 조문 하나로 저장 (없으면 검색에서 빠진다)
    if not articles and any(loose):
        articles.append(
            Article(
                article_no=SINGLE_ARTICLE_NO,
                title=None,
                text="\n".join(t for t in loose if t),
            )
        )

    return LawVersion(
        mst=str(mst),
        law=law,
        effective_date=eff,
        promulgation_date=parse_yyyymmdd(info.get("공포일자")),
        promulgation_no=_text(info.get("공포번호")).strip() or None,
        revision_type=_text(info.get("제개정구분")).strip() or None,
        articles=articles,
    )
