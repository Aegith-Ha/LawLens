"""평가셋 C: 법령해석례(expc) → 질의요지 + 인용 조문.

법제처가 실제 질의에 근거 조문을 달아 답한 자료라 정답 품질이 좋다. 다만 질의요지에
정답 조문이 그대로 적혀 있어서 두 가지 질의를 만든다.

- expc_masked: 법령명 · 조문번호를 가린 질의 (의미 검색 평가용, 주 지표)
- expc_raw:    질의요지 원문 (조문 직접 지정 · 키워드 검색 평가용, 쉬운 편)

정답 등급: 안건명 · 회답에 나온 조문 = 2(핵심), 질의요지에만 나온 조문 = 1(관련).
지금 DB 의 현행 조문에 있는 정답만 남긴다 (해석 이후 조문 번호가 바뀌었을 수 있어 최근 해석례 권장).
"""

import json
import re
from pathlib import Path
from typing import Any

from eval.dataset import EvalItem, Gold
from eval.golden.catalog import LawCatalog

MAX_QUERY_CHARS = 1000
MIN_QUERY_CHARS = 30

# "같은 법", "동법", "같은 법 시행령", "동법시행령", "같은 영" 처럼 앞에 나온 법령을 가리키는 표현
_SAME_LAW = re.compile(r"^(?:같은|동)\s?(법|영|규칙)(?:\s?(시행령|시행규칙))?$")
_ART = r"제\s?(\d+)\s?조(?:\s?의\s?(\d+))?"
# 인용 형태: 「법령명 제N조…」, 「법령명」 제N조, 같은 법 제N조, (앞 인용에 이어) 및 제N조
_REF = re.compile(
    r"「\s*(?P<l1>[^」]*?)\s*"
    + _ART.replace("(\\d+)", "(?P<a1>\\d+)", 1).replace("(\\d+)", "(?P<b1>\\d+)", 1)
    + r"[^」]*」"
    r"|「(?P<l2>[^」]+)」\s*"
    + _ART.replace("(\\d+)", "(?P<a2>\\d+)", 1).replace("(\\d+)", "(?P<b2>\\d+)", 1)
    + r"|(?P<l3>(?:같은|동)\s?(?:법|영|규칙)(?:\s?(?:시행령|시행규칙))?)\s*"
    + _ART.replace("(\\d+)", "(?P<a3>\\d+)", 1).replace("(\\d+)", "(?P<b3>\\d+)", 1)
    + r"|(?P<conj>(?:및|또는|와|과|,|ㆍ|·)\s*)"
    + _ART.replace("(\\d+)", "(?P<a4>\\d+)", 1).replace("(\\d+)", "(?P<b4>\\d+)", 1)
)
_PAREN = re.compile(r"\([^)]*\)|（[^）]*）")
_BRACKET_LAW = re.compile(r"「[^」]{1,120}」")
_ARTICLE = re.compile(r"제\s?\d+\s?조(?:\s?의\s?\d+)?(?:\s?제\s?\d+\s?항)?(?:\s?제\s?\d+\s?호)?")
_LOOSE_LAW = re.compile(
    r"(?:(?:같은|동)\s?(?:법|영|규칙)(?:\s?(?:시행령|시행규칙))?|[가-힣·]{1,40}(?:법률|법|령|규칙)"
    r"(?:\s시행령|\s시행규칙)?)(?=\s*제\s?\d+\s?조)"
)
_WS = re.compile(r"[ \t ]+")


def _clean(s: str) -> str:
    lines = [_WS.sub(" ", ln).strip() for ln in s.replace("\r", "").split("\n")]
    return "\n".join(ln for ln in lines if ln)


_SUB_ITEM = re.compile(r"제\s?\d+\s?(?:항|호|목)|[가-하]\.?목|별표\s?\d+(?:의\d+)?")
_FOOTNOTE = re.compile(r"(\d{1,2}\))\1")  # HTML 각주 표시가 "1)1)" 로 남은 것
# 법령 약칭 정의: (이하 “식품표시광고법”이라 함) → 법령명이 새므로 지운다
_ALIAS = re.compile(r"\(이하\s*[“\"‘'][^”\"’']{1,40}[”\"’']\s*(?:이)?라\s*(?:함|한다|합니다)\)")


def mask(query: str) -> str:
    """법령명 · 조문번호를 가린다. 질의의 상황 설명만 남겨 의미 검색을 평가하려는 것."""
    art = _ARTICLE.pattern
    q = _ALIAS.sub("", _FOOTNOTE.sub("", query))
    q = re.sub(_BRACKET_LAW.pattern + r"\s*(?:" + art + r")", "관련 규정", q)  # 「법령」 제N조
    q = re.sub(r"(?:" + _LOOSE_LAW.pattern + r")\s*(?:" + art + r")", "관련 규정", q)  # 법령 제N조
    q = re.sub(r"「[^」]*" + art + r"[^」]*」", "관련 규정", q)  # 「법령 제N조」
    q = _BRACKET_LAW.sub("관련 법령", q)
    q = _ARTICLE.sub("해당 규정", q)
    q = _SUB_ITEM.sub("해당 규정", q)
    q = re.sub(
        r"(?:(?:관련|해당) (?:법령|규정)\s*)(?:(?:및|또는|,|ㆍ|·|와|과)?\s*(?:관련|해당) (?:법령|규정)\s*)+",
        "관련 규정 ",
        q,
    )
    return re.sub(r"[ \t]{2,}", " ", q).strip()


def extract_refs(text: str) -> list[tuple[str, str]]:
    """해석례 본문의 조문 인용 → [(법령명, 조문번호)]. 법령명이 확실한 인용만."""
    out: list[tuple[str, str]] = []
    last_law: str | None = None  # "같은 법" 이 가리키는 법령
    prev_law: str | None = None  # 바로 앞 인용의 법령 ("및 제6조" 용)
    prev_end = -1
    for m in _REF.finditer(text):
        if m["conj"] is not None:
            if prev_law is None or m.start() - prev_end > 3:
                continue
            law, a, b = prev_law, m["a4"], m["b4"]
        else:
            raw = m["l1"] if m["l1"] is not None else m["l2"] if m["l2"] is not None else m["l3"]
            raw = _PAREN.sub("", raw or "").strip()
            a = m["a1"] or m["a2"] or m["a3"]
            b = m["b1"] or m["b2"] or m["b3"]
            same = _SAME_LAW.match(raw.replace(" ", "")) or _SAME_LAW.match(raw)
            if same:
                if last_law is None:
                    prev_law = None
                    continue
                base = re.sub(r"\s?(시행령|시행규칙)$", "", last_law)
                suffix = same[2] or {"영": "시행령", "규칙": "시행규칙"}.get(same[1], "")
                law = f"{base} {suffix}".strip()
            elif raw:
                law = raw
                last_law = law
            else:
                prev_law = None
                continue
        out.append((law, f"{int(a)}의{int(b)}" if b else str(int(a))))
        prev_law, prev_end = law, m.end()
    return out


def golds(fields: dict[str, Any], catalog: LawCatalog) -> list[Gold]:
    grades: dict[tuple[str, str], tuple[str, int]] = {}
    for key, grade in (("안건명", 2), ("회답", 2), ("질의요지", 1)):
        txt = _clean(str(fields.get(key) or ""))
        for name, no in extract_refs(txt):
            law_id = catalog.law_id(name)
            if law_id is None or not catalog.has_article(law_id, no):
                continue
            prev = grades.get((law_id, no))
            if prev is None or prev[1] < grade:
                grades[(law_id, no)] = (name, grade)
    return [
        Gold(article_no=no, law_name=name, law_id=law_id, grade=g)
        for (law_id, no), (name, g) in grades.items()
    ]


def build(
    raw_dir: Path, catalog: LawCatalog, min_date: str = ""
) -> tuple[list[EvalItem], dict[str, int]]:
    """raw_dir: data/raw/expc. min_date: 'YYYYMMDD' 이후 해석례만."""
    stats = {"files": 0, "old": 0, "no_gold": 0, "short": 0, "items": 0}
    items: list[EvalItem] = []
    for path in sorted(raw_dir.glob("*.json")):
        stats["files"] += 1
        data = json.loads(path.read_text("utf-8"))
        f = data.get("ExpcService") or {}
        if min_date and str(f.get("해석일자") or "") < min_date:
            stats["old"] += 1
            continue
        query = _clean(str(f.get("질의요지") or ""))[:MAX_QUERY_CHARS]
        if len(query) < MIN_QUERY_CHARS:
            stats["short"] += 1
            continue
        gold = golds(f, catalog)
        if not gold:
            stats["no_gold"] += 1
            continue
        sid = str(f.get("법령해석례일련번호") or path.stem)
        meta = {
            "안건번호": f.get("안건번호"),
            "안건명": f.get("안건명"),
            "해석일자": f.get("해석일자"),
        }
        for typ, q in (("expc_masked", mask(query)), ("expc_raw", query)):
            items.append(
                EvalItem(
                    id=f"expc-{sid}-{typ.split('_')[1]}",
                    query=q,
                    gold=gold,
                    type=typ,
                    source="expc",
                    meta=meta,
                )
            )
        stats["items"] += 1
    return items, stats
