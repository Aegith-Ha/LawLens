"""정답 후보 모으기 (pooling): 상황마다 여러 검색 방식의 상위 결과 + seed 를 합친다.

질의 소스(A~D)와 관계없이 쓰는 공통 단계. 결과(*.pool.jsonl)를 judge.py 가 등급 판정한다.

- 상황(case): 같은 정답을 공유하는 질의 묶음. 해석례 하나의 user · expert 질의가 한 상황이다.
- seed: 소스가 처음부터 가진 정답 후보 (해석례 인용 조문 등). pool 에 반드시 넣지만 판정은 똑같이 받는다.
- 후보 순서는 해시로 섞는다. 판정자가 검색 순위나 seed 여부를 짐작하지 못하게.

한 줄 = 한 상황:
{"case": "expc-338989", "need": "정보 요구 설명", "queries": [{"id", "type", "query"}],
 "seeds": {"001823:2": 2}, "candidates": [{"key", "law_id", "article_no", "law_name", "text",
 "seed": 2 | null, "best_rank": 1 | null, "found_by": ["hybrid/expert@1", ...]}]}
"""

import gzip
import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.orm import Session

from eval.dataset import EvalItem
from eval.golden.build_expc import _clean
from eval.golden.catalog import article_text
from lawlens.retrieval.hybrid import SearchMode
from lawlens.retrieval.service import SearchResponse

DEPTH = 20
MAX_TEXT = 2000  # 판정용 조문 본문 최대 글자 수 (긴 조문은 앞부분만)


class Service(Protocol):
    def search(
        self,
        query: str,
        limit: int = ...,
        as_of: Any = ...,
        kinds: list[str] | None = ...,
        mode: SearchMode = ...,
    ) -> SearchResponse: ...


@dataclass
class Candidate:
    key: str
    law_id: str
    article_no: str
    law_name: str
    text: str
    seed: int | None = None
    best_rank: int | None = None
    found_by: list[str] = field(default_factory=list)


@dataclass
class PoolCase:
    case: str
    need: str
    queries: list[dict[str, str]]
    seeds: dict[str, int]
    candidates: list[Candidate]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PoolCase:
        return cls(
            case=d["case"],
            need=d["need"],
            queries=d["queries"],
            seeds=d["seeds"],
            candidates=[Candidate(**c) for c in d["candidates"]],
        )


def case_id(item: EvalItem) -> str:
    """상황 ID. meta.case 가 있으면 그것, 없으면 문항 ID 의 마지막 "-유형" 을 뗀 것."""
    return str(item.meta.get("case") or item.id.rsplit("-", 1)[0])


def group_cases(items: Iterable[EvalItem]) -> dict[str, list[EvalItem]]:
    out: dict[str, list[EvalItem]] = {}
    for it in items:
        out.setdefault(case_id(it), []).append(it)
    return out


# ── 정보 요구 설명 (판정자가 보는 "이 상황에서 무엇을 찾는가") ─────────────────

NeedFn = Callable[[str, list[EvalItem]], str]

# 조문 번호는 지운다: 판정자가 해석례 인용 조문(seed)을 그대로 베끼지 않게. 법령명은 분야 파악에 필요해 남긴다.
_ARTICLE_REF = re.compile(
    r"제\s?\d+\s?조(?:의\s?\d+)?(?:\s?제\s?\d+\s?항)?(?:\s?제\s?\d+\s?호)?(?:\s?[가-하]목)?"
    r"|제\s?\d+\s?(?:항|호)|[가-하]목"
)


def strip_article_refs(s: str) -> str:
    return _ARTICLE_REF.sub("○○", s)


def need_from_queries(case: str, items: list[EvalItem]) -> str:
    """기본: 질의들을 그대로 정보 요구로 쓴다 (C · D 처럼 따로 설명이 없을 때)."""
    return "\n".join(f"- {it.query}" for it in items)


def need_from_meta(case: str, items: list[EvalItem]) -> str:
    """C 처럼 생성기가 상황 설명(meta.need)을 함께 만든 경우."""
    m = items[0].meta
    who = f"{m['persona']}: " if m.get("persona") else ""
    return f"{who}{m['need']}"


def need_from_expc(raw_dir: Path) -> NeedFn:
    """해석례: 안건명 · 질의요지 · 회답 (조문 번호는 지움). 이유는 인용 조문 목록이라 뺀다."""

    def need(case: str, items: list[EvalItem]) -> str:
        sid = case.split("-")[1]
        f = json.loads((raw_dir / f"{sid}.json").read_text("utf-8"))["ExpcService"]
        parts = [
            f"안건명: {_clean(str(f.get('안건명') or ''))}",
            f"[질의요지]\n{_clean(str(f.get('질의요지') or ''))}",
            f"[회답]\n{_clean(str(f.get('회답') or ''))}",
        ]
        return strip_article_refs("\n\n".join(parts))

    return need


# ── pool 만들기 ──────────────────────────────────────────────────────────────


def _order(key: str, case: str) -> str:
    return hashlib.md5(f"{case}:{key}".encode()).hexdigest()


def build_case(
    case: str,
    items: list[EvalItem],
    service: Service,
    session: Session,
    modes: list[SearchMode],
    depth: int,
    need: NeedFn,
) -> PoolCase:
    cands: dict[str, Candidate] = {}
    for it in items:
        short = it.type.rsplit("_", 1)[-1]
        for mode in modes:
            resp = service.search(it.query, limit=depth, as_of=it.as_of, kinds=it.kinds, mode=mode)
            for rank, a in enumerate([*resp.exact, *resp.results][:depth], start=1):
                key = f"{a.law_id}:{a.article_no}"
                c = cands.get(key)
                if c is None:
                    c = cands[key] = Candidate(key, a.law_id, a.article_no, a.law_name, "")
                c.found_by.append(f"{mode.value}/{short}@{rank}")
                c.best_rank = rank if c.best_rank is None else min(c.best_rank, rank)
    # pool 보조 질의 (meta.pool_queries): 평가에는 안 쓰고 후보만 더 모은다 (pool 편향 줄이기)
    extra = dict.fromkeys(q for it in items for q in it.meta.get("pool_queries") or [])
    for q in extra:
        for mode in modes:
            resp = service.search(
                q, limit=depth, as_of=items[0].as_of, kinds=items[0].kinds, mode=mode
            )
            for rank, a in enumerate([*resp.exact, *resp.results][:depth], start=1):
                key = f"{a.law_id}:{a.article_no}"
                c = cands.get(key)
                if c is None:
                    c = cands[key] = Candidate(key, a.law_id, a.article_no, a.law_name, "")
                c.found_by.append(f"{mode.value}/pool@{rank}")
    seeds: dict[str, int] = {}
    for it in items:
        for g in it.gold:
            if (k := g.key()) is None:
                continue
            seeds[k] = max(seeds.get(k, 0), g.grade)
            if k not in cands:
                assert g.law_id is not None
                cands[k] = Candidate(k, g.law_id, g.article_no, g.law_name or "", "")
    for k, c in list(cands.items()):
        c.seed = seeds.get(k)
        t = article_text(session, c.law_id, c.article_no)
        if t is None:  # 현행 DB 에 없는 조문 (삭제 · 폐지): 판정할 수 없으니 뺀다
            del cands[k]
            continue
        c.text = t if len(t) <= MAX_TEXT else t[:MAX_TEXT] + " …(생략)"
    ordered = sorted(cands.values(), key=lambda c: _order(c.key, case))
    queries = [{"id": it.id, "type": it.type, "query": it.query} for it in items]
    return PoolCase(case, need(case, items), queries, seeds, ordered)


def build(
    items: list[EvalItem],
    service: Service,
    session: Session,
    modes: list[SearchMode],
    depth: int = DEPTH,
    need: NeedFn = need_from_queries,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[PoolCase], dict[str, Any]]:
    groups = group_cases(items)
    out: list[PoolCase] = []
    for i, (case, its) in enumerate(sorted(groups.items()), start=1):
        out.append(build_case(case, its, service, session, modes, depth, need))
        if progress and (i % 10 == 0 or i == len(groups)):
            progress(f"{i}/{len(groups)}")
    sizes = [len(p.candidates) for p in out]
    seed_only = sum(1 for p in out for c in p.candidates if c.seed and not c.found_by)
    stats = {
        "cases": len(out),
        "queries": len(items),
        "candidates": sum(sizes),
        "per_case_avg": round(sum(sizes) / max(1, len(sizes)), 1),
        "per_case_max": max(sizes, default=0),
        "seeds": sum(len(p.seeds) for p in out),
        "seeds_not_retrieved": seed_only,
        "depth": depth,
        "modes": [m.value for m in modes],
    }
    return out, stats


def save(pool: list[PoolCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load(path: Path) -> list[PoolCase]:
    """*.pool.jsonl 또는 gzip 으로 압축한 *.pool.jsonl.gz (저장소에는 압축본만 커밋한다)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [PoolCase.from_dict(json.loads(ln)) for ln in f if ln.strip()]
