"""평가셋 형식 (eval/datasets/*.jsonl, 한 줄에 한 문항).

{"id": "expc-313107-m", "query": "...", "type": "situation", "as_of": null, "kinds": null,
 "source": "expc", "gold": [{"law_name": "주택임대차보호법", "article_no": "3의2", "grade": 2}]}

- 정답은 조문 단위 (법령 + 조문번호). 청크나 버전 ID 를 쓰면 재색인·개정 때 평가셋이 깨진다.
- law_id 가 있으면 그것으로, 없으면 law_name 으로 맞춘다 (불러올 때 law_id 로 바꾼다).
- grade: 2 = 핵심 조문, 1 = 관련 조문. nDCG 의 가중치로 쓴다.
"""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class Gold:
    article_no: str
    law_name: str | None = None
    law_id: str | None = None
    grade: int = 1

    def key(self) -> str | None:
        return f"{self.law_id}:{self.article_no}" if self.law_id else None


@dataclass
class EvalItem:
    id: str
    query: str
    gold: list[Gold]
    type: str = "general"
    source: str = ""
    as_of: date | None = None
    kinds: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def gains(self) -> dict[str, int]:
        """정답 조문 키 → 등급. law_id 로 못 바꾼 정답은 뺀다."""
        out: dict[str, int] = {}
        for g in self.gold:
            if (k := g.key()) is not None:
                out[k] = max(out.get(k, 0), g.grade)
        return out


def item_from_dict(d: Mapping[str, Any]) -> EvalItem:
    return EvalItem(
        id=str(d["id"]),
        query=str(d["query"]),
        gold=[Gold(**g) for g in d["gold"]],
        type=d.get("type", "general"),
        source=d.get("source", ""),
        as_of=date.fromisoformat(d["as_of"]) if d.get("as_of") else None,
        kinds=d.get("kinds"),
        meta=d.get("meta", {}),
    )


def load_jsonl(path: Path) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                try:
                    items.append(item_from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"{path}:{n}: 형식 오류 ({e})") from e
    return items


def save_jsonl(items: Iterable[EvalItem], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            d = asdict(it)
            d["as_of"] = it.as_of.isoformat() if it.as_of else None
            d["gold"] = [{k: v for k, v in g.items() if v is not None} for g in d["gold"]]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def resolve_law_ids(items: list[EvalItem], lookup: Callable[[str], str | None]) -> list[str]:
    """law_name 만 있는 정답에 law_id 를 채운다. 못 찾은 법령명 목록을 돌려준다."""
    missing: set[str] = set()
    for it in items:
        for g in it.gold:
            if g.law_id is None and g.law_name:
                law_id = lookup(g.law_name)
                if law_id is None:
                    missing.add(g.law_name)
                g.law_id = law_id
    return sorted(missing)
