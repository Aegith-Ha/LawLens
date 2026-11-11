"""원본 응답 보관소: data/raw/{target}/{id}.json

파서가 바뀌어도 API 를 다시 호출하지 않고 재처리할 수 있도록 원본을 그대로 저장한다.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lawlens.sources.lawgo.targets import Target


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, target: Target, doc_id: str) -> Path:
        return self.root / target.value / f"{doc_id}.json"

    def exists(self, target: Target, doc_id: str) -> bool:
        return self.path(target, doc_id).exists()

    def save(self, target: Target, doc_id: str, data: dict[str, Any]) -> Path:
        p = self.path(target, doc_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return p

    def load(self, target: Target, doc_id: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.path(target, doc_id).read_text(encoding="utf-8"))
        return data

    def iter_ids(self, target: Target) -> Iterator[str]:
        d = self.root / target.value
        if d.exists():
            for p in sorted(d.glob("*.json")):
                yield p.stem
