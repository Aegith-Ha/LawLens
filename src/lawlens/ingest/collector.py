"""목록 조회 → 본문 조회 → raw 저장."""

import logging
from dataclasses import dataclass
from typing import Any

from lawlens.ingest.raw_store import RawStore
from lawlens.sources.lawgo.client import LawGoClient
from lawlens.sources.lawgo.targets import SPECS, Target

log = logging.getLogger(__name__)


@dataclass
class CollectStats:
    listed: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: int = 0


def collect(
    client: LawGoClient,
    store: RawStore,
    target: Target,
    query: str | None = None,
    max_pages: int | None = None,
    overwrite: bool = False,
    **search_params: Any,
) -> CollectStats:
    """search_params 는 목록 조회에 그대로 전달 (예: eflaw 의 nw="3")."""
    spec = SPECS.get(target)
    if spec is None:
        raise ValueError(f"아직 지원하지 않는 target: {target}")

    stats = CollectStats()
    for item in client.iter_search(target, query=query, max_pages=max_pages, **search_params):
        stats.listed += 1
        params = spec.service_params(item)
        if not params:
            log.warning("본문 조회 파라미터 부족: %s", item)
            stats.failed += 1
            continue
        doc_id = spec.doc_id(item)
        if not overwrite and store.exists(target, doc_id):
            stats.skipped += 1
            continue
        try:
            detail = client.get(target, **params)
        except Exception:
            log.exception("본문 조회 실패: %s %s", target, doc_id)
            stats.failed += 1
            continue
        store.save(target, doc_id, detail)
        stats.fetched += 1
    return stats
