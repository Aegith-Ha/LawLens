"""법령 동기화: 전체 배치 · 일배치.

처리 단위는 법령 버전 (MST, 시행일자). 한 버전은 다음 순서로 처리된다.

    본문 조회(eflaw) → raw 해시 비교 → raw 저장 → 파싱 → DB 저장
    → 영향받은 법령의 유효기간 재계산 → Milvus 반영(reconcile_index)

법이 바뀌는 경우별 처리
    1. 개정·제정 공포      lsHstInf(regDt) 에 새 버전이 나옴 → 저장, 직전 버전 valid_to 확정
    2. 시행일 도래          이벤트 없음. 유효기간 [valid_from, valid_to) 필터로 자동 반영
    3. 기존 버전 데이터 정정 lsHstInf 에 옛 버전도 나옴 → 해시가 바뀌었으면 재저장·재색인
    4. 데이터 삭제          delHst(knd=1, delDt) → 해당 MST 의 버전 삭제
    5. 폐지                1 과 같음. 폐지 버전은 빈 유효기간이라 이후 현행 버전이 없음

모든 단계가 멱등이라 같은 날짜를 다시 돌려도 결과가 같다. 일배치는 늦게 등록되는 데이터를
놓치지 않도록 마지막 성공일 이전 며칠(overlap_days)을 겹쳐서 다시 확인한다.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from lawlens.domain.statute import make_version_key
from lawlens.indexing.indexer import Indexer
from lawlens.ingest.raw_store import RawStore
from lawlens.parsing.statute import parse_law
from lawlens.sources.lawgo.client import LawGoClient
from lawlens.sources.lawgo.targets import SPECS, Target
from lawlens.storage.repositories import LawRepository, SyncStateRepository
from lawlens.timeutil import date_range, parse_yyyymmdd, today_kst

log = logging.getLogger(__name__)

DAILY_JOB = "law_daily"
LAW_KIND = 1  # delHst knd: 법령


def content_hash(raw: dict[str, Any]) -> str:
    data = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class SyncStats:
    listed: int = 0
    skipped: int = 0  # 이미 DB 에 있어 본문 조회 생략 (전체 배치)
    fetched: int = 0
    unchanged: int = 0  # 조회했지만 내용이 같음
    saved: int = 0  # 새로 저장 또는 내용 변경
    deleted: int = 0
    failed: int = 0
    indexed: int = 0  # Milvus 재색인 (재임베딩)
    validity_updated: int = 0  # Milvus 유효기간만 갱신


class LawSync:
    def __init__(
        self,
        client: LawGoClient | None,
        store: RawStore,
        session: Session,
        indexer: Indexer | None = None,
        today: date | None = None,
    ) -> None:
        """indexer 가 None 이면 DB 까지만 반영하고 Milvus 는 건드리지 않는다.
        (나중에 reconcile_index 로 한꺼번에 반영 가능)
        client 가 None 이면 API 를 쓰지 않는 reconcile_index 만 쓸 수 있다."""
        self.client = client
        self.store = store
        self.session = session
        self.repo = LawRepository(session)
        self.state = SyncStateRepository(session)
        self.indexer = indexer
        self.today = today or today_kst()
        self.spec = SPECS[Target.EFLAW]

    # ── 전체 배치 ───────────────────────────────────────
    def full(
        self,
        nw: str = "2,3",
        query: str | None = None,
        max_pages: int | None = None,
        refetch: bool = False,
    ) -> SyncStats:
        """eflaw 목록 전체를 받아 저장한다.

        nw: 1 연혁, 2 시행예정, 3 현행. 기본은 시행예정+현행.
        refetch=False 면 DB 에 이미 있는 버전은 본문 조회를 건너뛴다 (변경은 일배치가 잡음).

        목록에 없는 버전을 지우는 정리 작업은 하지 않는다. 현행만 받는 경우 개정 전 버전이
        목록에서 빠지는 것이 정상이기 때문이다. 삭제는 delHst(일배치)로만 반영한다.
        """
        stats = SyncStats()
        affected: set[str] = set()
        existing = set() if refetch else self.repo.all_version_keys()

        for item in self._api.iter_search(Target.EFLAW, query=query, max_pages=max_pages, nw=nw):
            params = self.spec.service_params(item)
            eff = parse_yyyymmdd(params.get("efYd"))
            if params and eff and make_version_key(params["MST"], eff) in existing:
                stats.listed += 1
                stats.skipped += 1
                continue
            self._process_item(item, stats, affected, force=refetch)

        self._finalize(affected, stats)
        # 전체 범위를 받았다면 일배치는 오늘부터 이어서 하면 된다
        if query is None and max_pages is None and self.state.get(DAILY_JOB) is None:
            self.state.set(DAILY_JOB, self.today)
            self.session.commit()
        return stats

    # ── 일배치 ──────────────────────────────────────────
    def daily(
        self,
        since: date | None = None,
        until: date | None = None,
        overlap_days: int = 3,
    ) -> SyncStats:
        """since ~ until 의 변경을 하루씩 반영한다.

        since 를 주지 않으면 (마지막 성공일 - overlap_days) 부터, 기록이 없으면
        (until - overlap_days) 부터 시작한다. 하루가 끝날 때마다 진행 상태를 저장하므로
        중간에 실패해도 다음 실행 때 그 날짜부터 다시 한다.
        """
        end = until or self.today
        last = self.state.get(DAILY_JOB)
        start = since or ((last or end) - timedelta(days=overlap_days))

        stats = SyncStats()
        for d in date_range(start, end):
            ds = f"{d:%Y%m%d}"
            affected: set[str] = set()
            log.info("일배치 %s", ds)

            for item in self._api.iter_search(Target.LS_HST_INF, regDt=ds):
                self._process_item(item, stats, affected)

            for item in self._api.iter_search(Target.DEL_HST, knd=LAW_KIND, delDt=ds):
                mst = str(item.get("일련번호") or "").strip()
                if mst:
                    self._delete_mst(mst, stats, affected)

            self._finalize(affected, stats)
            if last is None or d > last:
                self.state.set(DAILY_JOB, d)
                last = d
            self.session.commit()
        return stats

    # ── Milvus 반영 ─────────────────────────────────────
    def reconcile_index(self, stats: SyncStats | None = None) -> SyncStats:
        """DB 와 Milvus 의 차이를 맞춘다.

        내용이 바뀐 버전은 raw 를 다시 파싱해 재색인하고, 유효기간만 바뀐 버전은
        재임베딩 없이 valid_from/valid_to 만 갱신한다.
        """
        stats = stats or SyncStats()
        if self.indexer is None:
            return stats
        for row in self.repo.pending_index():
            if row.indexed_hash != row.content_hash:
                raw = self.store.load(Target.EFLAW, row.version_key)
                version = parse_law(raw, mst=row.mst, effective_date=row.effective_date)
                self.indexer.index_version(version, row.valid_from, row.valid_to)
                stats.indexed += 1
            else:
                self.indexer.update_validity(row.version_key, row.valid_from, row.valid_to)
                stats.validity_updated += 1
            self.repo.mark_indexed(row)
            self.session.commit()
        return stats

    # ── 내부 ────────────────────────────────────────────
    @property
    def _api(self) -> LawGoClient:
        if self.client is None:
            raise RuntimeError("API 클라이언트 없이 생성된 LawSync 입니다")
        return self.client

    def _process_item(
        self,
        item: dict[str, Any],
        stats: SyncStats,
        affected: set[str],
        force: bool = False,
    ) -> None:
        stats.listed += 1
        params = self.spec.service_params(item)
        eff = parse_yyyymmdd(params.get("efYd"))
        if not params or eff is None:
            log.warning("본문 조회 파라미터 부족: %s", item)
            stats.failed += 1
            return
        mst = params["MST"]
        key = make_version_key(mst, eff)

        try:
            raw = self._api.get(Target.EFLAW, **params)
        except Exception:
            log.exception("본문 조회 실패: %s", key)
            stats.failed += 1
            return
        stats.fetched += 1

        h = content_hash(raw)
        if not force and self.repo.get_content_hash(key) == h:
            stats.unchanged += 1
            return

        self.store.save(Target.EFLAW, key, raw)
        try:
            version = parse_law(raw, mst=mst, effective_date=eff)
        except Exception:
            log.exception("파싱 실패: %s (raw 는 저장됨)", key)
            stats.failed += 1
            return

        self.repo.save_version(version, h)
        self.session.commit()
        affected.add(version.law.law_id)
        stats.saved += 1

    def _delete_mst(self, mst: str, stats: SyncStats, affected: set[str]) -> None:
        for key, law_id in self.repo.delete_by_mst(mst):
            if self.indexer is not None:
                self.indexer.delete_version(key)
            affected.add(law_id)
            stats.deleted += 1
        self.session.commit()

    def _finalize(self, affected: set[str], stats: SyncStats) -> None:
        for law_id in affected:
            self.repo.recompute_validity(law_id)
        self.session.commit()
        self.reconcile_index(stats)
