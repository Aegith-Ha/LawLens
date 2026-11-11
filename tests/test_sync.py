"""전체 배치 · 일배치 시나리오 테스트 (가짜 API + SQLite + 가짜 Indexer)."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from lawlens.config import Settings
from lawlens.domain.statute import LawVersion
from lawlens.ingest.raw_store import RawStore
from lawlens.ingest.sync import DAILY_JOB, LawSync
from lawlens.sources.lawgo.client import LawGoClient
from lawlens.storage.db import get_sessionmaker, init_db
from lawlens.storage.repositories import LawRepository, SyncStateRepository
from lawlens.timeutil import MAX_DATE

LAW_ID = "000001"


def law_raw(efyd: str, body: str, rev: str = "일부개정") -> dict[str, Any]:
    return {
        "법령": {
            "기본정보": {
                "법령ID": LAW_ID,
                "법령명_한글": "테스트법",
                "법종구분": {"content": "법률"},
                "시행일자": efyd,
                "공포일자": efyd,
                "제개정구분": rev,
            },
            "조문": {
                "조문단위": [
                    {"조문여부": "조문", "조문번호": "1", "조문내용": f"제1조(목적) {body}"}
                ]
            },
        }
    }


@dataclass
class FakeLawApi:
    """lawSearch.do (eflaw / lsHstInf / delHst) 와 lawService.do (eflaw) 흉내."""

    versions: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    changes: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # regDt → 버전
    deletions: dict[str, list[str]] = field(default_factory=dict)  # delDt → MST
    detail_calls: int = 0

    def put(self, mst: str, efyd: str, body: str, reg_dt: str, rev: str = "일부개정") -> None:
        self.versions[(mst, efyd)] = law_raw(efyd, body, rev)
        self.changes.setdefault(reg_dt, []).append((mst, efyd))

    def delete(self, mst: str, del_dt: str) -> None:
        self.versions = {k: v for k, v in self.versions.items() if k[0] != mst}
        self.deletions.setdefault(del_dt, []).append(mst)

    def handler(self, req: httpx.Request) -> httpx.Response:
        p = req.url.params
        if req.url.path.endswith("lawService.do"):
            self.detail_calls += 1
            raw = self.versions.get((p["MST"], p["efYd"]))
            return httpx.Response(200, json=raw or {"Law": "일치하는 법령이 없습니다."})

        target = p["target"]
        if target == "eflaw":
            keys = list(self.versions)
        elif target == "lsHstInf":
            keys = self.changes.get(p["regDt"], [])
        elif target == "delHst":
            items = [{"일련번호": m, "구분명": "법령"} for m in self.deletions.get(p["delDt"], [])]
            return _list(target, items)
        else:
            raise AssertionError(target)
        return _list(target, [{"법령일련번호": m, "시행일자": e} for m, e in keys])


def _list(target: str, items: list[dict[str, str]]) -> httpx.Response:
    root = {"target": target, "totalCnt": str(len(items)), "law": items}
    return httpx.Response(200, json={"LawSearch": root})


@dataclass
class FakeIndexer:
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def index_version(self, version: LawVersion, valid_from: date, valid_to: date) -> int:
        self.calls.append(("index", version.version_key, valid_from, valid_to))
        return len(version.articles)

    def update_validity(self, version_key: str, valid_from: date, valid_to: date) -> None:
        self.calls.append(("validity", version_key, valid_from, valid_to))

    def delete_version(self, version_key: str) -> None:
        self.calls.append(("delete", version_key))

    def take(self) -> list[tuple[Any, ...]]:
        out, self.calls = self.calls, []
        return sorted(out, key=str)


@dataclass
class Env:
    api: FakeLawApi
    indexer: FakeIndexer
    session: Session
    sync: LawSync

    def validity(self) -> dict[str, tuple[date, date]]:
        rows = LawRepository(self.session).versions_of_law(LAW_ID)
        return {r.version_key: (r.valid_from, r.valid_to) for r in rows}


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    api = FakeLawApi()
    indexer = FakeIndexer()
    settings = Settings(law_api_oc="test", law_api_min_interval=0)
    with (
        get_sessionmaker(engine)() as session,
        LawGoClient(settings, transport=httpx.MockTransport(api.handler)) as client,
    ):
        sync = LawSync(
            client, RawStore(tmp_path / "raw"), session, indexer, today=date(2026, 11, 3)
        )
        yield Env(api, indexer, session, sync)


def test_daily_handles_amendment_correction_and_deletion(env: Env) -> None:
    api, sync, idx = env.api, env.sync, env.indexer
    d = date

    # 1) 현행 버전 등록
    api.put("1", "20250101", "v1", reg_dt="20261101")
    sync.daily(since=d(2026, 11, 1))
    assert env.validity() == {"1_20250101": (d(2025, 1, 1), MAX_DATE)}
    assert idx.take() == [("index", "1_20250101", d(2025, 1, 1), MAX_DATE)]
    assert SyncStateRepository(env.session).get(DAILY_JOB) == d(2026, 11, 3)

    # 2) 시행예정 개정 공포 → 직전 버전 valid_to 확정 (재임베딩 없이 유효기간만 갱신)
    api.put("2", "20270101", "v2", reg_dt="20261104")
    sync.today = d(2026, 11, 4)
    stats = sync.daily()  # 마지막 성공일 11/3 - 3일 = 10/31 부터 다시 확인
    assert stats.unchanged == 1  # 11/1 의 버전 1 은 다시 받았지만 그대로
    assert env.validity() == {
        "1_20250101": (d(2025, 1, 1), d(2027, 1, 1)),
        "2_20270101": (d(2027, 1, 1), MAX_DATE),
    }
    assert idx.take() == [
        ("index", "2_20270101", d(2027, 1, 1), MAX_DATE),
        ("validity", "1_20250101", d(2025, 1, 1), d(2027, 1, 1)),
    ]

    # 3) 같은 날 다시 돌려도 아무 변화 없음 (멱등)
    sync.daily()
    assert idx.take() == []

    # 4) 기존 버전 데이터 정정 → 재색인
    api.put("1", "20250101", "v1-정정", reg_dt="20261105")
    sync.today = d(2026, 11, 5)
    sync.daily()
    assert idx.take() == [("index", "1_20250101", d(2025, 1, 1), d(2027, 1, 1))]

    # 5) 시행예정 버전 삭제 → Milvus 삭제, 버전 1 은 다시 현행 끝까지 유효
    api.delete("2", del_dt="20261106")
    sync.today = d(2026, 11, 6)
    sync.daily()
    assert env.validity() == {"1_20250101": (d(2025, 1, 1), MAX_DATE)}
    assert idx.take() == [
        ("delete", "2_20270101"),
        ("validity", "1_20250101", d(2025, 1, 1), MAX_DATE),
    ]


def test_daily_repeal(env: Env) -> None:
    env.api.put("1", "20200101", "v1", reg_dt="20261101")
    env.api.put("2", "20261201", "", reg_dt="20261102", rev="폐지")
    env.sync.daily(since=date(2026, 11, 1))
    assert env.validity() == {
        "1_20200101": (date(2020, 1, 1), date(2026, 12, 1)),
        "2_20261201": (date(2026, 12, 1), date(2026, 12, 1)),  # 폐지 이후 현행 없음
    }


def test_full_skips_existing_versions(env: Env) -> None:
    env.api.put("1", "20250101", "v1", reg_dt="20261101")
    env.api.put("2", "20270101", "v2", reg_dt="20261101")

    first = env.sync.full()
    assert (first.saved, first.indexed) == (2, 2)
    assert SyncStateRepository(env.session).get(DAILY_JOB) == date(2026, 11, 3)

    calls_before = env.api.detail_calls
    second = env.sync.full()
    assert second.skipped == 2
    assert env.api.detail_calls == calls_before  # 본문 재조회 없음


def test_without_indexer_db_only_then_reindex(env: Env) -> None:
    env.api.put("1", "20250101", "v1", reg_dt="20261101")
    env.sync.indexer = None
    env.sync.full()
    assert env.indexer.calls == []

    env.sync.indexer = env.indexer
    stats = env.sync.reconcile_index()  # raw 를 다시 파싱해서 색인
    assert stats.indexed == 1
    assert env.indexer.take() == [("index", "1_20250101", date(2025, 1, 1), MAX_DATE)]
