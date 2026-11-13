"""동기화 CLI. 대상(law, prec)별로 따로 실행한다.

  uv run python -m scripts.sync law full                 # 현행+시행예정 전체 (최초 1회)
  uv run python -m scripts.sync law full --nw 1,2,3      # 연혁까지 전부
  uv run python -m scripts.sync law daily                # 일배치 (cron 으로 매일)
  uv run python -m scripts.sync law daily --since 20261101 --until 20261110
  uv run python -m scripts.sync law reindex              # DB ↔ Milvus 차이만 반영
  uv run python -m scripts.sync law reindex --all        # Milvus 전체 재색인

  uv run python -m scripts.sync prec full|daily          # 판례 (아직 구현 전)

--no-index 를 주면 PostgreSQL 까지만 반영하고 Milvus 는 건드리지 않는다
(나중에 reindex 로 한꺼번에 반영).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date

import typer

from lawlens.config import get_settings
from lawlens.indexing.indexer import Indexer
from lawlens.ingest.law_sync import LawSync, SyncStats
from lawlens.ingest.raw_store import RawStore
from lawlens.sources.lawgo.client import LawGoClient
from lawlens.storage.db import get_sessionmaker, init_db
from lawlens.storage.repositories import LawRepository
from lawlens.timeutil import parse_yyyymmdd

app = typer.Typer(add_completion=False, no_args_is_help=True)
law_app = typer.Typer(no_args_is_help=True, help="법령 (eflaw)")
prec_app = typer.Typer(no_args_is_help=True, help="판례 (prec) - 아직 구현 전")
app.add_typer(law_app, name="law")
app.add_typer(prec_app, name="prec")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx 는 요청 URL 전체(OC 키 포함)를 INFO 로 남기므로 끈다
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_indexer() -> Indexer:
    from lawlens.indexing.encoder import BGEM3Encoder
    from lawlens.indexing.milvus_schema import ensure_law_collection
    from lawlens.indexing.milvus_store import MilvusIndexer, get_milvus_client

    settings = get_settings()
    client = get_milvus_client(settings)
    ensure_law_collection(client, settings.dense_dim)
    return MilvusIndexer(client, BGEM3Encoder(settings.embedding_model))


@contextmanager
def _law_sync(no_index: bool, with_api: bool = True) -> Iterator[LawSync]:
    _setup_logging()
    settings = get_settings()
    init_db()
    indexer = None if no_index else _build_indexer()
    store = RawStore(settings.raw_dir)
    with get_sessionmaker()() as session:
        if with_api:
            with LawGoClient(settings) as client:
                yield LawSync(client, store, session, indexer)
        else:
            yield LawSync(None, store, session, indexer)


def _parse_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    d = parse_yyyymmdd(value)
    if d is None:
        raise typer.BadParameter(f"{name} 는 YYYYMMDD 형식이어야 합니다: {value}")
    return d


def _echo(stats: SyncStats) -> None:
    typer.echo(stats)


@law_app.command()
def full(
    nw: str = typer.Option("2,3", help="1: 연혁, 2: 시행예정, 3: 현행 (쉼표로 조합)"),
    query: str | None = typer.Option(None, help="법령명 검색어 (테스트용, 없으면 전체)"),
    max_pages: int | None = typer.Option(None, help="최대 페이지 수 (테스트용)"),
    refetch: bool = typer.Option(False, help="DB 에 있는 버전도 다시 조회"),
    no_index: bool = typer.Option(False, help="Milvus 반영 생략"),
) -> None:
    """전체 배치."""
    with _law_sync(no_index) as sync:
        _echo(sync.full(nw=nw, query=query, max_pages=max_pages, refetch=refetch))


@law_app.command()
def daily(
    since: str | None = typer.Option(None, help="시작일 YYYYMMDD (기본: 마지막 성공일 - 겹침)"),
    until: str | None = typer.Option(None, help="종료일 YYYYMMDD (기본: 오늘)"),
    overlap_days: int = typer.Option(3, help="마지막 성공일 이전 며칠을 다시 확인할지"),
    no_index: bool = typer.Option(False, help="Milvus 반영 생략"),
) -> None:
    """일배치: 법령 변경이력(lsHstInf) + 삭제 데이터(delHst) 반영."""
    with _law_sync(no_index) as sync:
        _echo(
            sync.daily(
                since=_parse_date(since, "since"),
                until=_parse_date(until, "until"),
                overlap_days=overlap_days,
            )
        )


@law_app.command()
def reindex(
    all_: bool = typer.Option(False, "--all", help="Milvus 전체 재색인 (스키마·모델 변경 시)"),
) -> None:
    """DB ↔ Milvus 차이 반영. API 는 호출하지 않는다."""
    with _law_sync(no_index=False, with_api=False) as sync:
        if all_:
            LawRepository(sync.session).reset_indexed()
            sync.session.commit()
        _echo(sync.reconcile_index())


def _not_implemented() -> None:
    typer.echo(
        "판례 동기화는 아직 구현되지 않았습니다. 원본만 받으려면 scripts.ingest --target prec"
    )
    raise typer.Exit(1)


@prec_app.command("full")
def prec_full() -> None:
    """판례 전체 배치 (구현 예정)."""
    _not_implemented()


@prec_app.command("daily")
def prec_daily() -> None:
    """판례 일배치 (구현 예정)."""
    _not_implemented()


if __name__ == "__main__":
    app()
