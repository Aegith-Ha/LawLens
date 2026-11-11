"""원본 수집.

예)
  uv run python -m scripts.ingest --query 민법 --max-pages 1          # 현행 법령 (eflaw, nw=3)
  uv run python -m scripts.ingest --query 민법 --nw 1,2,3             # 연혁 · 시행예정 포함
  uv run python -m scripts.ingest --target prec --query 손해배상 --max-pages 1
"""

import logging

import typer

from lawlens.config import get_settings
from lawlens.ingest.collector import collect
from lawlens.ingest.raw_store import RawStore
from lawlens.sources.lawgo.client import LawGoClient
from lawlens.sources.lawgo.targets import Target

app = typer.Typer(add_completion=False)


@app.command()
def main(
    target: Target = typer.Option(Target.EFLAW, help="수집 대상"),
    query: str | None = typer.Option(None, help="검색어 (없으면 전체)"),
    nw: str = typer.Option(
        "3", help="eflaw 전용. 1: 연혁, 2: 시행예정, 3: 현행 (쉼표로 조합, 예: 1,2,3)"
    ),
    max_pages: int | None = typer.Option(None, help="최대 페이지 수 (테스트용)"),
    overwrite: bool = typer.Option(False, help="이미 받은 문서도 다시 받기"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx 는 요청 URL 전체(OC 키 포함)를 INFO 로 남기므로 끈다
    logging.getLogger("httpx").setLevel(logging.WARNING)

    search_params = {"nw": nw} if target is Target.EFLAW else {}
    settings = get_settings()
    store = RawStore(settings.raw_dir)
    with LawGoClient(settings) as client:
        stats = collect(
            client,
            store,
            target,
            query=query,
            max_pages=max_pages,
            overwrite=overwrite,
            **search_params,
        )
    typer.echo(f"{target}: {stats}")


if __name__ == "__main__":
    app()
