"""검색 평가.

  # 평가셋 만들기
  uv run python -m scripts.eval build-expc                 # C: data/raw/expc → eval/datasets/expc.jsonl
  uv run python -m scripts.eval build-synth --n 300        # B: Claude 로 합성 (ANTHROPIC_API_KEY 필요, 유료)

  # 평가 실행 (결과: eval/reports/<시각>/report.md, items.jsonl)
  uv run python -m scripts.eval run eval/datasets/expc.jsonl eval/datasets/synth.jsonl
  uv run python -m scripts.eval run eval/datasets/expc.jsonl --modes hybrid,dense,sparse --sample 100

평가셋 형식은 eval/dataset.py 참고.
"""

import logging
import random
from pathlib import Path
from typing import Any

import typer

from eval.dataset import EvalItem, load_jsonl, resolve_law_ids, save_jsonl
from eval.golden.catalog import LawCatalog
from eval.scoring.embed_cache import DEFAULT_PATH as EMBED_CACHE
from eval.scoring.embed_cache import EmbeddingCache
from eval.scoring.runner import run as run_eval
from eval.scoring.runner import summarize, to_markdown, write_report
from lawlens.config import get_settings
from lawlens.retrieval.hybrid import HybridSearcher, SearchMode
from lawlens.storage.db import get_sessionmaker

app = typer.Typer(add_completion=False, no_args_is_help=True)
DATASETS = Path("eval/datasets")


def _quiet_logs() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def run(
    datasets: list[Path] = typer.Argument(..., help="평가셋 jsonl (여러 개 가능)"),
    modes: str = typer.Option("hybrid,dense,sparse", help="검색 방식 (쉼표로 조합)"),
    sample: int | None = typer.Option(None, help="데이터셋마다 무작위로 이만큼만 (빠른 확인용)"),
    seed: int = typer.Option(0, help="--sample 난수 시드"),
    out: Path = typer.Option(Path("eval/reports"), help="결과 폴더"),
    cache: Path = typer.Option(EMBED_CACHE, help="질의 임베딩 캐시 파일"),
    no_cache: bool = typer.Option(False, "--no-cache", help="질의 임베딩을 파일에 저장하지 않음"),
    refresh_cache: bool = typer.Option(
        False, "--refresh-cache", help="저장된 질의 임베딩을 무시하고 새로 계산해 덮어씀"
    ),
) -> None:
    """평가셋으로 검색해 Recall@k · MRR · nDCG 등을 계산한다."""
    _quiet_logs()
    from lawlens.api.deps import get_search_service

    mode_list = [SearchMode(m.strip()) for m in modes.split(",") if m.strip()]
    loaded: dict[str, list[EvalItem]] = {}
    with get_sessionmaker()() as session:
        catalog = LawCatalog.load(session)
    for path in datasets:
        items = load_jsonl(path)
        missing = resolve_law_ids(items, catalog.law_id)
        if missing:
            typer.echo(f"{path.name}: 법령명을 찾지 못한 정답 {len(missing)}개 (예: {missing[:3]})")
        if sample and len(items) > sample:
            items = random.Random(seed).sample(items, sample)
        loaded[path.stem] = items
        typer.echo(f"{path.name}: {len(items)}문항")

    service = get_search_service()
    typer.echo("검색 모델 불러오는 중…")
    searcher = service.searcher  # 모델 로딩 시간이 첫 문항 지연시간에 섞이지 않게 미리 불러온다
    embed_cache = None
    if isinstance(searcher, HybridSearcher):
        # 질의 임베딩을 방식(hybrid/dense/sparse)끼리, 그리고 다음 평가 때도 다시 쓴다
        inner = searcher.encoder
        namespace = (
            f"{get_settings().embedding_model}|max_length={getattr(inner, 'max_length', '')}"
        )
        embed_cache = EmbeddingCache(
            inner, namespace, None if no_cache else cache, refresh=refresh_cache
        )
        searcher.encoder = embed_cache
    try:
        results = run_eval(loaded, service, mode_list, progress=typer.echo, encoder=embed_cache)
    finally:
        if embed_cache is not None and isinstance(searcher, HybridSearcher):
            searcher.encoder = embed_cache.inner
            embed_cache.close()
    if embed_cache is not None:
        typer.echo(f"질의 임베딩: 캐시 {embed_cache.hits}회, 새로 계산 {embed_cache.misses}회")
    summaries = summarize(results)
    config = {
        "datasets": {k: len(v) for k, v in loaded.items()},
        "modes": [m.value for m in mode_list],
        "embedding_model": get_settings().embedding_model,
        "sample": sample,
        "seed": seed,
        "embedding_cache": None if no_cache else str(cache),
    }
    report_dir = write_report(results, summaries, out, config)
    typer.echo("\n" + to_markdown(summaries))
    typer.echo(f"\n결과: {report_dir}/report.md")


@app.command("build-expc")
def build_expc(
    raw_dir: Path = typer.Option(None, help="법령해석례 원본 폴더 (기본: data/raw/expc)"),
    min_date: str = typer.Option("20150101", help="이 날짜(YYYYMMDD) 이후 해석례만"),
    output: Path = typer.Option(DATASETS / "expc.jsonl", help="저장 경로"),
) -> None:
    """평가셋 C: 법령해석례 원본으로 만든다. 원본은 먼저 scripts.ingest --target expc 로 받는다."""
    from eval.golden.build_expc import build

    raw = raw_dir or get_settings().raw_dir / "expc"
    if not raw.exists():
        typer.echo(
            f"{raw} 이 없습니다. 먼저: uv run python -m scripts.ingest --target expc --sort ddes --max-pages 5"
        )
        raise typer.Exit(1)
    with get_sessionmaker()() as session:
        catalog = LawCatalog.load(session)
    items, stats = build(raw, catalog, min_date=min_date)
    n = save_jsonl(items, output)
    typer.echo(f"{stats}\n→ {output} ({n}문항 = 해석례 {stats['items']}건 × masked/raw)")


@app.command("build-synth")
def build_synth(
    n: int = typer.Option(300, help="뽑을 조문 수 (조문당 질문 최대 2개)"),
    seed: str = typer.Option("lawlens-eval-v1", help="표본 시드 (같으면 같은 조문)"),
    workers: int = typer.Option(4, help="동시 요청 수"),
    output: Path = typer.Option(DATASETS / "synth.jsonl", help="저장 경로"),
) -> None:
    """평가셋 B: 현행 조문을 뽑아 Claude 로 질문을 만든다 (유료, 조문 본문이 Anthropic API 로 전송됨)."""
    import anthropic

    from eval.golden.build_synth import MODEL, build, sample_articles

    _quiet_logs()
    # .env 의 ANTHROPIC_API_KEY 는 SDK 가 직접 읽지 않으므로 설정에서 넘긴다 (없으면 환경변수 · ant 로그인)
    secret = get_settings().anthropic_api_key
    key = secret.get_secret_value() if secret else None
    try:
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    except anthropic.AnthropicError as e:
        typer.echo(
            f"Anthropic 인증 정보가 없습니다 ({e}). .env 에 ANTHROPIC_API_KEY 를 넣어 주세요."
        )
        raise typer.Exit(1) from e
    with get_sessionmaker()() as session:
        samples = sample_articles(session, n, seed)
    typer.echo(f"조문 {len(samples)}개 → {MODEL} 로 질문 생성")
    items, stats = build(samples, client, workers=workers, progress=typer.echo)
    saved = save_jsonl(items, output)
    typer.echo(f"{stats}\n→ {output} ({saved}문항)")


@app.command("rewrite-expc")
def rewrite_expc(
    n: int = typer.Option(50, help="다시 쓸 해석례 수 (해석례당 질의 2개: user · expert)"),
    seed: str = typer.Option("lawlens-rewrite-v1", help="표본 시드 (같으면 같은 해석례)"),
    workers: int = typer.Option(4, help="동시 요청 수"),
    source: Path = typer.Option(DATASETS / "expc.jsonl", help="기존 평가셋 C (정답을 그대로 씀)"),
    output: Path = typer.Option(DATASETS / "expc_rewrite.jsonl", help="저장 경로"),
    answers: Path = typer.Option(
        None, help="API 대신 미리 써 둔 질의 JSON ({해석례 번호: {usable, reason, user, expert}})"
    ),
) -> None:
    """평가셋 C 질의를 Claude 로 다시 쓴다 (유료, 해석례 · 조문 본문이 Anthropic API 로 전송됨).

    --answers 를 주면 API 를 부르지 않고 그 파일의 질의를 같은 유출 검사 · 저장 과정에 넣는다.
    output 옆에 기존 질의와 비교하는 검토 문서(.review.md)도 만든다.
    """
    import json
    from functools import partial

    from eval.golden.rewrite_expc import (
        MODEL,
        build,
        from_answers,
        generate_one,
        load_cases,
        review_markdown,
        sample_cases,
    )

    _quiet_logs()
    if answers:
        gen, label = from_answers(answers), str(answers)
    else:
        gen, label = partial(generate_one, _anthropic_client()), MODEL
    raw = get_settings().raw_dir / "expc"
    picked = sample_cases(source, n, seed)
    with get_sessionmaker()() as session:
        cases = load_cases(picked, raw, session)
    typer.echo(f"해석례 {len(cases)}건 → {label} 로 질의 재작성")
    items, review, stats = build(cases, gen, workers=workers, progress=typer.echo)
    saved = save_jsonl(items, output)
    base = output.with_suffix("")
    base.with_suffix(".review.md").write_text(review_markdown(review, stats), "utf-8")
    with base.with_suffix(".review.jsonl").open("w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    typer.echo(f"{stats}\n→ {output} ({saved}문항), 검토: {base.with_suffix('.review.md')}")


def _anthropic_client() -> Any:
    import anthropic

    # .env 의 ANTHROPIC_API_KEY 는 SDK 가 직접 읽지 않으므로 설정에서 넘긴다 (없으면 환경변수 · ant 로그인)
    secret = get_settings().anthropic_api_key
    key = secret.get_secret_value() if secret else None
    try:
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    except anthropic.AnthropicError as e:
        typer.echo(
            f"Anthropic 인증 정보가 없습니다 ({e}). .env 에 ANTHROPIC_API_KEY 를 넣어 주세요."
        )
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
