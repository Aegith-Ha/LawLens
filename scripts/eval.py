"""검색 평가.

  # 평가셋 만들기
  uv run python -m scripts.eval build-expc                 # C: data/raw/expc → eval/datasets/expc.jsonl
  uv run python -m scripts.eval build-synth --n 300        # B: Claude 로 합성 (ANTHROPIC_API_KEY 필요, 유료)

  # 정답 다듬기 (모든 소스 공통): 후보 모으기 → 등급 판정 → 사람 표본으로 검증
  uv run python -m scripts.eval pool eval/datasets/x.jsonl           # → x.pool.jsonl
  uv run python -m scripts.eval judge eval/datasets/x.pool.jsonl     # → x.judged.jsonl (새 정답 평가셋)
  uv run python -m scripts.eval judge-sample eval/datasets/x.pool.jsonl   # → x.human.jsonl
  uv run python -m scripts.eval judge-agree eval/datasets/x.human.jsonl

  # 평가 실행 (결과: eval/reports/<시각>/report.md, items.jsonl)
  uv run python -m scripts.eval run eval/datasets/expc.jsonl eval/datasets/synth.jsonl
  uv run python -m scripts.eval run eval/datasets/expc.jsonl --modes hybrid,dense,sparse --sample 100

평가셋 형식은 eval/dataset.py 참고.
"""

import json
import logging
import random
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _embedding_cache(service: Any, path: Path | None, refresh: bool) -> Iterator[Any]:
    """질의 임베딩을 검색 방식(hybrid/dense/sparse)끼리, 그리고 다음 실행 때도 다시 쓴다."""
    typer.echo("검색 모델 불러오는 중…")
    searcher = service.searcher  # 모델 로딩 시간이 첫 문항 지연시간에 섞이지 않게 미리 불러온다
    if not isinstance(searcher, HybridSearcher):
        yield None
        return
    inner = searcher.encoder
    namespace = f"{get_settings().embedding_model}|max_length={getattr(inner, 'max_length', '')}"
    cache = EmbeddingCache(inner, namespace, path, refresh=refresh)
    searcher.encoder = cache
    try:
        yield cache
    finally:
        searcher.encoder = cache.inner
        cache.close()
        typer.echo(f"질의 임베딩: 캐시 {cache.hits}회, 새로 계산 {cache.misses}회")


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
    with _embedding_cache(service, None if no_cache else cache, refresh_cache) as embed_cache:
        results = run_eval(loaded, service, mode_list, progress=typer.echo, encoder=embed_cache)
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
    keep: int = typer.Option(None, help="쓸 수 있는 조문을 표본 순서대로 이만큼만 남김"),
    seed: str = typer.Option("lawlens-eval-v1", help="표본 시드 (같으면 같은 조문)"),
    backend: str = typer.Option(
        "claude", help="생성기: claude (Anthropic API) | codex (Codex CLI)"
    ),
    model: str = typer.Option(None, help="생성 모델 (codex 만. 기본: codex 설정값)"),
    workers: int = typer.Option(4, help="동시 요청 수"),
    output: Path = typer.Option(DATASETS / "synth.jsonl", help="저장 경로"),
) -> None:
    """평가셋 B: 현행 조문을 뽑아 질문을 만든다. 정답 seed = 원래 조문 (pool · judge 로 다듬는다).

    조문 본문이 Anthropic API(claude, 유료) 또는 OpenAI(codex, 로그인한 계정 사용량)로 전송된다."""
    from functools import partial

    from eval.golden import build_synth as B

    _quiet_logs()
    if backend == "codex":
        gen = B.generate_codex(model, log_dir=Path(".cache/codex_synth"))
        label = f"codex:{model or 'default'}"
    elif backend == "claude":
        gen, label = partial(B.generate_one, _anthropic_client()), B.MODEL
    else:
        raise typer.BadParameter(f"backend: {backend}")
    with get_sessionmaker()() as session:
        samples = B.sample_articles(session, n, seed)
    typer.echo(f"조문 {len(samples)}개 → {label} 로 질문 생성")
    items, stats = B.build(samples, gen, workers=workers, progress=typer.echo, keep=keep)
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


@app.command()
def scenario(
    per_area: int = typer.Option(6, help="분야마다 만들 상황 수 (분야 15개)"),
    model: str = typer.Option(None, help="생성 모델 (codex. 기본: codex 설정값)"),
    workers: int = typer.Option(3, help="동시 요청 수"),
    output: Path = typer.Option(DATASETS / "scenario.jsonl", help="저장 경로"),
    answers: Path = typer.Option(
        None, help="생성기 대신 저장해 둔 생성 결과 JSON ({분야: [상황...]})"
    ),
) -> None:
    """평가셋 C: 일반인 상황 시나리오를 Codex 로 만든다 (조문을 보여 주지 않음). 정답은 pool · judge 로.

    생성 결과는 <이름>.answers.json 에 저장하고(--answers 로 다시 쓸 수 있음), 법령명 · 조문번호가
    들어간 질의는 빼며, 검토 문서(.review.md)를 만든다."""
    from eval.golden import scenario as S

    _quiet_logs()
    raw = output.with_name(output.name.removesuffix(".jsonl") + ".answers.json")
    if answers:
        gen, label = S.from_answers(answers), str(answers)
    else:
        gen = S.generate_codex(model, log_dir=Path(".cache/codex_scenario"))
        label = f"codex:{model or 'default'}"
    generated = S.generate_all(gen, per_area, workers, typer.echo)
    if not answers:
        raw.write_text(json.dumps(generated, ensure_ascii=False, indent=1) + "\n", "utf-8")
    with get_sessionmaker()() as session:
        names = S.law_names_for_leak(list(LawCatalog.load(session).name_to_id))
    items, stats, review = S.build(generated, names, label)
    save_jsonl(items, output)
    md = output.with_name(output.name.removesuffix(".jsonl") + ".review.md")
    md.write_text(S.review_markdown(review, stats), "utf-8")
    typer.echo(f"{stats}\n→ {output}, 생성 결과: {raw}, 검토: {md}")


@app.command()
def pool(
    dataset: Path = typer.Argument(..., help="평가셋 jsonl (질의 + seed 정답)"),
    depth: int = typer.Option(20, help="검색 방식 · 질의마다 상위 몇 개를 모을지"),
    modes: str = typer.Option("hybrid,dense,sparse", help="후보를 모을 검색 방식"),
    need: str = typer.Option(
        "auto", help="정보 요구 설명: expc(해석례 본문) | meta(meta.need) | query(질의) | auto"
    ),
    output: Path = typer.Option(None, help="저장 경로 (기본: <평가셋>.pool.jsonl)"),
    cache: Path = typer.Option(EMBED_CACHE, help="질의 임베딩 캐시 파일"),
) -> None:
    """정답 후보 모으기: 상황마다 여러 검색 방식의 상위 결과 + 기존 정답(seed)을 합친다."""
    from eval.golden import pool as P
    from lawlens.api.deps import get_search_service

    _quiet_logs()
    items = load_jsonl(dataset)
    mode_list = [SearchMode(m.strip()) for m in modes.split(",") if m.strip()]
    if need == "auto":
        if all(it.source == "expc" for it in items):
            need = "expc"
        else:
            need = "meta" if all(it.meta.get("need") for it in items) else "query"
    need_fn: P.NeedFn
    if need == "expc":
        need_fn = P.need_from_expc(get_settings().raw_dir / "expc")
    elif need == "meta":
        need_fn = P.need_from_meta
    else:
        need_fn = P.need_from_queries
    out = output or _sibling(dataset, "pool")
    service = get_search_service()
    with get_sessionmaker()() as session:
        resolve_law_ids(items, LawCatalog.load(session).law_id)
        with _embedding_cache(service, cache, False):
            cases, stats = P.build(items, service, session, mode_list, depth, need_fn, typer.echo)
    P.save(cases, out)
    typer.echo(f"{stats}\n→ {out}")


@app.command()
def judge(
    pool_path: Path = typer.Argument(..., help="pool 파일 (*.pool.jsonl)"),
    dataset: Path = typer.Option(None, help="원래 평가셋 (기본: pool 이름에서 .pool 을 뗀 것)"),
    answers: Path = typer.Option(
        None, help='API 대신 미리 판정한 JSON ({상황: {"2": [키], "1": [키]}})'
    ),
    backend: str = typer.Option(
        "claude", help="answers 가 없을 때 판정기: claude (Anthropic API) | codex (Codex CLI)"
    ),
    model: str = typer.Option(None, help="판정 모델 (codex 만. 기본: codex 설정값)"),
    tag: str = typer.Option(
        None, help="결과 파일 이름에 붙일 표시 (예: codex → <평가셋>.codex.judgments.jsonl)"
    ),
    workers: int = typer.Option(4, help="동시 요청 수"),
) -> None:
    """등급 판정 → <평가셋>.judgments.jsonl, .judged.jsonl(새 정답 평가셋), .judge.md(검토 문서).

    answers 가 없으면 backend 로 판정한다. claude 는 Anthropic API (유료), codex 는 로그인한
    Codex CLI 계정의 사용량을 쓴다. 둘 다 상황 설명 · 조문 본문이 외부로 전송된다.
    결과 파일에 이미 판정된 상황은 다시 판정하지 않는다 (중간에 끊겨도 이어서 돌릴 수 있다)."""
    from functools import partial

    from eval.golden import judge as J
    from eval.golden.pool import load as load_pool

    _quiet_logs()
    cases = load_pool(pool_path)
    base = _base(pool_path)
    dataset = dataset or base.with_name(base.name + ".jsonl")
    out = base.with_name(f"{base.name}.{tag}") if tag else base
    if answers:
        fn, label = J.from_answers(answers), str(answers)
    elif backend == "codex":
        fn = partial(J.judge_codex, model=model, log_dir=Path(".cache/codex_judge"))
        label = f"codex:{model or 'default'}"
    elif backend == "claude":
        fn, label = partial(J.judge_api, _anthropic_client()), J.MODEL
    else:
        raise typer.BadParameter(f"backend: {backend}")
    jpath = _sibling(out, "judgments")
    done = J.load_judgments(jpath) if jpath.exists() and not answers else {}
    todo = [p for p in cases if p.case not in done]
    n = sum(len(p.candidates) for p in todo)
    typer.echo(f"상황 {len(todo)}개 (이미 판정 {len(done)}개) · 후보 {n}개 → {label} 로 판정")
    judged, jstats = J.judge_all(todo, fn, workers=workers, progress=typer.echo)
    judged = {**done, **judged}
    jstats["cases"], jstats["judged"] = len(cases), len(judged)
    J.save_judgments(cases, judged, jpath, label)
    items = load_jsonl(dataset)
    with get_sessionmaker()() as session:
        resolve_law_ids(items, LawCatalog.load(session).law_id)
    new_items, astats = J.apply(items, cases, judged)
    save_jsonl(new_items, _sibling(out, "judged"))
    md = _sibling(out, "judge").with_suffix(".md")
    md.write_text(J.report_markdown(cases, judged, jstats, astats, label), "utf-8")
    typer.echo(f"{jstats}\n{astats}\n→ {_sibling(out, 'judged')}, 검토: {md}")


@app.command("judge-sample")
def judge_sample(
    pool_path: Path = typer.Argument(..., help="pool 파일"),
    n: int = typer.Option(90, help="표본 수 (판정 등급별로 3분의 1씩)"),
    seed: int = typer.Option(0, help="표본 시드"),
) -> None:
    """사람 판정용 표본 → <평가셋>.human.jsonl. 각 줄의 grade 를 0 · 1 · 2 로 채운 뒤 judge-agree."""
    from eval.golden import judge as J
    from eval.golden.pool import load as load_pool

    base = _base(pool_path)
    rows = J.sample_for_human(
        load_pool(pool_path), J.load_judgments(_sibling(base, "judgments")), n, seed
    )
    out = _sibling(base, "human")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    typer.echo(f"{len(rows)}건 → {out}")


@app.command("judge-agree")
def judge_agree(
    human: Path = typer.Argument(..., help="grade 를 채운 *.human.jsonl"),
) -> None:
    """사람 판정과 판정기 판정의 일치도 (정확도 · kappa · 2차 가중 kappa · 관련/무관 kappa)."""
    from eval.golden import judge as J

    base = _base(human)
    rows = [json.loads(ln) for ln in human.open(encoding="utf-8") if ln.strip()]
    result = J.agreement(rows, J.load_judgments(_sibling(base, "judgments")))
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("judge-merge")
def judge_merge(
    pool_path: Path = typer.Argument(..., help="pool 파일"),
    a: Path = typer.Option(..., help="판정 A (*.judgments.jsonl). 근거가 얕은 쪽"),
    b: Path = typer.Option(..., help="판정 B (*.judgments.jsonl). 1 차이는 B 를 따른다"),
    overrides: Path = typer.Option(
        None, help="0 ↔ 2 로 갈린 조문을 사람이 정한 JSON ({상황: {키: 등급}})"
    ),
    out: Path = typer.Option(..., help="합친 결과 (answers 형식 JSON). judge --answers 로 넣는다"),
) -> None:
    """두 판정기를 합친다. 같으면 그대로, 1 차이는 seed 면 높은 쪽 · 아니면 B, 0 ↔ 2 는 overrides.

    결과를 judge --answers 로 넣으면 .judgments · .judged · .judge.md 가 만들어진다."""
    from eval.golden import judge as J
    from eval.golden.pool import load as load_pool

    ja, jb = J.load_judgments(a), J.load_judgments(b)
    ov = json.loads(overrides.read_text("utf-8")) if overrides else None
    answers, stats, conflicts = J.merge(load_pool(pool_path), ja, jb, ov)
    out.write_text(json.dumps(answers, ensure_ascii=False, indent=1) + "\n", "utf-8")
    xa = [g for c in sorted(ja.keys() & jb.keys()) for _, g in sorted(ja[c].items())]
    xb = [g for c in sorted(ja.keys() & jb.keys()) for _, g in sorted(jb[c].items())]
    typer.echo(f"kappa(A, B) = {J.kappa(xa, xb):.3f}  가중 = {J.kappa(xa, xb, True):.3f}")
    typer.echo(f"{stats}\n→ {out}")
    unresolved = [c for c in conflicts if c["why"] == "unresolved"]
    for c in unresolved:
        typer.echo(f"미해결 0↔2: {c['case']} {c['key']} {c['law_name']} {c['article_no']}")


@app.command()
def split(
    dataset: Path = typer.Argument(..., help="평가셋 jsonl"),
    test: int = typer.Option(60, help="최종 확인용으로 떼어 둘 상황 수"),
    seed: str = typer.Option("lawlens-split-v1", help="분할 시드 (같으면 같은 분할)"),
) -> None:
    """상황 단위로 조정용(.dev)과 최종 확인용(.test)으로 나눈다. user · expert 질의는 같은 쪽에 둔다.

    튜닝(리랭커 등)은 .dev 로만 보고, .test 는 마지막 확인에만 쓴다."""
    import hashlib

    from eval.golden.pool import group_cases

    items = load_jsonl(dataset)
    cases = sorted(
        group_cases(items), key=lambda c: hashlib.md5(f"{seed}:{c}".encode()).hexdigest()
    )
    held = set(cases[:test])
    base = _base(dataset)
    name = dataset.name.removesuffix(".jsonl")
    for part, keep in (("test", True), ("dev", False)):
        out = base.with_name(f"{name}.{part}.jsonl")
        n = save_jsonl([it for it in items if (case_id(it) in held) == keep], out)
        typer.echo(f"{part}: {n}문항 → {out}")


def case_id(item: EvalItem) -> str:
    from eval.golden.pool import case_id as _case_id

    return _case_id(item)


def _base(path: Path) -> Path:
    """eval/datasets/x.pool.jsonl(.gz) → eval/datasets/x"""
    name = path.name.removesuffix(".gz").removesuffix(".jsonl")
    for suffix in (".pool", ".judgments", ".judged", ".human"):
        name = name.removesuffix(suffix)
    return path.with_name(name)


def _sibling(path: Path, kind: str) -> Path:
    """eval/datasets/x(.jsonl) → eval/datasets/x.<kind>.jsonl"""
    base = _base(path)
    return base.with_name(f"{base.name}.{kind}.jsonl")


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
