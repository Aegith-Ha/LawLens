"""평가 실행: 문항마다 검색하고 지표를 계산해 데이터셋 · 유형 · 검색 방식별로 묶는다."""

import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from eval.dataset import EvalItem
from eval.scoring.embed_cache import EmbeddingCache
from eval.scoring.metrics import hit_at_k, mean, ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from lawlens.retrieval.hybrid import SearchMode
from lawlens.retrieval.service import SearchResponse

KS = (1, 5, 10, 20)
TOP = max(KS)


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
class ItemResult:
    dataset: str
    id: str
    type: str
    mode: str
    query: str
    gold: dict[str, int]
    retrieved: list[str]
    metrics: dict[str, float]
    took_ms: int
    exact_count: int = 0


@dataclass
class Summary:
    dataset: str
    type: str
    mode: str
    n: int
    metrics: dict[str, float] = field(default_factory=dict)


def ranking(resp: SearchResponse) -> list[str]:
    """화면에 보이는 순서 그대로: 조문 바로가기 → 관련 조문."""
    return [f"{a.law_id}:{a.article_no}" for a in [*resp.exact, *resp.results]]


def score(retrieved: list[str], gains: dict[str, int]) -> dict[str, float]:
    rel = set(gains)
    m = {f"recall@{k}": recall_at_k(retrieved, rel, k) for k in KS if k > 1}
    m["hit@1"] = hit_at_k(retrieved, rel, 1)
    m["hit@10"] = hit_at_k(retrieved, rel, 10)
    m["mrr@10"] = reciprocal_rank(retrieved, rel, 10)
    m["ndcg@10"] = ndcg_at_k(retrieved, gains, 10)
    return m


def run(
    datasets: dict[str, list[EvalItem]],
    service: Service,
    modes: Iterable[SearchMode],
    progress: Callable[[str], None] | None = None,
    encoder: EmbeddingCache | None = None,
    every: int = 10,
) -> list[ItemResult]:
    out: list[ItemResult] = []
    modes = list(modes)
    for name, items in datasets.items():
        started = time.perf_counter()
        for i, it in enumerate(items, start=1):
            gains = it.gains()
            if not gains:
                continue  # 정답을 하나도 law_id 로 못 바꾼 문항
            for mode in modes:
                resp = service.search(
                    it.query, limit=TOP, as_of=it.as_of, kinds=it.kinds, mode=mode
                )
                retrieved = ranking(resp)
                took = resp.took_ms + (encoder.encode_ms if encoder and encoder.hit else 0)
                out.append(
                    ItemResult(
                        dataset=name,
                        id=it.id,
                        type=it.type,
                        mode=mode.value,
                        query=it.query,
                        gold=gains,
                        retrieved=retrieved,
                        metrics=score(retrieved, gains),
                        took_ms=took,
                        exact_count=len(resp.exact),
                    )
                )
            if progress and (i % every == 0 or i == len(items)):
                el = time.perf_counter() - started
                eta = el / i * (len(items) - i)
                progress(
                    f"{name}: {i}/{len(items)}  경과 {el / 60:.1f}분, 남은 시간 약 {eta / 60:.1f}분"
                )
    return out


def summarize(results: list[ItemResult]) -> list[Summary]:
    groups: dict[tuple[str, str, str], list[ItemResult]] = defaultdict(list)
    for r in results:
        groups[(r.dataset, "전체", r.mode)].append(r)
        groups[(r.dataset, r.type, r.mode)].append(r)
    out = []
    for (ds, typ, mode), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "전체", kv[0][1], kv[0][2])
    ):
        if typ != "전체" and len({r.type for r in results if r.dataset == ds}) == 1:
            continue  # 유형이 하나뿐이면 "전체" 와 같아서 생략
        names = list(rs[0].metrics)
        m = {k: mean([r.metrics[k] for r in rs]) for k in names}
        lat = [float(r.took_ms) for r in rs]
        m["p50_ms"] = percentile(lat, 50)
        m["p95_ms"] = percentile(lat, 95)
        out.append(Summary(ds, typ, mode, len(rs), m))
    return out


COLUMNS = [
    "recall@5",
    "recall@10",
    "recall@20",
    "hit@10",
    "mrr@10",
    "ndcg@10",
    "hit@1",
    "p50_ms",
    "p95_ms",
]


def to_markdown(summaries: list[Summary]) -> str:
    head = "| 데이터셋 | 유형 | 방식 | 문항 | " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * (4 + len(COLUMNS))
    rows = []
    for s in summaries:
        vals = [
            f"{s.metrics[c]:.0f}" if c.endswith("_ms") else f"{s.metrics[c]:.3f}" for c in COLUMNS
        ]
        rows.append(f"| {s.dataset} | {s.type} | {s.mode} | {s.n} | " + " | ".join(vals) + " |")
    return "\n".join([head, sep, *rows])


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def write_report(
    results: list[ItemResult], summaries: list[Summary], out_dir: Path, config: dict[str, Any]
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = out_dir / stamp
    d.mkdir(parents=True, exist_ok=True)
    config = {**config, "git": _git_commit(), "created_at": stamp}
    (d / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
    with (d / "items.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            row = asdict(r)
            row["retrieved"] = r.retrieved[:10]  # 오답 분석용으로 상위 10개만
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md = [
        "# 검색 평가 결과",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        to_markdown(summaries),
        "",
        "- recall@k: 정답 조문 중 상위 k 안에 든 비율 / mrr@10: 첫 정답 순위의 역수",
        "- hit@10: 정답이 하나라도 상위 10 안에 든 비율 / ndcg@10: 핵심(2)·관련(1) 등급 반영 순위 품질",
        "- hit@1: 1위가 정답인 비율 / p50·p95_ms: 질의당 검색 시간",
        "",
    ]
    (d / "report.md").write_text("\n".join(md), "utf-8")
    return d
