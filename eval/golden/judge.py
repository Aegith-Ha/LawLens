"""등급 판정 (judge): pool 의 후보 조문마다 2 핵심 / 1 관련 / 0 무관 을 매기고 정답을 다시 만든다.

판정자는 두 가지 중 하나:
- API: Claude 가 상황 하나(정보 요구 + 후보 전체)를 한 번에 판정. 상황 설명 · 조문 본문이 Anthropic API 로 전송된다.
- answers 파일: 사람이나 Claude Code 세션이 미리 판정한 JSON.
  {"expc-338989": {"2": ["001823:2"], "1": ["001823:45"]}}  ← 적지 않은 후보는 0

판정자는 seed 여부와 검색 순위를 보지 않는다 (pool 에서 후보 순서도 섞여 있다).
seed 와 판정을 비교한 표, 사람 판정과의 일치도(kappa)로 판정기를 검증한다.
"""

import json
import random
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from eval.dataset import EvalItem, Gold
from eval.golden.pool import PoolCase, case_id

MODEL = "claude-opus-5-5"
GRADES = (0, 1, 2)

SYSTEM = """당신은 한국 법령 검색 시스템의 평가 정답을 판정하는 사람입니다.
어떤 사람이 처한 상황(정보 요구)과 검색으로 모은 후보 조문들을 보고,
각 조문이 그 상황의 법적 쟁점에 답하는 데 얼마나 필요한지 등급을 매기세요.

- 2 (핵심): 쟁점의 결론을 내는 직접 근거. 문제되는 개념의 정의, 요건, 효과, 허용 · 금지를 정한 조문.
  이 조문을 찾지 못하면 답할 수 없다.
- 1 (관련): 결론의 직접 근거는 아니지만 함께 봐야 답이 완성되는 조문.
  핵심 조문이 위임 · 준용하는 조문, 같은 제도의 절차 · 예외 중 이 쟁점과 직접 닿는 것.
- 0 (무관): 쟁점에 답하는 데 쓰이지 않는 조문. 같은 법령이거나 비슷한 단어가 나와도 쟁점과 관계없으면 0.

규칙
- 상황 설명에 결론(회답)이 있으면 그 결론을 뒷받침하는 데 필요한지를 기준으로 판단하세요.
- 조문 본문으로 판단하세요. 법령 이름이 그럴듯하다는 이유만으로 등급을 주지 마세요.
- 애매하면 낮은 등급을 주세요. 핵심(2)은 보통 1~3개입니다. 핵심이 하나도 없을 수도 있습니다.
- 목적 · 정의 조문처럼 법 전체에 걸치는 조문은, 쟁점이 바로 그 정의나 목적에 달려 있을 때만 1 이상입니다.
- 모든 후보에 등급을 매기세요."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "grade": {"type": "integer", "enum": list(GRADES)},
                },
                "required": ["id", "grade"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}

Judgments = dict[str, int]  # 후보 키 → 등급
JudgeFn = Callable[[PoolCase], Judgments | None]


def prompt(p: PoolCase) -> str:
    qs = "\n".join(f"- {q['query']}" for q in p.queries)
    cands = "\n\n".join(f"<c{i}>\n{c.text}\n</c{i}>" for i, c in enumerate(p.candidates, start=1))
    return (
        f"[상황]\n{p.need}\n\n[이 상황에서 나온 검색 질의]\n{qs}\n\n"
        f"[후보 조문 {len(p.candidates)}개]\n{cands}"
    )


def judge_api(client: Any, p: PoolCase) -> Judgments | None:
    """Claude 호출 한 번. 거절되면 None."""
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt(p)}],
    )
    if resp.stop_reason == "refusal":
        return None
    return _parse(p, next((b.text for b in resp.content if b.type == "text"), ""))


def judge_codex(
    p: PoolCase, model: str | None = None, timeout: int = 900, log_dir: Path | None = None
) -> Judgments | None:
    """Codex CLI (`codex exec`) 로 판정. 로그인한 ChatGPT 계정의 사용량을 쓴다.

    도구를 쓰지 않도록 빈 임시 폴더에서 읽기 전용 샌드박스로 실행하고, 응답은 SCHEMA 로 고정한다."""
    with tempfile.TemporaryDirectory() as tmp:
        schema, out = Path(tmp, "schema.json"), Path(tmp, "out.json")
        schema.write_text(json.dumps(SCHEMA), "utf-8")
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "-C", tmp, "--output-schema", str(schema), "-o", str(out), "--color", "never"]  # fmt: skip
        if model:
            cmd += ["-m", model]
        r = subprocess.run(
            [*cmd, "-"], input=f"{SYSTEM}\n\n{prompt(p)}", capture_output=True, text=True,
            timeout=timeout,
        )  # fmt: skip
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{p.case}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex 종료 코드 {r.returncode}: {r.stderr[-500:]}")
        return _parse(p, out.read_text("utf-8"))


def _parse(p: PoolCase, txt: str) -> Judgments:
    by_id = {f"c{i}": c.key for i, c in enumerate(p.candidates, start=1)}
    out: Judgments = {}
    for j in json.loads(txt)["judgments"]:
        if j["id"] in by_id:
            out[by_id[j["id"]]] = int(j["grade"])
    missing = set(by_id.values()) - set(out)
    if missing:
        raise ValueError(f"{p.case}: 판정이 빠진 후보 {len(missing)}개")
    return out


def from_answers(path: Path) -> JudgeFn:
    """{상황: {"2": [키...], "1": [키...]}} 파일. 적지 않은 후보는 0."""
    answers: dict[str, dict[str, list[str]]] = json.loads(path.read_text("utf-8"))

    def judge(p: PoolCase) -> Judgments | None:
        if p.case not in answers:
            return None  # 아직 판정하지 않은 상황
        keys = {c.key for c in p.candidates}
        out = dict.fromkeys(keys, 0)
        for grade, ks in answers[p.case].items():
            for k in ks:
                if k not in keys:
                    raise KeyError(f"{path}: {p.case} 의 후보에 {k} 가 없습니다")
                out[k] = max(out[k], int(grade))
        return out

    return judge


def judge_all(
    pool: list[PoolCase],
    fn: JudgeFn,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Judgments], dict[str, int]]:
    """→ ({상황: 판정}, 통계). 판정하지 못한 상황은 결과에 없다."""
    out: dict[str, Judgments] = {}
    stats = {"cases": len(pool), "judged": 0, "skipped": 0, "errors": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, p): p for p in pool}
        for i, fut in enumerate(as_completed(futures), start=1):
            p = futures[fut]
            try:
                j = fut.result()
            except Exception as e:  # 한 건 실패로 전체를 멈추지 않는다
                stats["errors"] += 1
                if progress:
                    progress(f"오류 {p.case}: {type(e).__name__}: {e}")
                continue
            if j is None:
                stats["skipped"] += 1
                continue
            out[p.case] = j
            stats["judged"] += 1
            if progress and i % 10 == 0:
                progress(f"{i}/{len(pool)}")
    return out, stats


def save_judgments(
    pool: list[PoolCase], judged: dict[str, Judgments], path: Path, judge: str
) -> None:
    """한 줄 = 후보 하나: {case, key, law_name, article_no, grade, seed, judge}."""
    with path.open("w", encoding="utf-8") as f:
        for p in pool:
            if p.case not in judged:
                continue
            for c in p.candidates:
                row = {"case": p.case, "key": c.key, "law_name": c.law_name,
                       "article_no": c.article_no, "grade": judged[p.case][c.key],
                       "seed": c.seed, "judge": judge}  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_judgments(path: Path) -> dict[str, Judgments]:
    out: dict[str, Judgments] = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                out.setdefault(r["case"], {})[r["key"]] = int(r["grade"])
    return out


def apply(
    items: list[EvalItem], pool: list[PoolCase], judged: dict[str, Judgments]
) -> tuple[list[EvalItem], dict[str, int]]:
    """판정으로 정답을 바꾼 평가셋. 판정 안 된 상황, 관련 조문이 하나도 없는 상황은 뺀다.

    원래 정답(seed)은 meta.seeds 에 남긴다."""
    cands = {p.case: {c.key: c for c in p.candidates} for p in pool}
    out: list[EvalItem] = []
    stats = {"items": len(items), "kept": 0, "unjudged": 0, "no_relevant": 0}
    for it in items:
        case = case_id(it)
        j = judged.get(case)
        if j is None:
            stats["unjudged"] += 1
            continue
        gold = [
            Gold(article_no=c.article_no, law_name=c.law_name, law_id=c.law_id, grade=g)
            for k, g in sorted(j.items(), key=lambda kv: (-kv[1], kv[0]))
            if g > 0 and (c := cands[case][k])
        ]
        if not gold:
            stats["no_relevant"] += 1
            continue
        seeds = [
            {"law_name": g.law_name, "article_no": g.article_no, "grade": g.grade} for g in it.gold
        ]
        out.append(replace(it, gold=gold, meta={**it.meta, "case": case, "seeds": seeds}))
        stats["kept"] += 1
    return out, stats


# ── 검증 ─────────────────────────────────────────────────────────────────────


def seed_table(pool: list[PoolCase], judged: dict[str, Judgments]) -> dict[str, Counter[int]]:
    """seed 등급(2 · 1 · 없음)별로 판정 등급이 어떻게 나왔는지."""
    t: dict[str, Counter[int]] = {"seed 2": Counter(), "seed 1": Counter(), "seed 아님": Counter()}
    for p in pool:
        j = judged.get(p.case)
        if j is None:
            continue
        for c in p.candidates:
            row = f"seed {c.seed}" if c.seed else "seed 아님"
            t[row][j[c.key]] += 1
    return t


def kappa(a: list[int], b: list[int], weighted: bool = False) -> float:
    """Cohen's kappa (weighted=True 면 2차 가중: 0↔2 불일치를 0↔1 보다 크게 본다)."""
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(GRADES)

    def w(i: int, j: int) -> float:
        return ((i - j) ** 2) / ((k - 1) ** 2) if weighted else float(i != j)

    pa, pb = Counter(a), Counter(b)
    obs = sum(w(x, y) for x, y in zip(a, b, strict=True)) / n
    exp = sum(w(i, j) * pa[i] * pb[j] for i in GRADES for j in GRADES) / (n * n)
    return 1 - obs / exp if exp else 1.0


def sample_for_human(
    pool: list[PoolCase], judged: dict[str, Judgments], n: int, seed: int = 0
) -> list[dict[str, Any]]:
    """사람이 판정할 표본. 판정 등급별로 고르게 뽑고 (2 · 1 이 드물어서), 판정 결과는 숨긴다."""
    by_grade: dict[int, list[tuple[PoolCase, Any]]] = {g: [] for g in GRADES}
    for p in pool:
        if p.case in judged:
            for c in p.candidates:
                by_grade[judged[p.case][c.key]].append((p, c))
    rng = random.Random(seed)
    picked: list[tuple[PoolCase, Any]] = []
    for g in GRADES:
        rng.shuffle(by_grade[g])
        picked += by_grade[g][: n // len(GRADES)]
    rng.shuffle(picked)
    return [
        {
            "case": p.case,
            "key": c.key,
            "need": p.need,
            "queries": [q["query"] for q in p.queries],
            "article": c.text,
            "grade": None,
        }  # fmt: skip
        for p, c in picked
    ]


def agreement(human: list[dict[str, Any]], judged: dict[str, Judgments]) -> dict[str, Any]:
    pairs = [
        (int(h["grade"]), judged[h["case"]][h["key"]])
        for h in human
        if h.get("grade") is not None and h["case"] in judged
    ]
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    confusion = Counter(f"사람{x}-판정{y}" for x, y in pairs)
    return {
        "n": len(pairs),
        "exact": round(sum(x == y for x, y in pairs) / max(1, len(pairs)), 3),
        "kappa": round(kappa(a, b), 3),
        "kappa_weighted": round(kappa(a, b, weighted=True), 3),
        "relevant_kappa": round(kappa([min(x, 1) for x in a], [min(y, 1) for y in b]), 3),
        "confusion": dict(sorted(confusion.items())),
    }


def report_markdown(
    pool: list[PoolCase],
    judged: dict[str, Judgments],
    judge_stats: dict[str, int],
    apply_stats: dict[str, int],
    judge: str,
) -> str:
    t = seed_table(pool, judged)
    total = Counter[int]()
    for cnt in t.values():
        total.update(cnt)
    lines = [
        "# 정답 판정 결과",
        "",
        f"- 판정자: {judge}",
        f"- 판정: {json.dumps(judge_stats, ensure_ascii=False)}",
        f"- 평가셋: {json.dumps(apply_stats, ensure_ascii=False)}",
        f"- 전체 후보 등급: 2 = {total[2]}, 1 = {total[1]}, 0 = {total[0]}",
        "",
        "## seed 와 판정 비교",
        "",
        "| | 판정 2 | 판정 1 | 판정 0 | 계 |",
        "|---|---|---|---|---|",
    ]
    for row, cnt in t.items():
        lines.append(f"| {row} | {cnt[2]} | {cnt[1]} | {cnt[0]} | {sum(cnt.values())} |")
    lines += [
        "",
        "- seed 2 가 판정 0 으로 많이 떨어지면 판정 기준이 너무 엄격하거나 seed 가 잘못된 것",
        "- seed 아님 → 판정 2 · 1 은 기존 정답이 놓친 조문 (새로 찾은 정답)",
        "",
        "## 상황별",
        "",
    ]
    for p in pool:
        j = judged.get(p.case)
        if j is None:
            continue
        lines.append(f"### {p.case}")
        lines.append("")
        lines.append(f"- 질의: {' / '.join(q['query'] for q in p.queries)}")
        for c in sorted(p.candidates, key=lambda c: (-j[c.key], -(c.seed or 0))):
            if j[c.key] == 0 and not c.seed:
                continue
            head = c.text.split("\n", 1)[0]
            seed = f" · seed {c.seed}" if c.seed else ""
            lines.append(f"- **{j[c.key]}** {head}{seed}")
        lines.append("")
    return "\n".join(lines)


# ── 두 판정기 합치기 ─────────────────────────────────────────────────────────


def merge(
    pool: list[PoolCase],
    a: dict[str, Judgments],
    b: dict[str, Judgments],
    overrides: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, int], list[dict[str, Any]]]:
    """두 판정(a, b)을 합쳐 answers 형식({상황: {"2": [키], "1": [키]}})으로 만든다.

    - 같으면 그대로.
    - 1 차이: seed 면 높은 쪽 (해석례가 인용했으니 관련은 있다), 아니면 b.
      b 는 조문 전문을 읽은 판정기로 둔다 (a 는 근거가 얕은 판정).
    - 0 ↔ 2: overrides({상황: {키: 등급}})로 사람이 정한다. 없으면 1 로 두고 미해결로 보고한다.
    → (answers, 통계, 0 ↔ 2 목록)"""
    overrides = overrides or {}
    answers: dict[str, dict[str, list[str]]] = {}
    stats = Counter[str]()
    conflicts: list[dict[str, Any]] = []
    for p in pool:
        if p.case not in a or p.case not in b:
            stats["case_missing"] += 1
            continue
        out: dict[str, list[str]] = {"2": [], "1": []}
        for c in p.candidates:
            ga, gb = a[p.case][c.key], b[p.case][c.key]
            ov = overrides.get(p.case, {}).get(c.key)
            if ov is not None:
                g, why = ov, "override"
            elif ga == gb:
                g, why = ga, "same"
            elif abs(ga - gb) == 1:
                g, why = (max(ga, gb), "seed_max") if c.seed else (gb, "b")
            else:
                g, why = 1, "unresolved"
            if abs(ga - gb) == 2:
                conflicts.append({"case": p.case, "key": c.key, "law_name": c.law_name,
                                  "article_no": c.article_no, "seed": c.seed, "a": ga, "b": gb,
                                  "final": g, "why": why})  # fmt: skip
            stats[why] += 1
            if g:
                out[str(g)].append(c.key)
        answers[p.case] = out
    return answers, dict(stats), conflicts
