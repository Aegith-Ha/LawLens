"""평가셋 C 재작성: 법령해석례 질의요지 → Claude 가 다시 쓴 검색 질의.

지금 expc_masked 는 법령명 · 조문번호를 정규식으로 가린 것이라 문장이 깨지고("관련 규정에 따른
관련 규정…"), 어느 법 이야기인지 알 수 없는 질의가 섞인다. 그래서 해석례 전체(질의요지 · 회답 ·
이유)와 핵심 정답 조문을 Claude 에게 보여 주고 사람이 검색창에 칠 법한 질의로 다시 쓴다.

- expc_user:   일반인 · 민원인이 자기 상황으로 묻는 질의 (구어체, 짧게)
- expc_expert: 실무자가 쟁점을 요약해 묻는 질의 (법률 용어 가능)

정답(gold)은 기존 expc.jsonl 그대로 쓴다 (정답 다듬기는 다음 단계).
해석례와 조문 본문(공개 자료)이 Anthropic API 로 전송된다.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from eval.dataset import EvalItem, Gold
from eval.golden.build_expc import _clean
from eval.golden.catalog import article_text, norm_name

MODEL = "claude-opus-5-5"

SYSTEM = """당신은 한국 법령 검색 시스템의 평가 데이터를 만드는 사람입니다.
법제처 법령해석례 하나(질의요지 · 회답 · 이유)와 그 근거가 되는 핵심 조문을 읽고,
사람이 이 쟁점의 근거 조문을 찾으려고 검색창에 입력할 질의를 두 가지 만드세요.

- user: 법을 잘 모르는 일반인이나 민원인이 자기 상황을 설명하며 묻는 질의. 구어체, 1~3문장, 200자 이내.
- expert: 공무원 · 실무자가 쟁점을 요약해 묻는 질의. 법률 용어를 써도 됨. 1~2문장, 150자 이내.

규칙
- 법령 이름(약칭 · 줄임말 포함), 조 · 항 · 호 번호, 「 」 표시를 쓰지 마세요.
- 질의요지 문장을 옮기지 말고 새로 쓰세요. 원문과 다섯 어절 이상 같은 표현이 이어지면 안 됩니다.
- 회답의 결론(해당한다/아니다, 할 수 있다/없다)을 질의에 넣지 마세요. 답이 아니라 질문입니다.
- 검색으로 정답 조문에 닿을 수 있도록, 어떤 분야 · 업종 · 자격 · 행위 · 절차에 관한 것인지는
  드러나게 쓰세요. 분야를 짐작할 단서가 전혀 없는 질의는 실패입니다.
- 질의요지에 쟁점이 여러 개(가, 나, …)면 안건명의 핵심 쟁점 하나만 다루세요.
- 쟁점이 법령 이름 없이는 설명할 수 없는 순수한 조문 해석(부칙 경과조치의 적용 범위, 조문 사이의
  인용 관계 등)이라 위 규칙으로 쓸 수 없으면 usable 을 false 로 하고 reason 에 이유를 적으세요."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "reason": {"type": "string"},
        "user": {"type": "string"},
        "expert": {"type": "string"},
    },
    "required": ["usable", "reason", "user", "expert"],
    "additionalProperties": False,
}

_ARTICLE_NO = re.compile(r"제\s?\d+\s?(?:조|항|호)")
# (이하 "중소기업창업법"이라 함) 같은 약칭 정의
_ALIAS = re.compile(r"\(이하\s*[“\"‘']([^”\"’']{1,40})[”\"’']\s*(?:이)?라\s*(?:함|한다|합니다)\)")
_BRACKET = re.compile(r"「\s*([^」]{1,120}?)\s*」")
_LAWISH = re.compile(r"(?:법|법률|령|규칙|조례)$")


@dataclass
class Case:
    sid: str
    fields: dict[str, Any]  # 해석례 원본 (ExpcService)
    gold: list[Gold]
    old_query: str  # 기존 expc_masked 질의 (비교용)
    core_texts: list[str]  # 핵심 정답 조문 본문
    banned: set[str]  # 질의에 나오면 안 되는 법령명 · 약칭 (공백 제거)


def _hash(s: str, seed: str) -> str:
    return hashlib.md5(f"{seed}:{s}".encode()).hexdigest()


def sample_cases(expc_path: Path, n: int, seed: str, per_law: int = 2) -> list[EvalItem]:
    """expc.jsonl 의 masked 문항에서 n 건. 한 법령(첫 핵심 정답 기준)에 per_law 건까지만."""
    rows = [json.loads(ln) for ln in expc_path.open(encoding="utf-8") if ln.strip()]
    masked = [r for r in rows if r["type"] == "expc_masked"]
    masked.sort(key=lambda r: _hash(r["id"], seed))
    picked, per = [], Counter[str]()
    for r in masked:
        core = [g for g in r["gold"] if g.get("grade") == 2] or r["gold"]
        law = core[0].get("law_id") or core[0].get("law_name")
        if per[law] >= per_law:
            continue
        per[law] += 1
        picked.append(r)
        if len(picked) >= n:
            break
    from eval.dataset import item_from_dict

    return [item_from_dict(r) for r in picked]


def load_cases(items: list[EvalItem], raw_dir: Path, session: Session) -> list[Case]:
    out = []
    for it in items:
        sid = it.id.split("-")[1]
        f = json.loads((raw_dir / f"{sid}.json").read_text("utf-8"))["ExpcService"]
        whole = " ".join(str(f.get(k) or "") for k in ("안건명", "질의요지", "회답", "이유"))
        banned = {norm_name(g.law_name) for g in it.gold if g.law_name}
        banned |= {norm_name(re.sub(r"\s?(시행령|시행규칙)$", "", b)) for b in list(banned)}
        banned |= {norm_name(a) for a in _ALIAS.findall(whole) if _LAWISH.search(a)}
        banned |= {norm_name(b) for b in _BRACKET.findall(whole) if _LAWISH.search(b)}
        banned = {b for b in banned if len(b) >= 3}
        core = [g for g in it.gold if g.grade == 2] or it.gold
        texts = [
            t for g in core if g.law_id and (t := article_text(session, g.law_id, g.article_no))
        ]
        out.append(Case(sid, f, it.gold, it.query, texts, banned))
    return out


def leak(q: str, case: Case) -> str | None:
    """질의에 법령명 · 조문번호 · 원문 구절이 들어갔으면 그 이유."""
    if not q:
        return "빈 질의"
    if "「" in q or _ARTICLE_NO.search(q):
        return "조문번호"
    nq = norm_name(q)
    for b in case.banned:
        if b in nq:
            return f"법령명:{b}"
    src = " ".join(
        _clean(str(case.fields.get(k) or "")).replace("\n", " ") for k in ("질의요지", "회답")
    )
    src += " " + " ".join(case.core_texts).replace("\n", " ")
    words = q.split()
    for i in range(max(0, len(words) - 4)):
        if " ".join(words[i : i + 5]) in src:
            return "원문구절"
    return None


def _prompt(case: Case) -> str:
    f = case.fields
    core = "\n\n".join(case.core_texts) or "(본문 없음)"
    return (
        f"안건명: {_clean(str(f.get('안건명') or ''))}\n\n"
        f"[질의요지]\n{_clean(str(f.get('질의요지') or ''))}\n\n"
        f"[회답]\n{_clean(str(f.get('회답') or ''))}\n\n"
        f"[이유]\n{_clean(str(f.get('이유') or ''))}\n\n"
        f"[핵심 정답 조문]\n{core}"
    )


def generate_one(client: Any, case: Case) -> dict[str, Any] | None:
    """Claude 호출 한 번. 거절되면 None."""
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # 안전 분류기가 거절하면 서버가 다른 모델로 재시도
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": _prompt(case)}],
    )
    if resp.stop_reason == "refusal":
        return None
    txt = next((b.text for b in resp.content if b.type == "text"), "")
    data: dict[str, Any] = json.loads(txt)
    data["_usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return data


def from_answers(path: Path) -> Callable[[Case], dict[str, Any] | None]:
    """미리 써 둔 질의 파일(JSON: {해석례 번호: {usable, reason, user, expert}})을 생성기로 쓴다.

    API 를 쓰지 않고 사람이나 Claude Code 세션이 직접 쓴 질의를 같은 검사 · 저장 과정에 넣기 위한 것.
    """
    answers: dict[str, dict[str, Any]] = json.loads(path.read_text("utf-8"))

    def gen(case: Case) -> dict[str, Any] | None:
        if case.sid not in answers:
            raise KeyError(f"{path} 에 해석례 {case.sid} 가 없습니다")
        return {**answers[case.sid], "_usage": {"in": 0, "out": 0}}

    return gen


def build(
    cases: list[Case],
    gen: Callable[[Case], dict[str, Any] | None],
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[EvalItem], list[dict[str, Any]], dict[str, int]]:
    """gen: 해석례 → 질의 (API 는 functools.partial(generate_one, client), 파일은 from_answers).

    → (평가 문항, 검토용 기록, 통계). 유출된 질의도 검토용 기록에는 남긴다."""
    stats = {"cases": len(cases), "unusable": 0, "refused": 0, "leaked": 0, "errors": 0,
             "items": 0, "tokens_in": 0, "tokens_out": 0}  # fmt: skip
    items: list[EvalItem] = []
    review: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(gen, c): c for c in cases}
        for i, fut in enumerate(as_completed(futures), start=1):
            c = futures[fut]
            try:
                data = fut.result()
            except Exception as e:  # 한 건 실패로 전체를 멈추지 않는다
                stats["errors"] += 1
                if progress:
                    progress(f"오류 {c.sid}: {type(e).__name__}: {e}")
                continue
            if progress and i % 10 == 0:
                progress(f"{i}/{len(cases)}")
            if data is None:
                stats["refused"] += 1
                continue
            stats["tokens_in"] += data["_usage"]["in"]
            stats["tokens_out"] += data["_usage"]["out"]
            rec: dict[str, Any] = {
                "sid": c.sid,
                "안건명": c.fields.get("안건명"),
                "usable": data.get("usable"),
                "reason": data.get("reason"),
                "old_query": c.old_query,
                "gold": [f"{g.law_name} {g.article_no} ({g.grade})" for g in c.gold],
            }
            if not data.get("usable"):
                stats["unusable"] += 1
                review.append(rec)
                continue
            meta = {
                "안건번호": c.fields.get("안건번호"),
                "안건명": c.fields.get("안건명"),
                "해석일자": c.fields.get("해석일자"),
            }
            for typ in ("user", "expert"):
                q = str(data.get(typ) or "").strip()
                why = leak(q, c)
                rec[typ] = q
                rec[f"{typ}_leak"] = why
                if why:
                    stats["leaked"] += 1
                    continue
                items.append(
                    EvalItem(
                        id=f"expc-{c.sid}-{typ}",
                        query=q,
                        gold=c.gold,
                        type=f"expc_{typ}",
                        source="expc",
                        meta=meta,
                    )
                )
                stats["items"] += 1
            review.append(rec)
    items.sort(key=lambda it: it.id)
    review.sort(key=lambda r: r["sid"])
    return items, review, stats


def review_markdown(review: list[dict[str, Any]], stats: dict[str, int]) -> str:
    """기존 masked 질의와 새 질의를 나란히 보는 검토용 문서."""
    out = ["# 법령해석례 질의 재작성 시범", "", f"`{json.dumps(stats, ensure_ascii=False)}`", ""]
    for n, r in enumerate(review, start=1):
        out += [f"## {n}. {r['안건명']}", "", f"- 해석례 {r['sid']} · 정답: {', '.join(r['gold'])}"]
        if not r["usable"]:
            out += [f"- **사용 불가**: {r['reason']}", ""]
            continue
        for typ, label in (("user", "일반인"), ("expert", "실무자")):
            flag = f" ⚠️ 유출({r[f'{typ}_leak']})" if r.get(f"{typ}_leak") else ""
            out.append(f"- **{label}**{flag}: {r.get(typ, '')}")
        old = r["old_query"]
        out += ["", "<details><summary>기존 masked 질의</summary>", "", old, "", "</details>", ""]
    return "\n".join(out)
