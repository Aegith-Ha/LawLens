"""평가셋 B: 현행 조문을 무작위로 뽑아 Claude 가 "일반인이 할 법한 질문" 을 만든다. 정답 = 원래 조문.

질문이 원문 표현을 따라가면 키워드 검색에 유리해지므로, 법령명 · 조문번호 · 원문의 긴 구절을
쓰지 말고 상황으로 묻게 하고, 생성 후에도 한 번 더 걸러 낸다.

조문 본문(공개 법령)이 Anthropic API 로 전송된다.
"""

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from eval.dataset import EvalItem, Gold
from lawlens.timeutil import today_kst

MODEL = "claude-opus-5-5"

# 법령 종류별 표본 비율 (현행 법령 수와 실제 질문 빈도를 고려해 법률 비중을 높게)
DEFAULT_MIX = {"법률": 0.45, "대통령령": 0.30, "총리령·부령": 0.15, "규칙": 0.05, "헌법": 0.05}

SYSTEM = """당신은 한국 법령 검색 시스템의 평가 데이터를 만드는 사람입니다.
조문 하나를 읽고, 그 조문이 답이 되는 질문을 만드세요.

규칙
- situation: 법을 모르는 일반인이 자기 상황을 설명하며 묻는 질문 (1~2문장, 구어체)
- term: 실무자가 법률 용어로 짧게 묻는 질문 (한 문장)
- 두 질문 모두 법령 이름, 조문 번호(제N조), 조문의 표현을 다섯 어절 이상 그대로 쓰지 마세요.
- 조문이 정의 규정, 시행일, 위임 근거, 벌칙 금액표처럼 일반인이 직접 물을 일이 거의 없는 내용이면
  usable 을 false 로 하세요."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "situation": {"type": "string"},
        "term": {"type": "string"},
    },
    "required": ["usable", "situation", "term"],
    "additionalProperties": False,
}

# 법령 종류 묶음 (검색 필터 LAW_KINDS 와 같은 기준의 SQL 판)
SQL_KINDS = {
    "헌법": "l.law_type = '헌법'",
    "법률": "l.law_type = '법률'",
    "대통령령": "l.law_type in ('대통령령', '대통령긴급명령')",
    "총리령·부령": "l.law_type like '%령' and l.law_type not in ('대통령령', '대통령긴급명령')",
    "규칙": "l.law_type like '%규칙'",
}

_ARTICLE_NO = re.compile(r"제\s?\d+\s?조")


@dataclass
class Sample:
    law_id: str
    law_name: str
    law_type: str
    article_no: str
    label: str
    text: str


def _label(article_no: str, title: str) -> str:
    main, _, branch = article_no.partition("의")
    base = f"제{main}조의{branch}" if branch else f"제{main}조"
    return f"{base}({title})" if title else base


def sample_articles(
    session: Session, n: int, seed: str, mix: dict[str, float] | None = None
) -> list[Sample]:
    """현행 조문에서 법령 종류별 비율대로 무작위 추출. seed 가 같으면 같은 표본."""
    mix = mix or DEFAULT_MIX
    out: list[Sample] = []
    for kind, ratio in mix.items():
        k = max(1, round(n * ratio))
        expr = SQL_KINDS[kind]
        rows = session.execute(
            text(
                "select l.law_id, l.name, l.law_type, a.article_no, coalesce(a.title, ''), a.text"
                " from article a join law_version v using (version_key) join law l using (law_id)"
                " where v.valid_from <= :d and v.valid_to > :d and not a.is_deleted"
                " and length(a.text) between 120 and 2500"
                f" and ({expr})"
                " order by md5(a.article_id || :seed) limit :k"
            ),
            {"d": today_kst(), "seed": seed, "k": k},
        )
        out += [Sample(r[0], r[1], r[2], r[3], _label(r[3], r[4]), r[5]) for r in rows]
    return out


def _leaks(q: str, s: Sample) -> bool:
    """질문에 법령명 · 조문번호 · 원문의 긴 구절이 들어갔는지."""
    if not q or _ARTICLE_NO.search(q) or s.law_name.replace(" ", "") in q.replace(" ", ""):
        return True
    words = q.split()
    body = s.text.replace("\n", " ")
    return any(" ".join(words[i : i + 5]) in body for i in range(max(0, len(words) - 4)))


def generate_one(client: Any, s: Sample) -> dict[str, Any] | None:
    """Claude 호출 한 번. 거절되면 None."""
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # 안전 분류기가 거절하면 서버가 다른 모델로 재시도
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": f"법령: {s.law_name} ({s.law_type})\n조문: {s.label}\n\n{s.text}",
            }
        ],
    )
    if resp.stop_reason == "refusal":
        return None
    txt = next((b.text for b in resp.content if b.type == "text"), "")
    data: dict[str, Any] = json.loads(txt)
    return data


def build(
    samples: list[Sample],
    client: Any,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[EvalItem], dict[str, int]]:
    stats = {
        "samples": len(samples),
        "unusable": 0,
        "refused": 0,
        "leaked": 0,
        "errors": 0,
        "items": 0,
    }
    items: list[EvalItem] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, client, s): s for s in samples}
        for i, fut in enumerate(as_completed(futures), start=1):
            s = futures[fut]
            try:
                data = fut.result()
            except Exception as e:  # 한 문항 실패로 전체를 멈추지 않는다
                stats["errors"] += 1
                if progress:
                    progress(f"오류 {s.law_name} {s.label}: {type(e).__name__}: {e}")
                continue
            if data is None:
                stats["refused"] += 1
                continue
            if not data.get("usable"):
                stats["unusable"] += 1
                continue
            gold = [Gold(article_no=s.article_no, law_name=s.law_name, law_id=s.law_id, grade=2)]
            meta = {"law_type": s.law_type, "label": s.label}
            for typ in ("situation", "term"):
                q = str(data.get(typ) or "").strip()
                if _leaks(q, s):
                    stats["leaked"] += 1
                    continue
                items.append(
                    EvalItem(
                        id=f"synth-{s.law_id}-{s.article_no}-{typ}",
                        query=q,
                        gold=gold,
                        type=typ,
                        source="synth",
                        meta=meta,
                    )
                )
                stats["items"] += 1
            if progress and i % 25 == 0:
                progress(f"{i}/{len(samples)}")
    items.sort(key=lambda it: it.id)
    return items, stats
