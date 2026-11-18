"""평가셋 C: 상황 시나리오 먼저 (조문을 보지 않고 질의 생성). 주 사용자는 일반인.

분야 목록(AREAS)마다 생성기(LLM)에게 일반인이 실제로 겪을 법한 상황을 쓰게 한다. 생성기에게는 조문을
보여 주지 않는다. 정답(seed)은 없고 pool + judge 로 정한다.

상황 하나 = 질의 2개 (정답 공유):
- scenario_user:  자기 상황을 구어체로 설명하며 묻는 질의 (1~3문장)
- scenario_short: 검색창에 치는 짧은 검색어 (2~6어절)

pool 편향을 줄이려고, 생성기가 법률 용어로 바꿔 쓴 검색어(pool_queries)도 만든다. 이것은 평가 질의가
아니라 pool 에 후보를 더 모으는 데만 쓴다 (meta.pool_queries).
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.dataset import EvalItem
from eval.golden.catalog import norm_name


@dataclass(frozen=True)
class Area:
    key: str
    name: str
    hint: str  # 생성기에게 주는 예시 (이것만 쓰라는 뜻은 아님)


AREAS: list[Area] = [
    Area(
        "housing",
        "주택 임대차",
        "전세 · 월세 보증금 반환, 계약 갱신, 수리비, 묵시적 갱신, 전입신고",
    ),
    Area(
        "labor",
        "근로 · 임금",
        "임금 체불, 퇴직금, 해고, 연차, 주휴수당, 아르바이트, 직장 내 괴롭힘",
    ),
    Area(
        "traffic",
        "교통 · 자동차",
        "교통사고 처리, 보험, 음주운전, 벌점 · 면허, 주정차 과태료, 킥보드",
    ),
    Area(
        "consumer",
        "소비자 · 거래",
        "온라인 쇼핑 환불, 중고거래, 할부, 헬스장 · 학원 해지, 방문판매",
    ),
    Area(
        "family",
        "가족 · 상속",
        "이혼, 양육비, 면접교섭, 상속 포기 · 한정승인, 유언, 친권, 혼인신고",
    ),
    Area("neighbor", "이웃 · 생활", "층간소음, 누수, 주차, 반려동물, 소음 · 악취, 공동주택 관리비"),
    Area("debt", "금융 · 채무", "대출, 보증, 빚 독촉, 개인회생 · 파산, 보이스피싱, 신용카드 분실"),
    Area("tax", "세금 · 보험료", "연말정산, 양도소득세, 증여, 건강보험료, 국민연금, 자동차세"),
    Area(
        "privacy", "개인정보 · 온라인", "개인정보 유출, CCTV, 명예훼손 · 모욕, 사진 무단 사용, 스팸"
    ),
    Area("medical", "의료 · 돌봄", "의료사고, 진료기록 열람, 장기요양, 요양원, 응급실, 약 부작용"),
    Area(
        "education",
        "교육 · 아동",
        "학교폭력, 어린이집 · 유치원, 학원비 환불, 체벌 · 아동학대, 전학",
    ),
    Area(
        "welfare", "복지 · 지원금", "기초생활수급, 실업급여, 장애인 지원, 기초연금, 출산 · 육아휴직"
    ),
    Area(
        "business",
        "자영업 · 창업",
        "상가 임대차, 영업신고 · 허가, 식품위생, 아르바이트생 고용, 폐업",
    ),
    Area(
        "property",
        "부동산 거래 · 건축",
        "매매 계약 해제, 중개수수료, 불법 건축물, 하자, 분양, 토지 경계",
    ),
    Area(
        "civic",
        "행정 · 병역 · 신분",
        "과태료 이의, 정보공개 청구, 병역 연기, 주민등록, 여권, 외국인 체류",
    ),
]

PER_AREA = 6
TYPES = ("user", "short")

SYSTEM = """당신은 한국 법령 검색 시스템의 평가 데이터를 만드는 사람입니다.
법을 잘 모르는 일반인이 생활하다가 겪는 문제로 법령(법률 · 시행령 · 시행규칙)을 찾아보는 상황을 만듭니다.
법령 조문은 보지 말고, 실제로 있을 법한 상황만 생각해서 쓰세요.

상황마다 다음을 쓰세요.
- persona: 누구인지 (예: 30대 세입자, 편의점 아르바이트 대학생, 70대 어르신의 자녀)
- need: 판정자가 읽을 상황 설명. 무슨 일이 있었고 무엇을 알고 싶은지 3~5문장. 결론이나 법적 답은 쓰지 마세요.
- user: 그 사람이 검색창이나 상담 게시판에 쓸 질의. 구어체, 1~3문장, 150자 이내.
- short: 같은 사람이 검색창에 칠 짧은 검색어. 2~6어절.
- pool_queries: 같은 쟁점을 법률 용어로 바꿔 쓴 검색어 2개. (평가에 쓰지 않고, 정답 후보를 넓게 모으는 데만 씀)

규칙
- user · short 에는 법령 이름(약칭 포함), 조 · 항 · 호 번호, 「 」 를 쓰지 마세요. 일반인은 법 이름을 모르는 것이 보통입니다.
- 답이 법령 조문에 있는 상황이어야 합니다. 판례나 계약서 해석만으로 답하는 문제, 사실관계 다툼은 피하세요.
- 상황마다 쟁점은 하나입니다.
- 같은 분야 안에서도 쟁점 · 사람 · 상황이 서로 겹치지 않게 다양하게 쓰세요.
  흔한 상황 절반, 덜 흔하지만 실제로 있는 구체적인 상황 절반으로 하세요.
- 질의 말투도 다양하게: 맞춤법이 조금 틀리거나 감정이 섞인 질의, 아주 짧은 질의도 섞으세요."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "persona": {"type": "string"},
                    "need": {"type": "string"},
                    "user": {"type": "string"},
                    "short": {"type": "string"},
                    "pool_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["persona", "need", "user", "short", "pool_queries"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}

Scenario = dict[str, Any]
GenFn = Callable[[Area, int], list[Scenario]]


def prompt(area: Area, n: int) -> str:
    return f"{SYSTEM}\n\n분야: {area.name} (예: {area.hint})\n이 분야에서 상황 {n}개를 만드세요."


def generate_codex(model: str | None = None, log_dir: Path | None = None) -> GenFn:
    from eval.golden.codex import run_json

    def gen(area: Area, n: int) -> list[Scenario]:
        log = log_dir / f"{area.key}.log" if log_dir else None
        out: list[Scenario] = run_json(prompt(area, n), SCHEMA, model, log=log)["scenarios"]
        return out

    return gen


def from_answers(path: Path) -> GenFn:
    """{분야 key: [상황...]} 파일 (생성 결과를 저장해 둔 것)."""
    data: dict[str, list[Scenario]] = json.loads(path.read_text("utf-8"))

    def gen(area: Area, n: int) -> list[Scenario]:
        return data.get(area.key, [])[:n]

    return gen


def generate_all(
    gen: GenFn, n: int = PER_AREA, workers: int = 3, progress: Callable[[str], None] | None = None
) -> dict[str, list[Scenario]]:
    out: dict[str, list[Scenario]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(gen, a, n): a for a in AREAS}
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                out[a.key] = fut.result()
            except Exception as e:  # 한 분야 실패로 전체를 멈추지 않는다
                if progress:
                    progress(f"오류 {a.key}: {type(e).__name__}: {e}")
                continue
            if progress:
                progress(f"{a.key}: {len(out[a.key])}개")
    return {a.key: out[a.key] for a in AREAS if a.key in out}


# ── 유출 검사 · 평가셋 만들기 ────────────────────────────────────────────────

_ARTICLE_NO = re.compile(r"제\s?\d+\s?(?:조|항|호)")
_HANGUL = re.compile(r"[가-힣]")


_PARTICLES = (
    "상",
    "에",
    "에서",
    "은",
    "는",
    "이",
    "을",
    "의",
    "으로",
    "로",
    "과",
    "와",
    "도",
    "만",
    "엔",
)


def leak(q: str, law_names: list[str]) -> str | None:
    """질의에 법령명 · 조문번호가 들어갔으면 그 이유.

    법령명은 공백을 무시하고 찾는다. 뒤에 공백 · 문장부호 · 조사가 오면 법령명으로 보고,
    다른 한글이 이어지면 다른 단어의 일부로 본다 ("감정평가법인" 의 "감정평가법")."""
    if not q.strip():
        return "빈 질의"
    if "「" in q or _ARTICLE_NO.search(q):
        return "조문번호"
    pos = [i for i, ch in enumerate(q) if not ch.isspace()]  # 공백 뺀 글자 → 원래 위치
    nq = "".join(q[i] for i in pos)
    for name in law_names:
        i = nq.find(name)
        while i >= 0:
            rest = q[pos[i + len(name) - 1] + 1 :]
            if not _HANGUL.match(rest[:1]) or rest.startswith(_PARTICLES):
                return f"법령명:{name}"
            i = nq.find(name, i + 1)
    return None


def law_names_for_leak(names: list[str]) -> list[str]:
    """유출 검사용 법령명. 너무 짧은 일반 단어는 빼고, 법 · 령 · 규칙으로 끝나는 이름만."""
    out = {norm_name(n) for n in names}
    return sorted(
        (n for n in out if len(n) >= 2 and re.search(r"(?:법|법률|령|규칙)$", n)),
        key=len,
        reverse=True,
    )


def build(
    generated: dict[str, list[Scenario]], law_names: list[str], generator: str
) -> tuple[list[EvalItem], dict[str, int], list[dict[str, Any]]]:
    """→ (평가셋, 통계, 검토용 행). 유출된 질의는 빼고, 질의가 하나도 안 남은 상황은 뺀다."""
    items: list[EvalItem] = []
    review: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for area in AREAS:
        for i, s in enumerate(generated.get(area.key, []), start=1):
            case = f"scn-{area.key}-{i:02d}"
            stats["scenarios"] += 1
            row: dict[str, Any] = {"case": case, "area": area.name, **s}
            kept = 0
            for typ in TYPES:
                q = str(s.get(typ) or "").strip()
                why = leak(q, law_names)
                if why:
                    row[f"{typ}_leak"] = why
                    stats["leaked"] += 1
                    continue
                kept += 1
                meta = {"case": case, "area": area.key, "persona": s.get("persona", ""),
                        "need": s.get("need", ""), "pool_queries": s.get("pool_queries", []),
                        "generator": generator}  # fmt: skip
                items.append(EvalItem(id=f"{case}-{typ}", query=q, gold=[],
                                      type=f"scenario_{typ}", source="scenario", meta=meta))  # fmt: skip
            if not kept:
                stats["dropped"] += 1
            review.append(row)
    stats["items"] = len(items)
    return items, dict(stats), review


def review_markdown(review: list[dict[str, Any]], stats: dict[str, int]) -> str:
    lines = ["# 평가셋 C 상황 시나리오 검토", "", f"- 통계: {stats}", ""]
    area = None
    for r in review:
        if r["area"] != area:
            area = r["area"]
            lines += [f"## {area}", ""]
        lines.append(f"### {r['case']} · {r.get('persona', '')}")
        lines.append(f"- 상황: {r.get('need', '')}")
        for typ in TYPES:
            flag = f" ⚠️ 유출({r[f'{typ}_leak']})" if r.get(f"{typ}_leak") else ""
            lines.append(f"- {typ}: {r.get(typ, '')}{flag}")
        lines.append(f"- pool 보조: {' / '.join(r.get('pool_queries') or [])}")
        lines.append("")
    return "\n".join(lines)
