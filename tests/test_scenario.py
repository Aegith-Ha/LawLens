"""평가셋 C 시나리오 생성기 테스트 (DB · Codex 없이)."""

from eval.dataset import EvalItem
from eval.golden import scenario as S
from eval.golden.pool import case_id, need_from_meta

NAMES = S.law_names_for_leak(
    ["주택임대차보호법", "감정평가 및 감정평가사에 관한 법률", "민법", "상"]
)


def _scn(user: str, short: str) -> S.Scenario:
    return {"persona": "세입자", "need": "보증금을 못 받았다.", "user": user, "short": short,
            "pool_queries": ["임대차보증금 반환"]}  # fmt: skip


def test_leak() -> None:
    assert S.leak("주택임대차보호법상 보증금 반환", NAMES) == "법령명:주택임대차보호법"
    assert S.leak("민법 몇 조에 나와요?", NAMES) == "법령명:민법"
    assert S.leak("제3조의2 확정일자", NAMES) == "조문번호"
    assert S.leak("전세 보증금 못 받았어요", NAMES) is None
    assert S.leak("감정평가법인에 다녀요", ["감정평가법"]) is None  # 다른 단어의 일부
    assert S.leak("감정평가법 에 나와요", ["감정평가법"]) == "법령명:감정평가법"
    assert "상" not in NAMES  # 법 · 령 · 규칙으로 끝나지 않는 짧은 이름은 검사하지 않음


def test_build_items_and_drop_leaks() -> None:
    gen = {"housing": [_scn("보증금 언제 받아요?", "보증금 반환"), _scn("민법 보면 돼요?", "민법")]}
    items, stats, review = S.build(gen, NAMES, "test")
    assert [it.id for it in items] == ["scn-housing-01-user", "scn-housing-01-short"]
    assert stats == {"scenarios": 2, "leaked": 2, "dropped": 1, "items": 2}
    assert review[1]["user_leak"] == "법령명:민법"
    it = items[0]
    assert it.type == "scenario_user" and it.source == "scenario" and it.gold == []
    assert case_id(it) == case_id(items[1]) == "scn-housing-01"
    assert it.meta["pool_queries"] == ["임대차보증금 반환"]


def test_need_from_meta() -> None:
    it = EvalItem(id="scn-a-01-user", query="q", gold=[],
                  meta={"persona": "세입자", "need": "보증금을 못 받았다."})  # fmt: skip
    assert need_from_meta("scn-a-01", [it]) == "세입자: 보증금을 못 받았다."


def test_from_answers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    p = tmp_path / "a.json"
    p.write_text(json.dumps({"housing": [_scn("a", "b")] * 3}), "utf-8")
    gen = S.from_answers(p)
    assert len(gen(S.AREAS[0], 2)) == 2 and gen(S.AREAS[1], 2) == []
