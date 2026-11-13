import json
from datetime import date
from pathlib import Path

from lawlens.parsing.statute import parse_law

FIXTURE = Path(__file__).parent / "fixtures" / "eflaw_188376_20161220.json"


def test_parse_real_response_refugee_act() -> None:
    """실제 eflaw 응답 (난민법, MST 188376, 시행 2016-12-20)."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    v = parse_law(raw, mst="188376", effective_date=date(2016, 12, 20))

    assert v.version_key == "188376_20161220"
    assert (v.law.law_id, v.law.name, v.law.law_type) == ("011546", "난민법", "법률")
    assert v.law.ministry == "법무부"
    assert v.promulgation_date == date(2016, 12, 20)
    assert len(v.articles) == 48  # 조문여부 "조문" 48개 (장 제목 9개 제외)

    art1 = v.articles[0]
    assert art1.article_no == "1"
    assert art1.title == "목적"
    assert art1.path == ["제1장 총칙"]

    # 항이 객체 하나({"호": [...]})로 오는 조문
    art2 = next(a for a in v.articles if a.article_no == "2")
    assert '1. "난민"이란' in art2.text

    # 항 목록 + 호, 개정 표시 제거
    art5 = next(a for a in v.articles if a.article_no == "5")
    lines = art5.text.split("\n")
    assert lines[0] == "제5조(난민인정 신청)"
    assert "② 제1항에 따른 신청을 하는 때에는" in art5.text
    assert "1. 여권 또는 외국인등록증" in art5.text
    assert "<개정" not in art5.text


def _raw(units: list[dict[str, object]] | dict[str, object], **info: object) -> dict[str, object]:
    base = {"법령ID": "000001", "법령명_한글": "테스트법", "법종구분": {"content": "법률"}}
    return {"법령": {"기본정보": {**base, **info}, "조문": {"조문단위": units}}}


def test_parse_divisions_branch_mok_and_deleted() -> None:
    units = [
        {"조문여부": "전문", "조문내용": "제1장 총칙"},
        {"조문여부": "전문", "조문내용": "제1절 통칙"},
        {"조문여부": "조문", "조문번호": "1", "조문내용": "제1조(목적) 목적이다."},
        {"조문여부": "전문", "조문내용": "제2장 각칙"},  # 새 장 → 절 초기화
        {
            "조문여부": "조문",
            "조문번호": "3",
            "조문가지번호": "2",
            "조문제목": "정의",
            "조문내용": "제3조의2(정의)",
            "항": {
                "호": {
                    "호내용": "1. 다음 각 목",
                    "목": [{"목내용": "가. 첫째"}, {"목내용": ["나. 둘째", "(표)"]}],
                }
            },
        },
        {"조문여부": "조문", "조문번호": "4", "조문내용": "제4조 삭제 <2020. 1. 1.>"},
    ]
    v = parse_law(_raw(units, 시행일자="20250101"), mst="9")

    assert v.effective_date == date(2025, 1, 1)  # efYd 를 안 주면 응답의 시행일자
    a1, a3, a4 = v.articles
    assert a1.path == ["제1장 총칙", "제1절 통칙"]
    assert a3.article_no == "3의2"
    assert a3.label == "제3조의2(정의)"
    assert a3.path == ["제2장 각칙"]
    assert a3.text.split("\n") == [
        "제3조의2(정의)",
        "1. 다음 각 목",
        "가. 첫째",
        "나. 둘째",
        "(표)",
    ]
    assert a4.is_deleted and not a3.is_deleted


def test_requested_effective_date_wins() -> None:
    """같은 MST 에 시행일이 여러 개일 때 요청한 efYd 가 버전을 결정한다."""
    v = parse_law(_raw([], 시행일자="20210729"), mst="234435", effective_date=date(2022, 1, 1))
    assert v.version_key == "234435_20220101"


def test_single_body_law_without_articles() -> None:
    """조문 없이 본문만 "전문"(조문번호 0)으로 오는 법령 (예: 표준시에 관한 법률)."""
    units = {
        "조문여부": "전문",
        "조문번호": "0",
        "조문키": "0000000",
        "조문제목": "",
        "조문내용": "대한민국의 표준시는 동경 135도의   자오선을 기준으로 한다.",
    }
    v = parse_law(_raw(units, 시행일자="20110519"), mst="113175")

    (art,) = v.articles
    assert art.article_no == "0"
    assert art.label == "본문"
    assert art.title is None
    assert art.path == []
    assert art.text == "대한민국의 표준시는 동경 135도의 자오선을 기준으로 한다."
    assert not art.is_deleted


def test_heading_only_units_do_not_become_body() -> None:
    """조문이 있는 법령의 장 제목 · 비분류 "전문" 은 본문 조문으로 만들지 않는다."""
    units = [
        {"조문여부": "전문", "조문내용": "제1장 총칙"},
        {"조문여부": "전문", "조문내용": "제1속 굴착면의 기울기 등"},
        {"조문여부": "조문", "조문번호": "1", "조문내용": "제1조(목적) 목적이다."},
    ]
    v = parse_law(_raw(units, 시행일자="20250101"), mst="9")
    assert [a.article_no for a in v.articles] == ["1"]
