from datetime import date

from lawlens.domain.statute import Article, Law, LawVersion
from lawlens.indexing.chunker import chunk_law_version
from lawlens.indexing.milvus_schema import DENSE_FIELD, SPARSE_FIELD, law_schema
from lawlens.retrieval.hybrid import build_filter
from lawlens.retrieval.query import parse_query
from lawlens.timeutil import MAX_DATE


def test_parse_query_structured() -> None:
    pq = parse_query("민법 제750조 관련 판례 2019다12345 알려줘")
    assert pq.is_structured
    assert pq.case_numbers == ["2019다12345"]
    assert pq.article_refs[0].law_name == "민법"
    assert pq.article_refs[0].article_no == "750"


def test_parse_query_enforcement_decree() -> None:
    pq = parse_query("근로기준법 시행령 제23조 내용")
    assert pq.article_refs[0].law_name == "근로기준법 시행령"


def test_parse_query_natural_language() -> None:
    assert not parse_query("집주인이 보증금을 안 돌려줘요").is_structured


def test_build_filter_uses_validity_window() -> None:
    assert build_filter(today=date(2026, 11, 11)) == (
        "valid_from <= 20261111 and valid_to > 20261111"
    )
    assert build_filter(date(2020, 3, 1), ["법률"]) == (
        'valid_from <= 20200301 and valid_to > 20200301 and law_type in ["법률"]'
    )


def test_chunker_adds_header_validity_and_splits() -> None:
    law = Law(law_id="001706", name="민법", law_type="법률")
    art = Article(
        article_no="750",
        title="불법행위의 내용",
        text="고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        path=["제3편 채권", "제5장 불법행위"],
    )
    deleted = Article("751", None, "제751조 삭제", is_deleted=True)
    v = LawVersion("123", law, date(2025, 1, 31), articles=[art, deleted])
    [c] = chunk_law_version(v, date(2025, 1, 31), MAX_DATE)  # 삭제 조문은 제외
    assert c.chunk_id == "123_20250131:750:0"
    assert c.version_key == "123_20250131"
    assert (c.valid_from, c.valid_to) == (20250131, 99991231)
    assert c.text.startswith("「민법」 > 제3편 채권 > 제5장 불법행위 > 제750조(불법행위의 내용)\n")

    long = Article("1", None, "\n".join(["가" * 50] * 10))
    v2 = LawVersion("9", law, date(2025, 1, 1), articles=[long])
    assert len(chunk_law_version(v2, date(2025, 1, 1), MAX_DATE, max_chars=120)) > 1


def test_chunker_splits_single_long_line() -> None:
    """줄바꿈 없는 긴 줄도 max_chars 이하로 잘려 Milvus text 한도를 넘지 않는다."""
    law = Law(law_id="1", name="부가가치세법 시행령", law_type="대통령령")
    sentence = "가" * 30 + "한다. "
    one_line = sentence * 200 + "나" * 500  # 줄바꿈 없음, 마지막은 공백 없는 덩어리
    v = LawVersion("9", law, date(2025, 1, 1), articles=[Article("101", None, one_line)])
    chunks = chunk_law_version(v, date(2025, 1, 1), MAX_DATE, max_chars=300)
    bodies = [c.text.split("\n", 1)[1] for c in chunks]
    assert len(chunks) > 1
    assert all(len(b) <= 300 for b in bodies)
    assert bodies[0].endswith("한다.")  # 문장 경계에서 자름
    assert "".join(bodies).replace(" ", "") == one_line.replace(" ", "")  # 내용 손실 없음
    assert all(len(c.text.encode()) <= 8192 for c in chunks)


def test_milvus_schema_builds_offline() -> None:
    names = [f.name for f in law_schema(1024).fields]
    assert names[0] == "chunk_id"
    assert DENSE_FIELD in names and SPARSE_FIELD in names
    assert {"version_key", "valid_from", "valid_to"} <= set(names)
