from lawlens.parsing.references import extract_article_refs
from lawlens.parsing.text import clean_article_text


def test_clean_article_text_removes_amendment_tags() -> None:
    raw = "제1조(목적) 이 법은  ...  목적으로 한다. <개정 2020. 1. 1.>\r\n\r\n\r\n[본조신설 2019. 12. 3.]"
    assert clean_article_text(raw) == "제1조(목적) 이 법은 ... 목적으로 한다."


def test_extract_article_refs() -> None:
    refs = extract_article_refs("「민법」 제750조 및 제3조의2 제2항 제1호에 따라")
    assert refs[0].law_name == "민법"
    assert refs[0].article_no == "750"
    assert refs[1].law_name is None
    assert refs[1].article_no == "3의2"
    assert (refs[1].paragraph, refs[1].item) == (2, 1)
