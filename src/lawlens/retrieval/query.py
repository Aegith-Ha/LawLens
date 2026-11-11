"""쿼리 분석: 구조화된 참조(조문번호, 사건번호)는 검색 대신 DB 직접 조회로 보낸다."""

import re
from dataclasses import dataclass, field

from lawlens.parsing.references import ArticleRef, extract_article_refs

# 사건번호: 2019다12345, 2020도1234, 2018헌바123, 2021두56789 ...
_CASE_NO = re.compile(r"\b(?:19|20)\d{2}\s?[가-힣]{1,3}\s?\d{1,6}\b")

# 법령명 + 조문: "민법 제750조", "근로기준법 시행령 제23조" (「」 없이 쓴 경우)
# 공백이 들어간 긴 법령명("…에 관한 법률")은 「」로 감싼 경우만 확실히 인식한다.
_LAW_WITH_ARTICLE = re.compile(
    r"(?P<law>[가-힣A-Za-z·]{0,40}(?:법률|법|령|규칙|규정)(?:\s시행령|\s시행규칙)?)\s*(?=제\d+조)"
)


@dataclass
class ParsedQuery:
    text: str
    case_numbers: list[str] = field(default_factory=list)
    article_refs: list[ArticleRef] = field(default_factory=list)

    @property
    def is_structured(self) -> bool:
        return bool(self.case_numbers or any(r.law_name for r in self.article_refs))


def parse_query(text: str) -> ParsedQuery:
    case_numbers = [m.group(0).replace(" ", "") for m in _CASE_NO.finditer(text)]

    refs = extract_article_refs(text)
    # 「」 없이 쓴 법령명을 보완
    law_names = [m["law"].strip() for m in _LAW_WITH_ARTICLE.finditer(text)]
    if law_names:
        refs = [
            ArticleRef(
                law_name=r.law_name or law_names[min(i, len(law_names) - 1)],
                article=r.article,
                branch=r.branch,
                paragraph=r.paragraph,
                item=r.item,
                raw=r.raw,
            )
            for i, r in enumerate(refs)
        ]
    return ParsedQuery(text=text, case_numbers=case_numbers, article_refs=refs)
