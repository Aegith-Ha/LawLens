"""조문 인용 추출: "제12조제1항", "제3조의2 제2항 제1호", 「민법」 제750조 등."""

import re
from dataclasses import dataclass

_REF = re.compile(
    r"(?:「(?P<law>[^」]+)」\s*)?"
    r"제(?P<article>\d+)조(?:의(?P<branch>\d+))?"
    r"(?:\s*제(?P<paragraph>\d+)항)?"
    r"(?:\s*제(?P<item>\d+)호)?"
)


@dataclass(frozen=True)
class ArticleRef:
    law_name: str | None  # None 이면 같은 법령 내 인용
    article: int
    branch: int | None = None  # 조문 가지번호 (제3조의2 → 2)
    paragraph: int | None = None
    item: int | None = None
    raw: str = ""

    @property
    def article_no(self) -> str:
        """'3' 또는 '3의2' 형태."""
        return f"{self.article}의{self.branch}" if self.branch else str(self.article)


def extract_article_refs(text: str) -> list[ArticleRef]:
    refs: list[ArticleRef] = []
    for m in _REF.finditer(text):
        refs.append(
            ArticleRef(
                law_name=m["law"],
                article=int(m["article"]),
                branch=int(m["branch"]) if m["branch"] else None,
                paragraph=int(m["paragraph"]) if m["paragraph"] else None,
                item=int(m["item"]) if m["item"] else None,
                raw=m.group(0),
            )
        )
    return refs
