"""법령 도메인 모델. 법령(정체성)과 법령 버전(특정 시행일의 본문)을 분리한다.

버전 식별자는 (법령일련번호 MST, 시행일자) 쌍이다. 한 MST 에 시행일자가 여러 개인 경우가
실제로 있다 (부칙으로 일부 조문의 시행일을 늦춘 경우).
"""

from dataclasses import dataclass, field
from datetime import date

# 법령 전체를 없애는 제·개정 구분. "폐지제정"은 새 법령 제정이므로 포함하지 않는다.
REPEAL_TYPES = frozenset({"폐지", "일괄폐지", "타법폐지"})


def make_version_key(mst: str, effective_date: date) -> str:
    return f"{mst}_{effective_date:%Y%m%d}"


def make_article_id(version_key: str, article_no: str) -> str:
    return f"{version_key}:{article_no}"


def is_repeal(revision_type: str | None) -> bool:
    return (revision_type or "").strip() in REPEAL_TYPES


@dataclass
class Law:
    law_id: str  # 법령ID (버전과 무관하게 고정)
    name: str
    law_type: str  # 법률 / 대통령령 / 부령 ...
    ministry: str | None = None


@dataclass
class Article:
    article_no: str  # "750", "3의2"
    title: str | None
    text: str
    path: list[str] = field(default_factory=list)  # ["제5편 채권", "제5장 불법행위"]
    is_deleted: bool = False  # "제7조 삭제 <2014.3.18>"

    @property
    def label(self) -> str:
        main, _, branch = self.article_no.partition("의")
        base = f"제{main}조의{branch}" if branch else f"제{main}조"
        return f"{base}({self.title})" if self.title else base


@dataclass
class LawVersion:
    mst: str  # 법령일련번호
    law: Law
    effective_date: date  # 시행일자
    promulgation_date: date | None = None
    promulgation_no: str | None = None
    revision_type: str | None = None  # 제정 / 일부개정 / 전부개정 / 폐지 ...
    articles: list[Article] = field(default_factory=list)

    @property
    def version_key(self) -> str:
        return make_version_key(self.mst, self.effective_date)

    @property
    def is_repeal(self) -> bool:
        return is_repeal(self.revision_type)
