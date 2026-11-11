"""색인 인터페이스. 동기화 로직은 Milvus 를 직접 알지 않고 이 인터페이스만 쓴다."""

from datetime import date
from typing import Protocol

from lawlens.domain.statute import LawVersion


class Indexer(Protocol):
    def index_version(self, version: LawVersion, valid_from: date, valid_to: date) -> int:
        """버전의 청크를 모두 지우고 다시 넣는다. 넣은 청크 수 반환."""
        ...

    def update_validity(self, version_key: str, valid_from: date, valid_to: date) -> None:
        """내용은 그대로, 유효기간만 바꾼다 (재임베딩 없음)."""
        ...

    def delete_version(self, version_key: str) -> None: ...
