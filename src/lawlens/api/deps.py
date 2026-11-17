"""API 의존성. 무거운 객체(Milvus 클라이언트, bge-m3)는 프로세스당 한 번만 만든다."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.orm import Session

from lawlens.config import get_settings
from lawlens.retrieval.hybrid import HybridSearcher
from lawlens.retrieval.service import SearchService
from lawlens.storage.db import get_sessionmaker


@lru_cache
def get_search_service() -> SearchService:
    from lawlens.indexing.milvus_store import get_milvus_client

    settings = get_settings()
    client = get_milvus_client(settings)

    def make_searcher() -> HybridSearcher:
        from lawlens.indexing.encoder import BGEM3Encoder

        return HybridSearcher(client, BGEM3Encoder(settings.embedding_model))

    return SearchService(client, searcher_factory=make_searcher)


def get_session() -> Iterator[Session]:
    with get_sessionmaker()() as session:
        yield session
