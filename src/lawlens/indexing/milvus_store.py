"""Milvus 연결 및 Indexer 구현."""

from dataclasses import asdict
from datetime import date

from pymilvus import MilvusClient

from lawlens.config import Settings, get_settings
from lawlens.domain.statute import LawVersion
from lawlens.indexing.chunker import chunk_law_version
from lawlens.indexing.encoder import Encoder
from lawlens.indexing.milvus_schema import DENSE_FIELD, LAW_COLLECTION, SPARSE_FIELD
from lawlens.timeutil import to_yyyymmdd

# Milvus query 한 번에 돌려받을 수 있는 최대 행 수
_QUERY_LIMIT = 16384


def get_milvus_client(settings: Settings | None = None) -> MilvusClient:
    s = settings or get_settings()
    return MilvusClient(uri=s.milvus_uri, token=s.milvus_token, db_name=s.milvus_db_name)


def _key_filter(version_key: str) -> str:
    return f'version_key == "{version_key}"'


class MilvusIndexer:
    def __init__(
        self,
        client: MilvusClient,
        encoder: Encoder,
        collection: str = LAW_COLLECTION,
        batch_size: int = 64,
    ) -> None:
        self.client = client
        self.encoder = encoder
        self.collection = collection
        self.batch_size = batch_size

    def index_version(self, version: LawVersion, valid_from: date, valid_to: date) -> int:
        self.delete_version(version.version_key)
        chunks = chunk_law_version(version, valid_from, valid_to)
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            enc = self.encoder.encode_documents([c.text for c in batch])
            rows = [
                {**asdict(c), DENSE_FIELD: d, SPARSE_FIELD: s}
                for c, d, s in zip(batch, enc.dense, enc.sparse, strict=True)
            ]
            self.client.insert(collection_name=self.collection, data=rows)
        return len(chunks)

    def update_validity(self, version_key: str, valid_from: date, valid_to: date) -> None:
        hits = self.client.query(
            collection_name=self.collection,
            filter=_key_filter(version_key),
            output_fields=["chunk_id"],
            limit=_QUERY_LIMIT,
        )
        rows = [
            {
                "chunk_id": h["chunk_id"],
                "valid_from": to_yyyymmdd(valid_from),
                "valid_to": to_yyyymmdd(valid_to),
            }
            for h in hits
        ]
        if rows:
            self.client.upsert(collection_name=self.collection, data=rows, partial_update=True)

    def delete_version(self, version_key: str) -> None:
        self.client.delete(collection_name=self.collection, filter=_key_filter(version_key))
