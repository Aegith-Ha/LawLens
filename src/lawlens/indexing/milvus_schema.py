"""Milvus collection 스키마 · 인덱스 정의."""

from pymilvus import DataType, MilvusClient
from pymilvus.milvus_client.index import IndexParams
from pymilvus.orm.schema import CollectionSchema

LAW_COLLECTION = "law_chunks"
PRECEDENT_COLLECTION = "precedent_chunks"

DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"

LAW_OUTPUT_FIELDS = [
    "chunk_id",
    "law_id",
    "mst",
    "version_key",
    "article_id",
    "law_name",
    "law_type",
    "article_no",
    "valid_from",
    "valid_to",
    "text",
]


def law_schema(dense_dim: int) -> CollectionSchema:
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True)
    s.add_field("law_id", DataType.VARCHAR, max_length=32)
    s.add_field("mst", DataType.VARCHAR, max_length=32)
    s.add_field("version_key", DataType.VARCHAR, max_length=48)  # {MST}_{시행일자}
    s.add_field("article_id", DataType.VARCHAR, max_length=96)
    s.add_field("law_name", DataType.VARCHAR, max_length=300)
    s.add_field("law_type", DataType.VARCHAR, max_length=50)
    s.add_field("article_no", DataType.VARCHAR, max_length=24)
    s.add_field("valid_from", DataType.INT64)  # YYYYMMDD, 시행일자
    s.add_field("valid_to", DataType.INT64)  # YYYYMMDD, 다음 버전 시행일 (없으면 99991231)
    s.add_field("text", DataType.VARCHAR, max_length=8192)
    s.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dense_dim)
    s.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    return s


def law_index_params() -> IndexParams:
    p = MilvusClient.prepare_index_params()
    p.add_index(
        field_name=DENSE_FIELD,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    p.add_index(
        field_name=SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    # 필터 · 버전 단위 삭제/갱신에 쓰는 스칼라 필드
    p.add_index(field_name="valid_from", index_type="STL_SORT")
    p.add_index(field_name="valid_to", index_type="STL_SORT")
    p.add_index(field_name="version_key", index_type="INVERTED")
    p.add_index(field_name="law_type", index_type="INVERTED")
    p.add_index(field_name="law_id", index_type="INVERTED")
    return p


def ensure_law_collection(client: MilvusClient, dense_dim: int, drop: bool = False) -> bool:
    """collection 이 없으면 만든다. 새로 만들었으면 True."""
    if client.has_collection(LAW_COLLECTION):
        if not drop:
            return False
        client.drop_collection(LAW_COLLECTION)
    client.create_collection(
        collection_name=LAW_COLLECTION,
        schema=law_schema(dense_dim),
        index_params=law_index_params(),
    )
    return True
