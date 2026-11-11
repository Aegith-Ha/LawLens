"""dense + sparse 벡터 인코더.

기본 구현은 bge-m3 (FlagEmbedding). `uv sync --extra ml` 로 설치한다.
"""

from dataclasses import dataclass
from typing import Any, Protocol

SparseVector = dict[int, float]


@dataclass
class Encoded:
    dense: list[list[float]]
    sparse: list[SparseVector]


class Encoder(Protocol):
    dim: int

    def encode_documents(self, texts: list[str]) -> Encoded: ...

    def encode_queries(self, texts: list[str]) -> Encoded: ...


class BGEM3Encoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel  # 무거운 의존성이므로 지연 import

        self._model: Any = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length
        self.dim = 1024

    def _encode(self, texts: list[str]) -> Encoded:
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
        )
        dense = [v.tolist() for v in out["dense_vecs"]]
        sparse = [{int(k): float(w) for k, w in lw.items()} for lw in out["lexical_weights"]]
        return Encoded(dense=dense, sparse=sparse)

    def encode_documents(self, texts: list[str]) -> Encoded:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> Encoded:
        return self._encode(texts)
