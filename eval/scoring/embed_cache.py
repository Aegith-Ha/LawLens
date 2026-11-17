"""질의 임베딩 캐시 (평가용).

평가 시간의 대부분은 질의 임베딩이다 (CPU 에서 질의당 0.7~2.3초, Milvus 검색은 80ms 안팎).
평가셋 질의는 매번 같으므로 한 번 임베딩한 결과를 SQLite 파일에 저장해 다음 평가 때 다시 쓴다.

- 키: 모델명 · max_length · 질의 원문. 모델이나 설정이 바뀌면 키가 달라져 자동으로 새로 임베딩한다.
- 처음 임베딩에 걸린 시간(encode_ms)도 함께 저장한다. 캐시를 쓴 검색은 took_ms 에 임베딩이
  빠지므로 평가 쪽에서 이 값을 더해 실제 서비스 지연시간과 같은 기준으로 맞춘다.
- 같은 모델명으로 모델 파일만 바뀐 경우는 알 수 없으니 `--refresh-cache` 로 다시 임베딩해 덮어쓴다.
"""

import hashlib
import json
import sqlite3
import threading
import time
from array import array
from pathlib import Path

from lawlens.indexing.encoder import Encoded, Encoder

DEFAULT_PATH = Path("eval/cache/embeddings.sqlite")


class EmbeddingCache:
    """Encoder 를 감싸 질의 임베딩을 캐시한다.

    path 가 None 이면 메모리에만 둔다 (한 실행 안에서만). refresh 면 파일의 값을 읽지 않고
    새로 임베딩해 덮어쓴다.
    """

    def __init__(
        self,
        inner: Encoder,
        namespace: str,
        path: Path | None = DEFAULT_PATH,
        refresh: bool = False,
    ) -> None:
        self.inner = inner
        self.refresh = refresh
        self.dim = inner.dim
        self.namespace = namespace
        self._mem: dict[str, tuple[Encoded, int]] = {}
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute(
                "create table if not exists query_embedding ("
                " key text primary key, namespace text, query text,"
                " dense blob, sparse text, encode_ms integer, created_at text)"
            )
        # 마지막 encode_queries 호출 결과: 캐시에서 왔는지, 원래 임베딩에 걸린 시간
        self.hit = False
        self.encode_ms = 0
        self.hits = 0
        self.misses = 0

    def _key(self, query: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{query}".encode()).hexdigest()

    def _get(self, key: str) -> tuple[Encoded, int] | None:
        if key in self._mem:
            return self._mem[key]
        if self._db is None or self.refresh:
            return None
        row = self._db.execute(
            "select dense, sparse, encode_ms from query_embedding where key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        dense = array("f")
        dense.frombytes(row[0])
        sparse = {int(k): float(v) for k, v in json.loads(row[1]).items()}
        value = (Encoded(dense=[dense.tolist()], sparse=[sparse]), int(row[2]))
        self._mem[key] = value
        return value

    def _put(self, key: str, query: str, enc: Encoded, encode_ms: int) -> None:
        self._mem[key] = (enc, encode_ms)
        if self._db is None:
            return
        self._db.execute(
            "insert or replace into query_embedding values (?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                key,
                self.namespace,
                query,
                array("f", enc.dense[0]).tobytes(),
                json.dumps(enc.sparse[0]),
                encode_ms,
            ),
        )
        self._db.commit()

    def encode_documents(self, texts: list[str]) -> Encoded:
        return self.inner.encode_documents(texts)

    def encode_queries(self, texts: list[str]) -> Encoded:
        if len(texts) != 1:  # 검색은 질의 하나씩 부른다. 그 밖의 경우는 캐시하지 않는다.
            self.hit = False
            return self.inner.encode_queries(texts)
        query = texts[0]
        key = self._key(query)
        with self._lock:
            cached = self._get(key)
            if cached is not None:
                self.hit, self.encode_ms = True, cached[1]
                self.hits += 1
                return cached[0]
            started = time.perf_counter()
            enc = self.inner.encode_queries(texts)
            ms = round((time.perf_counter() - started) * 1000)
            self._put(key, query, enc, ms)
            self.hit, self.encode_ms = False, ms
            self.misses += 1
            return enc

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
