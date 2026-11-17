"""검색 평가 지표. retrieved 는 순위 순서의 조문 키 목록."""

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """정답이 하나라도 상위 k 안에 있으면 1."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    for i, doc in enumerate(retrieved[:k] if k else retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gains: Mapping[str, int], k: int) -> float:
    """등급(gain)을 반영한 nDCG. 같은 조문이 여러 번 나오면 첫 번째만 센다."""
    if not gains:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in gains and doc not in seen:
            seen.add(doc)
            dcg += (2 ** gains[doc] - 1) / math.log2(i + 1)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]
