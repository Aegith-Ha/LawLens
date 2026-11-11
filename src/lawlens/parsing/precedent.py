"""판례 본문 응답 → domain.precedent 모델.

TODO(Phase 1): 판시사항 · 판결요지 · 참조조문 · 참조판례 · 전문 분리.
"""

from typing import Any

from lawlens.domain.precedent import Precedent


def parse_precedent(raw: dict[str, Any]) -> Precedent:
    raise NotImplementedError("실제 응답 샘플 확보 후 구현")
