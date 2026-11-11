"""LLM 클라이언트 추상화. 제공사를 바꿔도 generation/agent 코드는 그대로 둔다.

TODO(Phase 2): Anthropic SDK 구현체 추가 (`uv add anthropic`).
"""

from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str: ...
