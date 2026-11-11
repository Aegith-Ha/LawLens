"""답변에 붙이는 출처 표기."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Citation:
    source_type: str  # "law" | "prec" | ...
    source_id: str  # chunk_id 또는 원문 ID
    title: str  # 「민법」 제750조(불법행위의 내용) / 대법원 2019다12345
    as_of: date | None = None  # 법령: 시행일자, 판례: 선고일자

    def render(self) -> str:
        if self.as_of is None:
            return self.title
        label = "시행" if self.source_type == "law" else "선고"
        return f"{self.title} ({self.as_of:%Y. %-m. %-d.} {label})"
