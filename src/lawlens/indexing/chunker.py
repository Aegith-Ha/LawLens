"""조문 단위 청크 생성. 모든 청크 앞에 계층 헤더를 붙인다."""

from dataclasses import dataclass
from datetime import date

from lawlens.domain.statute import Article, LawVersion, make_article_id
from lawlens.timeutil import to_yyyymmdd

MAX_CHARS = 2000


@dataclass
class Chunk:
    chunk_id: str
    law_id: str
    mst: str
    version_key: str
    article_id: str
    law_name: str
    law_type: str
    article_no: str
    valid_from: int  # YYYYMMDD
    valid_to: int  # YYYYMMDD (다음 버전 시행일, 없으면 99991231)
    text: str


def build_header(law_name: str, article: Article) -> str:
    parts = [f"「{law_name}」", *article.path, article.label]
    return " > ".join(parts)


def _split_line(line: str, limit: int) -> list[str]:
    """limit 보다 긴 한 줄을 문장 끝(". ") → 공백 → 글자 수 순으로 자른다."""
    out: list[str] = []
    while len(line) > limit:
        window = line[: limit + 1]
        cut = window.rfind(". ") + 1  # 마침표까지 앞 조각에 포함
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        out.append(line[:cut].rstrip())
        line = line[cut:].lstrip()
    if line:
        out.append(line)
    return out


def _split(text: str, limit: int) -> list[str]:
    """항(①②…) 경계(줄바꿈)를 우선으로 limit 이하로 자른다.

    줄바꿈 없이 limit 을 넘는 줄(긴 단서 조항 등)은 문장 단위로 더 자른다.
    Milvus text 필드 한도(8192 바이트, 한글 1자 = 3바이트)를 넘지 않게 하기 위함이다.
    """
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    buf = ""
    lines = [piece for line in text.split("\n") for piece in _split_line(line, limit)]
    for line in lines:
        if buf and len(buf) + len(line) + 1 > limit:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{line}" if buf else line
    if buf:
        out.append(buf)
    return out


def chunk_law_version(
    version: LawVersion,
    valid_from: date,
    valid_to: date,
    max_chars: int = MAX_CHARS,
) -> list[Chunk]:
    """삭제된 조문("제7조 삭제")은 검색 대상에서 뺀다."""
    key = version.version_key
    vf, vt = to_yyyymmdd(valid_from), to_yyyymmdd(valid_to)
    chunks: list[Chunk] = []
    for art in version.articles:
        if art.is_deleted:
            continue
        header = build_header(version.law.name, art)
        article_id = make_article_id(key, art.article_no)
        for seq, body in enumerate(_split(art.text, max_chars)):
            chunks.append(
                Chunk(
                    chunk_id=f"{article_id}:{seq}",
                    law_id=version.law.law_id,
                    mst=version.mst,
                    version_key=key,
                    article_id=article_id,
                    law_name=version.law.name,
                    law_type=version.law.law_type,
                    article_no=art.article_no,
                    valid_from=vf,
                    valid_to=vt,
                    text=f"{header}\n{body}",
                )
            )
    return chunks
