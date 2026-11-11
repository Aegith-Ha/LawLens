"""조문 텍스트 정리."""

import re

# <개정 2020. 1. 1.>, [본조신설 2019. 12. 3.], <신설 2015. 7. 24., 2020. 6. 9.> 등
_AMENDMENT_TAG = re.compile(
    r"[<\[]\s*(?:개정|신설|본조신설|전문개정|타법개정|삭제|종전)[^>\]]*[>\]]"
)
_WS = re.compile(r"[ \t　]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def strip_amendment_tags(text: str) -> str:
    return _AMENDMENT_TAG.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def clean_article_text(text: str) -> str:
    return normalize_whitespace(strip_amendment_tags(text))
