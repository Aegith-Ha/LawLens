"""날짜 유틸. 법령 시행일은 한국 시간 기준으로 판단한다."""

from collections.abc import Iterator
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# 아직 다음 버전이 없는 법령 버전의 valid_to
MAX_DATE = date(9999, 12, 31)


def today_kst() -> date:
    return datetime.now(KST).date()


def now_kst() -> datetime:
    return datetime.now(KST)


def parse_yyyymmdd(value: str | int | None) -> date | None:
    """'20161220', '2016.12.20', 20161220 → date. 형식이 아니면 None."""
    s = str(value or "").strip().replace(".", "").replace("-", "")
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None


def to_yyyymmdd(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def date_range(start: date, end: date) -> Iterator[date]:
    """start ~ end (양 끝 포함)."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
