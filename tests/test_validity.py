from datetime import date

from lawlens.storage.repositories import VersionOrder, compute_validity
from lawlens.timeutil import MAX_DATE


def v(key: str, eff: date, rev: str = "일부개정", prom: date | None = None) -> VersionOrder:
    return VersionOrder(key, key, eff, prom, rev)


def test_versions_chain_and_pending_version() -> None:
    got = compute_validity(
        [
            v("3", date(2027, 4, 21)),  # 시행예정
            v("1", date(2013, 7, 1), "제정"),
            v("2", date(2016, 12, 20)),
        ]
    )
    assert got == {
        "1": (date(2013, 7, 1), date(2016, 12, 20)),
        "2": (date(2016, 12, 20), date(2027, 4, 21)),  # 시행일이 오면 자동으로 3 으로 넘어감
        "3": (date(2027, 4, 21), MAX_DATE),
    }


def test_same_effective_date_later_promulgation_wins() -> None:
    got = compute_validity(
        [
            v("10", date(2025, 1, 1), prom=date(2024, 12, 1)),
            v("11", date(2025, 1, 1), prom=date(2024, 12, 20)),
        ]
    )
    assert got["10"] == (date(2025, 1, 1), date(2025, 1, 1))  # 빈 구간
    assert got["11"] == (date(2025, 1, 1), MAX_DATE)


def test_repeal_leaves_no_current_version() -> None:
    got = compute_validity([v("1", date(2010, 1, 1), "제정"), v("2", date(2020, 1, 1), "폐지")])
    assert got["1"] == (date(2010, 1, 1), date(2020, 1, 1))
    assert got["2"] == (date(2020, 1, 1), date(2020, 1, 1))


def test_repeal_and_reenact_is_not_repeal() -> None:
    got = compute_validity([v("1", date(2010, 1, 1)), v("2", date(2020, 1, 1), "폐지제정")])
    assert got["2"] == (date(2020, 1, 1), MAX_DATE)
