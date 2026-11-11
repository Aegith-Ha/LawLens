"""국가법령정보 Open API HTTP 클라이언트.

- 목록 조회: GET {base}/lawSearch.do?OC=..&target=..&type=JSON&query=..&page=..&display=..
- 본문 조회: GET {base}/lawService.do?OC=..&target=..&type=JSON&ID=..(또는 MST=..)
"""

import time
from collections.abc import Iterator
from typing import Any, Self

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from lawlens.config import Settings, get_settings
from lawlens.sources.lawgo.targets import SPECS, Target


class LawApiError(RuntimeError):
    pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class LawGoClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.law_api_oc:
            raise LawApiError("LAW_API_OC 가 설정되지 않았습니다 (.env 확인)")
        self._http = httpx.Client(
            base_url=self.settings.law_api_base_url,
            timeout=self.settings.law_api_timeout,
            transport=transport,
        )
        self._last_request = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ── low level ─────────────────────────────────────────
    def _throttle(self) -> None:
        wait = self.settings.law_api_min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, max=20),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        query = {"OC": self.settings.law_api_oc, "type": "JSON", **params}
        resp = self._http.get(path, params=query)
        resp.raise_for_status()
        try:
            data: dict[str, Any] = resp.json()
        except ValueError as e:
            # OC 미등록 등 오류 시 HTML 이 내려오는 경우가 있음
            raise LawApiError(f"JSON 이 아닌 응답: {resp.text[:200]!r}") from e
        return data

    # ── public API ────────────────────────────────────────
    def search(
        self,
        target: Target,
        query: str | None = None,
        page: int = 1,
        display: int = 100,
        **params: Any,
    ) -> dict[str, Any]:
        """목록 조회 원본 응답."""
        p: dict[str, Any] = {"target": target.value, "page": page, "display": display, **params}
        if query:
            p["query"] = query
        return self._get("/lawSearch.do", p)

    def get(self, target: Target, **id_params: Any) -> dict[str, Any]:
        """본문 조회 원본 응답. 예: get(Target.LAW, MST="123456")

        본문이 없으면 API 가 200 과 함께 {"Law": "일치하는 판례가 없습니다..."} 처럼
        문자열 하나만 돌려주므로 오류로 바꾼다. (예: 데이터출처가 '국세법령정보시스템'인 판례)
        """
        data = self._get("/lawService.do", {"target": target.value, **id_params})
        if len(data) == 1 and isinstance(next(iter(data.values())), str):
            raise LawApiError(f"본문 없음 {target} {id_params}: {next(iter(data.values()))}")
        return data

    def iter_search(
        self,
        target: Target,
        query: str | None = None,
        display: int = 100,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        """목록 조회를 페이지 끝까지 돌며 항목을 하나씩 반환."""
        page = 1
        while True:
            data = self.search(target, query=query, page=page, display=display, **params)
            total, items = extract_items(data, target)
            yield from items
            if not items or page * display >= total:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1


def extract_items(data: dict[str, Any], target: Target) -> tuple[int, list[dict[str, Any]]]:
    """목록 응답에서 (전체 건수, 항목 목록)을 꺼낸다.

    응답 형태: {"LawSearch": {"totalCnt": "123", "law": [...]}} 처럼
    최상위 키 하나 아래에 target 이름으로 항목이 들어 있다고 가정한다.
    결과가 1건이면 list 대신 dict 로 오는 경우도 처리한다.
    """
    if not data:
        return 0, []
    root = next(iter(data.values()))
    if not isinstance(root, dict):
        return 0, []
    total = int(root.get("totalCnt") or 0)
    spec = SPECS.get(target)
    items = root.get(spec.list_key if spec else target.value) or []
    if isinstance(items, dict):
        items = [items]
    return total, items
