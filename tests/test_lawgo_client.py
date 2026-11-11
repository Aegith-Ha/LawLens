import httpx
import pytest

from lawlens.config import Settings
from lawlens.sources.lawgo.client import LawApiError, LawGoClient, extract_items
from lawlens.sources.lawgo.targets import Target


def _settings() -> Settings:
    return Settings(law_api_oc="test", law_api_min_interval=0)


def test_requires_oc() -> None:
    with pytest.raises(LawApiError):
        LawGoClient(Settings(law_api_oc=""))


def test_extract_items_single_dict() -> None:
    data = {"LawSearch": {"totalCnt": "1", "law": {"법령일련번호": "1"}}}
    assert extract_items(data, Target.LAW) == (1, [{"법령일련번호": "1"}])


def test_extract_items_eflaw_uses_law_key() -> None:
    data = {"LawSearch": {"totalCnt": "1", "law": [{"법령일련번호": "1"}]}}
    assert extract_items(data, Target.EFLAW) == (1, [{"법령일련번호": "1"}])


def test_iter_search_paginates_and_sends_params() -> None:
    seen: list[httpx.QueryParams] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.params)
        page = int(req.url.params["page"])
        items = [{"법령일련번호": f"{page}-{i}"} for i in range(2)]
        return httpx.Response(200, json={"LawSearch": {"totalCnt": "3", "law": items}})

    with LawGoClient(_settings(), transport=httpx.MockTransport(handler)) as c:
        got = list(c.iter_search(Target.LAW, query="민법", display=2))

    assert len(got) == 4  # 2 pages × 2 (totalCnt=3 → 2페이지에서 멈춤)
    assert seen[0]["OC"] == "test"
    assert seen[0]["type"] == "JSON"
    assert seen[0]["target"] == "law"
    assert seen[0]["query"] == "민법"
    assert [p["page"] for p in seen] == ["1", "2"]


def test_detail_not_found_message_raises() -> None:
    # 실제 응답: 본문을 제공하지 않는 판례(데이터출처: 국세법령정보시스템)
    body = {"Law": "일치하는 판례가 없습니다.  판례명을 확인하여 주십시오."}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    with LawGoClient(_settings(), transport=transport) as c, pytest.raises(LawApiError):
        c.get(Target.PREC, ID="625237")


def test_non_json_response_raises() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="<html>error</html>"))
    with LawGoClient(_settings(), transport=transport) as c, pytest.raises(LawApiError):
        c.get(Target.LAW, MST="1")
