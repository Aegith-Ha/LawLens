from pathlib import Path

import httpx

from lawlens.config import Settings
from lawlens.ingest.collector import collect
from lawlens.ingest.raw_store import RawStore
from lawlens.sources.lawgo.client import LawGoClient
from lawlens.sources.lawgo.targets import SPECS, Target


def test_eflaw_service_params_and_doc_id() -> None:
    spec = SPECS[Target.EFLAW]
    item = {"법령일련번호": "188376", "시행일자": "20161220"}
    assert spec.service_params(item) == {"MST": "188376", "efYd": "20161220"}
    assert spec.doc_id(item) == "188376_20161220"
    assert spec.service_params({"법령일련번호": "188376"}) == {}  # efYd 누락


def test_collect_eflaw(tmp_path: Path) -> None:
    calls: list[httpx.QueryParams] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.params)
        if req.url.path.endswith("lawSearch.do"):
            items = [
                {"법령일련번호": "1", "시행일자": "20250101"},
                {"법령일련번호": "2"},  # 시행일자 없음 → 실패로 집계
            ]
            return httpx.Response(200, json={"LawSearch": {"totalCnt": "2", "law": items}})
        return httpx.Response(200, json={"법령": {"기본정보": {}}})

    settings = Settings(law_api_oc="test", law_api_min_interval=0)
    store = RawStore(tmp_path)
    with LawGoClient(settings, transport=httpx.MockTransport(handler)) as c:
        stats = collect(c, store, Target.EFLAW, nw="3")

    assert (stats.listed, stats.fetched, stats.failed) == (2, 1, 1)
    assert calls[0]["target"] == "eflaw" and calls[0]["nw"] == "3"
    assert calls[1]["MST"] == "1" and calls[1]["efYd"] == "20250101"
    assert store.exists(Target.EFLAW, "1_20250101")
