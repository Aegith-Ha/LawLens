"""Open API target 정의.

가이드: https://open.law.go.kr/LSO/openApi/guideList.do
법령은 시행일 기준 `eflaw` 를 기본으로 쓴다 (연혁 · 시행예정 · 현행 버전을 모두 받을 수 있음).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Target(StrEnum):
    LAW = "law"  # 현행법령
    EFLAW = "eflaw"  # 시행일 법령
    PREC = "prec"  # 판례
    DETC = "detc"  # 헌재결정례
    EXPC = "expc"  # 법령해석례
    DECC = "decc"  # 행정심판례
    ADMRUL = "admrul"  # 행정규칙
    ORDIN = "ordin"  # 자치법규
    LSTRM = "lstrm"  # 법령용어
    TRTY = "trty"  # 조약
    # 변경 감지용 (일배치)
    LS_HST_INF = "lsHstInf"  # 법령 변경이력 목록: regDt=YYYYMMDD 에 등록·변경된 법령 버전
    DEL_HST = "delHst"  # 삭제 데이터 목록: knd=1(법령), delDt=YYYYMMDD


@dataclass(frozen=True)
class TargetSpec:
    target: Target
    # 목록 응답 항목에서 본문 조회용 ID를 꺼낼 필드명
    id_field: str
    # 본문 조회(lawService.do) 시 ID를 넘길 파라미터명
    service_param: str = "ID"
    # 목록 응답에서 항목 배열이 들어 있는 키. None 이면 target 값과 같음
    # (eflaw 는 항목이 "law" 키 아래로 온다 — 실제 응답으로 확인)
    item_key: str | None = None
    # 본문 조회에 함께 넘길 추가 파라미터 → 목록 항목의 필드명
    # (eflaw 본문 조회는 efYd=시행일자 가 필수)
    extra_params: dict[str, str] = field(default_factory=dict)

    @property
    def list_key(self) -> str:
        return self.item_key or self.target.value

    def service_params(self, item: dict[str, Any]) -> dict[str, str]:
        """목록 항목 → 본문 조회 파라미터. 필요한 값이 없으면 빈 dict."""
        params = {self.service_param: str(item.get(self.id_field) or "").strip()}
        for param, item_field in self.extra_params.items():
            params[param] = str(item.get(item_field) or "").strip()
        return params if all(params.values()) else {}

    def doc_id(self, item: dict[str, Any]) -> str:
        """raw 저장용 문서 ID. eflaw 는 {MST}_{시행일자} (한 MST 에 시행일이 여러 개일 수 있음)."""
        return "_".join(self.service_params(item).values())


SPECS: dict[Target, TargetSpec] = {
    Target.EFLAW: TargetSpec(
        Target.EFLAW,
        id_field="법령일련번호",
        service_param="MST",
        item_key="law",
        extra_params={"efYd": "시행일자"},
    ),
    # 목록 항목이 eflaw 와 같은 필드(법령일련번호, 시행일자)를 가지므로 본문은 eflaw 로 조회한다
    Target.LS_HST_INF: TargetSpec(
        Target.LS_HST_INF,
        id_field="법령일련번호",
        service_param="MST",
        item_key="law",
        extra_params={"efYd": "시행일자"},
    ),
    Target.DEL_HST: TargetSpec(Target.DEL_HST, id_field="일련번호", item_key="law"),
    Target.LAW: TargetSpec(Target.LAW, id_field="법령일련번호", service_param="MST"),
    Target.PREC: TargetSpec(Target.PREC, id_field="판례일련번호"),
    Target.DETC: TargetSpec(Target.DETC, id_field="헌재결정례일련번호"),
    Target.EXPC: TargetSpec(Target.EXPC, id_field="법령해석례일련번호"),
}
