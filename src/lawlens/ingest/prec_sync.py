"""판례 동기화: 전체 배치 · 일배치. (아직 구현 전)

법령(law_sync.py)과 실행 단위·진행 상태를 분리한다. 한쪽이 실패해도 다른 쪽은 영향이 없다.
API 클라이언트, RawStore, DB 세션, Indexer 같은 인프라는 같이 쓴다.

예정된 처리 방식
    - 판례는 버전이 없다. 신규 추가와 삭제만 반영한다.
    - 전체 배치: prec 목록 → DB 에 없는 판례만 본문 조회 → raw 저장 → 파싱 → DB → Milvus
    - 일배치: 날짜 필터로 최근 등록분 조회 + delHst(판례 종류) 삭제 반영.
      진행 상태는 sync_state 의 PREC_DAILY_JOB 으로 법령과 따로 저장한다.
    - 참조조문은 "법령명 + 조문번호 + 선고일" 로만 저장하고, 검색 시 그 날짜에 유효한
      조문과 연결한다. 그래서 법령 배치와 실행 순서에 의존하지 않는다.
    - 국세청 출처 판례는 API 가 본문을 HTML 로만 줘서 조회가 실패한다(건너뜀).
"""

from datetime import date

PREC_DAILY_JOB = "prec_daily"


class PrecSync:
    def full(self, query: str | None = None, max_pages: int | None = None) -> None:
        raise NotImplementedError("판례 전체 배치는 아직 구현되지 않았습니다")

    def daily(
        self, since: date | None = None, until: date | None = None, overlap_days: int = 3
    ) -> None:
        raise NotImplementedError("판례 일배치는 아직 구현되지 않았습니다")
