# LawLens

[국가법령정보 공동활용 Open API](https://open.law.go.kr/LSO/openApi/guideList.do)로 받은 법령·판례 데이터를 바탕으로 만드는 **법률 검색 RAG 시스템**입니다. 이후 단계에서 **법률 리서치 Agent**로 확장합니다.

> 이 문서는 프로젝트의 목표 구조와 설계 방향을 정리한 문서입니다. 아직 구현 전이며, 단계마다 내용을 업데이트합니다.

---

## 1. 목표

| 단계 | 목표 | 결과물 |
|---|---|---|
| **Phase 1. 데이터 수집** | Open API에서 법령·판례 등을 수집해 정규화된 형태로 저장 | 원본 저장소, 정규화 DB, 증분 동기화 |
| **Phase 2. 검색 (RAG)** | 조문 단위 하이브리드 검색과 출처가 달린 답변 생성 | 검색 API, 답변 API, 평가셋 |
| **Phase 3. Agent** | 여러 번 검색하고, 인용 조문을 따라가고, 판례를 교차 확인하는 리서치 Agent | Tool 기반 Agent, 대화 UI |

**원칙**
- **출처를 반드시 단다.** 모든 답변은 `법령명 + 조문번호 + 시행일자` 또는 `사건번호 + 선고일자`로 근거를 표시합니다.
- **시점을 구분한다.** 법령은 개정되므로 *어느 시점에 시행 중인 조문인지*를 데이터 모델의 기본 속성으로 둡니다.
- **원문을 보존한다.** 수집한 원본(XML/JSON)을 그대로 남겨 두어, 파싱 로직이 바뀌어도 다시 수집하지 않고 재처리할 수 있게 합니다.

---

## 2. 데이터 소스

Open API는 크게 **목록 조회**(`lawSearch.do`)와 **본문 조회**(`lawService.do`) 두 가지 엔드포인트를 제공하고, `target` 파라미터로 데이터 종류를 고릅니다. 호출하려면 신청 후 발급받는 `OC`(사용자 ID) 값이 필요합니다.

| 우선순위 | 데이터 | `target` (참고) | 활용 |
|---|---|---|---|
| ★★★ | 법령 (시행일 기준: 현행·연혁·시행예정) | `eflaw` | 핵심 검색 대상 (조문 단위). 목록 `nw=1,2,3`, 본문은 `MST`+`efYd` 필수 |
| ★★★ | 판례 | `prec` | 법령 해석·적용 사례. 국세청 출처 판례는 본문이 HTML만 제공됨 |
| ★★ | 헌재결정례 | `detc` | 위헌·헌법불합치 여부 |
| ★★ | 법령해석례 | `expc` | 법제처 유권해석 |
| ★★ | 법령용어 | `lstrm` | 동의어 사전, 쿼리 확장 |
| ★ | 행정규칙 / 자치법규 | `admrul`, `ordin` | 범위 확장 (데이터량이 많음) |
| ★ | 행정심판례, 조약, 별표·서식 등 | `decc`, `trty`, … | 필요 시 추가 |

> `target` 값, 요청·응답 필드는 [가이드 페이지](https://open.law.go.kr/LSO/openApi/guideList.do)를 기준으로 구현할 때 다시 확인합니다.

**Phase 1 범위:** 현행법령(법률·대통령령·부령) + 판례로 시작하고, 이후 헌재결정례와 법령해석례를 추가합니다.

---

## 3. 전체 아키텍처

```
                 ┌─────────────────────────────────────────────┐
                 │           open.law.go.kr Open API           │
                 └───────────────┬─────────────────────────────┘
                                 │ lawSearch.do / lawService.do
                                 ▼
┌──────────────────────── Ingestion (Phase 1) ─────────────────────────┐
│  Collector ──▶ Raw Store ──▶ Parser/Normalizer ──▶ Metadata DB       │
│  (rate limit,   (원본 XML/    (법령→편/장/절/조/항/   (PostgreSQL)    │
│   retry, 증분)    JSON 보관)     호/목, 판례→섹션)                     │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────── Indexing (Phase 2) ──────────────────────────┐
│  Chunker ──▶ Encoder ──────────▶ Milvus Standalone                   │
│  (조문 단위,   (dense + sparse     collection: law_chunks,            │
│   계층 컨텍스트)  벡터 생성)          precedent_chunks                  │
│                                  (dense/sparse 벡터 + 스칼라 메타데이터) │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────── Retrieval & Generation ──────────────────────┐
│  Query 분석 ─▶ Milvus hybrid_search ─▶ Rerank ─▶ Context ─▶ LLM 답변  │
│  (용어 정규화,   (dense+sparse, RRF,    (cross-   (인용 조문·  (출처 인용) │
│   법령명 인식)    시행일 필터)           encoder)  판례 확장)             │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────── Agent (Phase 3) ─────────────────────────────┐
│  Planner/LLM ◀──▶ Tools: search_law, get_article, search_precedent,  │
│                          follow_reference, get_history, live_api ... │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
                         FastAPI (REST/SSE)  ──▶  Web UI
```

---

## 4. 디렉터리 구조

```
lawlens/
├── README.md
├── pyproject.toml              # uv로 의존성 관리
├── .env.example                # LAW_API_OC, DB_URL, MILVUS_URI, LLM API KEY 등
├── docker-compose.yml          # milvus standalone v3.0.5(+etcd, minio), attu, postgres 18
│
├── src/lawlens/
│   ├── config.py               # pydantic-settings 기반 설정
│   │
│   ├── sources/                # ── Phase 1: 외부 API 클라이언트
│   │   └── lawgo/
│   │       ├── client.py       # HTTP 클라이언트 (rate limit, retry, 캐시)
│   │       ├── targets.py      # target별 요청 파라미터·응답 스키마
│   │
│   ├── ingest/                 # ── Phase 1: 수집 파이프라인
│   │   ├── collector.py        # 목록 조회 → 본문 조회 → raw 저장 (판례 등 raw 수집용)
│   │   ├── sync.py             # 법령 전체 배치 · 일배치 (6장 참고)
│   │   └── raw_store.py        # data/raw/{target}/{id}.json 저장·조회
│   │
│   ├── parsing/                # ── Phase 1: 정규화
│   │   ├── statute.py          # 법령 → 조/항/호/목 트리
│   │   ├── precedent.py        # 판례 → 판시사항/판결요지/참조조문/전문
│   │   ├── references.py       # "제12조제1항", "같은 법 시행령" 등 인용 추출
│   │   └── text.py             # 공백·한자 병기·개정 표시(<개정 …>) 정리
│   │
│   ├── domain/                 # 핵심 도메인 모델 (DB/인덱스와 무관)
│   │   ├── statute.py          # Law, LawVersion, Article, Paragraph ...
│   │   ├── precedent.py        # Precedent, Holding ...
│   │   └── citation.py         # Citation (출처 표기 규칙)
│   │
│   ├── storage/                # 저장소 계층
│   │   ├── db.py               # SQLAlchemy 엔진/세션, init_db
│   │   ├── schema.py           # 테이블 정의 (law, law_version, article, sync_state …)
│   │   └── repositories.py     # 버전 저장·삭제, 유효기간 계산, 색인 대기 목록
│   │
│   ├── indexing/               # ── Phase 2: 색인
│   │   ├── chunker.py          # 조문 단위 청크 + 계층 헤더 부착
│   │   ├── encoder.py          # dense + sparse 벡터 생성 (모델 추상화)
│   │   ├── milvus_schema.py    # collection 스키마·인덱스 정의
│   │   ├── indexer.py          # Indexer 인터페이스 (버전 색인 / 유효기간 갱신 / 삭제)
│   │   └── milvus_store.py     # MilvusIndexer 구현
│   │
│   ├── retrieval/              # ── Phase 2: 검색
│   │   ├── query.py            # 쿼리 분석: 법령명·조문번호·사건번호 인식, 용어 확장
│   │   ├── hybrid.py           # Milvus hybrid_search (dense + sparse, RRF) + 필터
│   │   ├── reranker.py         # cross-encoder 재정렬
│   │   └── expand.py           # 인용 조문·하위법령·관련 판례로 컨텍스트 확장
│   │
│   ├── generation/             # ── Phase 2: 답변 생성
│   │   ├── llm.py              # LLM 클라이언트 추상화
│   │   ├── prompts/            # 프롬프트 템플릿
│   │   └── answer.py           # 컨텍스트 → 인용 포함 답변, 근거 검증
│   │
│   ├── agent/                  # ── Phase 3: Agent
│   │   ├── tools.py            # retrieval/storage를 감싼 Tool 정의
│   │   ├── loop.py             # tool-use 루프 (계획 → 검색 → 검증 → 답변)
│   │   ├── memory.py           # 대화/세션 상태
│   │   └── policies.py         # 안전장치: 면책 문구, 근거 없는 단정 금지
│   │
│   └── api/                    # FastAPI
│       ├── main.py
│       └── routers/            # /search, /answer, /agent, /laws/{id}
│
├── scripts/                    # CLI 진입점 (typer)
│   ├── sync.py                 # 법령 동기화: full / daily / reindex
│   ├── ingest.py               # raw 수집만 (판례 등): --target prec --query 손해배상
│   ├── init_milvus.py          # collection 생성·인덱스 빌드
│   └── eval.py
│
├── eval/                       # 평가
│   ├── datasets/               # 질문 - 정답 조문/판례 쌍 (직접 구축)
│   └── metrics.py              # Recall@k, MRR, 인용 정확도
│
├── data/                       # (git ignore) raw/, processed/
├── notebooks/                  # 탐색용
├── web/                        # (Phase 3) 프론트엔드
└── tests/
    ├── fixtures/               # 실제 API 응답 샘플 (난민법 eflaw 본문)
    ├── test_statute_parser.py  # 실제 응답 파싱
    ├── test_validity.py        # 유효기간 계산
    ├── test_sync.py            # 전체·일배치 시나리오 (개정·정정·삭제·폐지)
    └── …
```

**레이어 의존 방향:** `api → agent → retrieval/generation → indexing → storage → domain`. `sources`와 `parsing`은 `ingest`에서만 사용합니다. Agent는 retrieval을 **Tool로 감싸서** 쓰기 때문에, Phase 2 코드를 고치지 않고 Phase 3을 올릴 수 있습니다.

---

## 5. 데이터 모델

법령은 **"법령(정체성)"과 "법령 버전(특정 시행일의 본문)"을 분리**해서 저장합니다.

```
Law              (law_id[법령ID], 법령명, 법령구분[법률/대통령령/부령…], 소관부처)
 └─ LawVersion   (version_key = {법령일련번호}_{시행일자}, law_id, 공포일자, 시행일자, 제·개정구분,
                  valid_from, valid_to, content_hash, Milvus 반영 상태)
     └─ Article  (article_id = {version_key}:{조문번호}, 조문제목, 편/장/절 경로, 본문(항·호·목 포함), 삭제 여부)

Precedent        (prec_id, 사건번호, 사건명, 법원명, 선고일자, 사건종류, 판시사항, 판결요지, 전문)
Reference        (from_type, from_id, to_type, to_id, raw_text)   # 조문↔조문, 판례→참조조문
Term             (용어, 정의, 동의어)                                 # 법령용어 사전
```

- `Reference` 테이블이 있으면 "이 조문을 인용한 판례", "이 법률의 위임을 받은 시행령 조문"을 그래프처럼 따라갈 수 있습니다. Agent에서 특히 유용합니다.
- **버전 식별자는 (법령일련번호, 시행일자) 쌍입니다.** 한 법령일련번호에 시행일자가 여러 개인 경우가 실제로 있습니다(부칙으로 일부 조문의 시행일을 늦춘 경우).
- **유효기간 `[valid_from, valid_to)`**: `valid_from`은 시행일자, `valid_to`는 같은 법령의 다음 버전 시행일자입니다. 다음 버전이 없으면 `9999-12-31`이고, 폐지 버전은 `valid_from`과 같아 빈 구간이 됩니다.
  - 현행 검색: `valid_from <= 오늘 < valid_to`
  - 과거 시점 검색("2020년 당시"): `valid_from <= 그날 < valid_to`
  - 시행예정 버전이 들어오는 순간 직전 버전의 `valid_to`가 정해지므로, **시행일이 지나면 아무 작업 없이 검색 결과가 바뀝니다.**

**역할 분담:** PostgreSQL은 원문·관계·버전 정보의 기준 저장소(source of truth)이고, Milvus는 검색용 인덱스입니다. Milvus 데이터는 언제든 PostgreSQL에서 다시 만들 수 있게 합니다.

### Milvus collection 스키마 (초안)

`law_chunks` (판례는 같은 구조의 `precedent_chunks`)

| 필드 | 타입 | 용도 |
|---|---|---|
| `chunk_id` | VARCHAR (PK) | `{version_key}:{article_no}:{seq}` 형태 |
| `version_key` | VARCHAR | `{법령일련번호}_{시행일자}`. 버전 단위 삭제·유효기간 갱신에 사용 |
| `law_id`, `mst`, `article_id` | VARCHAR | PostgreSQL 원문 조회용 |
| `law_name` | VARCHAR | 표시·필터 |
| `law_type` | VARCHAR | 법률 / 대통령령 / 부령 … 필터 (partition key 후보) |
| `article_no` | VARCHAR | 조문번호 (가지번호 포함, 예: `3의2`) |
| `valid_from` | INT64 | 시행일자 `YYYYMMDD` |
| `valid_to` | INT64 | 다음 버전 시행일자 `YYYYMMDD` (없으면 `99991231`) |
| `text` | VARCHAR | 계층 헤더가 붙은 청크 본문 |
| `dense_vector` | FLOAT_VECTOR | 의미 검색 (HNSW, COSINE) |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | 키워드 검색 (SPARSE_INVERTED_INDEX, IP) |

- `valid_from`, `valid_to`, `version_key`, `law_type`, `law_id`에 스칼라 인덱스를 겁니다.
- 내용이 바뀐 버전은 청크를 지우고 다시 넣습니다(재임베딩). 유효기간만 바뀐 버전은 `partial_update`로 `valid_from`/`valid_to`만 갱신합니다(재임베딩 없음).
- 이전 버전은 과거 시점 검색을 위해 지우지 않습니다. API에서 삭제된 데이터만 지웁니다.

---

## 6. 데이터 동기화 (전체 배치 · 일배치)

### 법이 바뀌는 경우와 처리

| 경우 | 감지 방법 | 처리 |
|---|---|---|
| 개정·제정 공포 (새 버전) | `lsHstInf`(법령 변경이력) `regDt=날짜` | 새 버전 저장 → 직전 버전 `valid_to` 확정 |
| 시행일 도래 (시행예정 → 현행) | 이벤트 없음 | 유효기간 필터로 자동 반영 (작업 없음) |
| 기존 버전 데이터 정정 | `lsHstInf`에 옛 버전도 나옴 | 원본 해시가 바뀌었으면 재저장·재색인 |
| 데이터 삭제 | `delHst`(삭제 데이터) `knd=1&delDt=날짜` | 해당 법령일련번호의 버전 삭제 |
| 폐지 | 개정과 같음 (제·개정구분 "폐지") | 폐지 버전은 빈 유효기간 → 이후 현행 없음 |

실제 호출해 보니 `lsHstInf`는 하루에 150~300건 정도의 버전을 돌려줍니다.

### 처리 흐름 (버전 하나)

```
본문 조회(eflaw, MST+efYd) → 원본 해시 비교 ─(같음)→ 건너뜀
                                └─(다름)→ raw 저장 → 파싱 → PostgreSQL 저장
→ 영향받은 법령의 유효기간 재계산 → Milvus 반영
     · 내용이 바뀐 버전: 청크 삭제 후 다시 넣기 (재임베딩)
     · 유효기간만 바뀐 버전: valid_from/valid_to 만 partial update
```

PostgreSQL의 `law_version`에 Milvus에 반영된 상태(`indexed_hash`, `indexed_valid_*`)를 함께 저장합니다. 그래서 Milvus 반영이 중간에 실패해도 `reindex`로 차이만 다시 맞출 수 있습니다.

### 배치 종류

| 명령 | 용도 | 동작 |
|---|---|---|
| `scripts.sync full` | 최초 적재, 가끔 점검 | `eflaw` 목록(기본 `nw=2,3`: 시행예정+현행)을 받습니다. DB에 이미 있는 버전은 건너뜁니다(`--refetch`로 강제). 목록에 없는 버전을 지우는 정리는 하지 않습니다(개정 전 버전이 현행 목록에서 빠지는 건 정상이므로). |
| `scripts.sync daily` | 매일 (cron) | 마지막 성공일 3일 전부터 오늘까지 하루씩 `lsHstInf` + `delHst`를 반영합니다. 하루가 끝날 때마다 진행 상태(`sync_state`)를 저장하므로 실패해도 그 날짜부터 이어서 합니다. |
| `scripts.sync reindex` | Milvus 복구 | API 호출 없이 DB ↔ Milvus 차이만 반영합니다. `--all`이면 전체 재색인(스키마·임베딩 모델 변경 시). |

- 모든 단계가 **멱등**이라 같은 날짜를 다시 돌려도 결과가 같습니다. 늦게 등록되는 데이터를 놓치지 않도록 일배치는 최근 3일을 겹쳐서 다시 확인합니다(`--overlap-days`).
- `--no-index`를 주면 PostgreSQL까지만 반영합니다. Milvus는 나중에 `reindex`로 한꺼번에 반영합니다.
- 판례 일배치는 아직 없습니다(판례 목록의 날짜 필터 파라미터 확인 후 추가 예정).

---

## 7. 검색(RAG) 설계 포인트

| 항목 | 방향 | 이유 |
|---|---|---|
| **청크 단위** | 기본 **조(條)** 단위. 너무 길면 항 단위로 나누고, 모든 청크 앞에 `「법령명」 제N장 > 제M조(제목)` 헤더를 붙임 | 법률 질의의 답은 대부분 특정 조문이고, 인용도 조문 단위로 함 |
| **판례 청크** | 판시사항·판결요지를 별도 청크로 만들고, 전문은 길이 기준으로 분할 | 요지만으로 검색되는 경우가 많음 |
| **하이브리드 검색** | Milvus `hybrid_search`로 dense + sparse 두 필드를 동시에 검색하고 `RRFRanker`로 결합 | 법률 용어나 조문번호는 정확히 일치해야 하는 경우가 많아 dense만으로는 부족. 별도 검색엔진 없이 Milvus 하나로 처리 |
| **Sparse 벡터** | ① `bge-m3`의 sparse 출력을 그대로 사용하거나 ② Kiwi 형태소 분석 기반 BM25 가중치를 직접 계산해 넣는 방식 중 평가로 결정 | 조사·어미 때문에 공백 기준 토큰화는 품질이 낮음. Milvus 내장 BM25 함수는 한국어 analyzer 지원 여부를 확인한 뒤 검토 |
| **임베딩** | 다국어/한국어 지원 모델(예: `bge-m3`, dense+sparse 동시 생성)로 시작하고 평가 후 교체 | `encoder.py` 추상화로 교체 비용을 낮춤 |
| **구조화 쿼리 우선** | "민법 제750조", "2019다12345" 같은 패턴은 정규식으로 인식해 DB에서 직접 조회 | 검색보다 빠르고 정확함 |
| **용어 확장** | 법령용어 사전으로 일상어 → 법률용어 확장 (예: "월세" → "차임") | 일반 사용자 질의와 법률 문언 사이의 차이를 메움 |
| **컨텍스트 확장** | 검색된 조문이 인용하는 조문, 정의 조항(제2조), 관련 판례 요지를 추가 | 조문 하나만으로는 해석이 불완전한 경우가 많음 |
| **답변 검증** | 답변에 나온 인용이 실제 컨텍스트에 있는지 후처리로 확인 | 존재하지 않는 조문을 만들어내는 문제 방지 |

---

## 8. Agent 확장 계획 (Phase 3)

Phase 2의 기능을 **Tool**로 노출하고, LLM이 여러 단계로 조사하게 합니다.

| Tool | 설명 |
|---|---|
| `search_law(query, as_of_date?)` | 조문 하이브리드 검색 |
| `get_article(law_name, article_no, as_of_date?)` | 특정 조문 원문 조회 |
| `search_precedent(query, court?, date_range?)` | 판례 검색 |
| `get_precedent(case_no)` | 판례 상세 (판시사항, 요지, 참조조문) |
| `follow_references(article_id)` | 인용·위임 관계 탐색 (법률 → 시행령 → 시행규칙) |
| `get_amendment_history(law_name, article_no)` | 조문 개정 연혁 |
| `lookup_term(term)` | 법령용어 정의 |
| `live_law_api(...)` | 색인에 없는 최신 자료를 Open API에서 실시간 조회 (fallback) |

**예상 흐름:** 질문 분석 → 관련 법령 탐색 → 조문 원문 확인 → 하위법령·정의 조항 확인 → 판례로 해석 확인 → 근거를 인용하며 답변.

**구현 방식:** 우선 LLM 제공사의 tool-use API로 직접 루프를 구현하고(`agent/loop.py`), 분기·체크포인트가 복잡해지면 LangGraph 같은 프레임워크 도입을 검토합니다. Tool은 MCP 서버로도 노출할 수 있게 설계해, 다른 Agent 클라이언트에서도 재사용할 수 있게 합니다.

---

## 9. 기술 스택 (초안)

| 영역 | 선택 | 비고 |
|---|---|---|
| 언어 | **Python 3.14**, `uv` | 버그 수정 지원 중(~2030-10). pymilvus 3.0.3·torch 2.15·kiwipiepy 0.24.1·sentence-transformers 6.2 모두 3.14 지원 확인 (2026-11 기준). 3.15는 pymilvus 지원 목록에 아직 없음 |
| API 서버 | FastAPI | 답변 스트리밍은 SSE |
| 메타데이터 DB | PostgreSQL | |
| 벡터 DB | **Milvus Standalone v3.0.5** (`milvusdb/milvus:v3.0.5`, Docker Compose: etcd + MinIO + milvus) | dense·sparse 하이브리드 검색과 스칼라 필터를 한 곳에서 처리. 관리 UI는 Attu |
| Milvus 클라이언트 | **`pymilvus>=3.0.2,<3.1`** | Milvus 3.0.5 릴리스 노트의 호환 Python SDK가 3.0.2. 2.6.x 클라이언트와 섞어 쓰지 않음 |
| 키워드 검색 | Milvus sparse 벡터 (bge-m3 sparse 또는 Kiwi 기반 BM25) | 별도 검색엔진을 두지 않음 |
| 임베딩 / 리랭커 | `bge-m3` / `bge-reranker-v2-m3` 등 | 평가 결과로 결정 |
| LLM | Claude 등, 설정으로 교체 가능 | `generation/llm.py`에서 추상화 |
| 작업 실행 | 처음에는 CLI 스크립트 + cron → 필요 시 Prefect 등 | |
| 테스트 / 품질 | pytest, ruff, mypy | |

---

## 10. 평가

검색 품질을 숫자로 보지 않으면 개선 방향을 정하기 어렵기 때문에, **Phase 2 초기에 평가셋을 먼저 만듭니다.**

- **데이터셋:** 질문 50~200개 + 정답 조문/판례 ID (직접 작성, 법령해석례의 질의–회답 활용 가능)
- **검색 지표:** Recall@k, MRR, nDCG
- **답변 지표:** 인용 정확도(인용한 조문이 실제로 존재하고 근거가 되는지), 근거 충실도(faithfulness)
- **회귀 방지:** 청킹·임베딩·프롬프트를 바꿀 때마다 `scripts/eval.py` 실행

---

## 11. 로드맵

- [ ] **Phase 1 – 수집**
  - [x] Open API `OC` 발급, `.env` 설정
  - [x] `lawgo` 클라이언트 (목록/본문, 페이지네이션, rate limit)
  - [x] 법령 파서 (조/항/호/목)
  - [x] PostgreSQL 스키마 + 유효기간 모델
  - [x] 전체 배치 · 일배치 (개정·정정·삭제·폐지 반영)
  - [ ] Docker 환경에서 PostgreSQL · Milvus 연동 확인
  - [ ] 현행법령 전체 적재 (`sync full`), 일배치 cron 등록
  - [ ] 판례 파서 · 판례 일배치
- [ ] **Phase 2 – RAG**
  - [ ] 평가셋 초안
  - [ ] Milvus Standalone 구성 + collection 스키마 생성
  - [ ] 조문 청킹 + dense/sparse 인코딩 + Milvus 적재
  - [ ] Milvus hybrid_search + 유효기간 필터
  - [ ] 리랭커, 구조화 쿼리 처리
  - [ ] 인용 포함 답변 생성 + 인용 검증
  - [ ] `/search`, `/answer` API
- [ ] **Phase 3 – Agent**
  - [ ] Tool 정의 및 Agent 루프
  - [ ] 참조 그래프 탐색, 개정 연혁
  - [ ] 웹 UI (대화, 근거 조문 패널)
  - [ ] (선택) MCP 서버로 Tool 노출

---

## 12. 시작하기

```bash
# 1. uv 설치 (최초 1회)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치 — Python 3.14 는 uv 가 자동으로 받음
uv sync                       # 기본 (API·수집·Milvus 클라이언트)
uv sync --extra ml            # + 임베딩/형태소 (torch, FlagEmbedding, kiwipiepy) — 수 GB

# 3. 환경 변수
cp .env.example .env          # LAW_API_OC=<발급받은 OC 값>

# 4. 인프라 (Milvus standalone + etcd + MinIO + Attu + PostgreSQL)
docker compose up -d          # Attu: http://localhost:3000

# 5. 법령 적재 · 색인 (6장 참고)
uv run python -m scripts.init_milvus
uv run python -m scripts.sync full --query 난민법      # 소량으로 먼저 확인
uv run python -m scripts.sync full                    # 현행+시행예정 약 6,700건 (API만 40분 내외, 임베딩 별도)
uv run python -m scripts.sync daily                   # 이후 매일 (cron)

# 6. API 서버
uv run uvicorn lawlens.api.main:app --reload  # http://localhost:8000/docs
```

**개발 명령**

```bash
uv run pytest          # 테스트
uv run ruff check .    # 린트
uv run ruff format .   # 포맷
uv run mypy src scripts  # 타입 체크
```

---

## 고지

LawLens는 법령 정보 검색을 돕는 도구이며 **법률 자문을 제공하지 않습니다.** 답변에는 항상 근거 조문과 시행일자를 함께 표시하고, 중요한 판단은 원문과 전문가 확인을 거치도록 안내합니다. 데이터의 출처는 법제처 국가법령정보센터이며, Open API 이용약관을 따릅니다.
