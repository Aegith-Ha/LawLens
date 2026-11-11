# scripts

명령줄 실행 파일 모음입니다. **프로젝트 루트(`lawlens/`)에서** `uv run python -m scripts.<이름>` 형태로 실행합니다.

모든 명령은 `--help`로 옵션을 볼 수 있습니다.

```bash
uv run python -m scripts.sync --help
```

| 파일 | 용도 | 필요한 것 |
|---|---|---|
| [sync.py](sync.py) | 법령 수집·적재·색인 (전체 배치 / 일배치 / 재색인) | `.env`의 `LAW_API_OC`, PostgreSQL, Milvus |
| [ingest.py](ingest.py) | 원본(raw) 수집만. 현재는 판례 수집용 | `LAW_API_OC` |
| [init_milvus.py](init_milvus.py) | Milvus collection 생성 | Milvus |
| [eval.py](eval.py) | 검색 평가 (아직 구현 전) | – |

PostgreSQL과 Milvus는 `docker compose up -d`로 띄웁니다.

---

## sync.py — 법령 동기화

API에서 법령(`eflaw`)을 받아 **원본 저장 → 파싱 → PostgreSQL → Milvus** 순서로 반영합니다. 테이블은 처음 실행할 때 자동으로 만들어집니다. 처리 방식은 [README 6장](../README.md#6-데이터-동기화-전체-배치--일배치)에 있습니다.

### full — 전체 배치

최초 적재할 때 씁니다. DB에 이미 있는 버전은 건너뜁니다.

```bash
uv run python -m scripts.sync full --query 난민법   # 일부만 (동작 확인용)
uv run python -m scripts.sync full                 # 현행 + 시행예정 전체 (약 6,700건)
uv run python -m scripts.sync full --nw 1,2,3      # 연혁까지 전부 (약 17만 건)
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--nw` | `2,3` | 1: 연혁, 2: 시행예정, 3: 현행 (쉼표로 조합) |
| `--query` | 없음 (전체) | 법령명 검색어 |
| `--max-pages` | 없음 | 최대 페이지 수 (1페이지 = 100건) |
| `--refetch` | 끔 | DB에 있는 버전도 다시 조회 |
| `--no-index` | 끔 | Milvus 반영 생략 (PostgreSQL까지만) |

### daily — 일배치

매일 cron으로 실행합니다. 마지막 성공일 3일 전부터 오늘까지 하루씩 **법령 변경이력**과 **삭제 데이터**를 반영합니다. 중간에 실패해도 다음 실행 때 이어서 합니다.

```bash
uv run python -m scripts.sync daily
uv run python -m scripts.sync daily --since 20261101 --until 20261110   # 기간 지정
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--since` | 마지막 성공일 − 겹침 일수 | 시작일 `YYYYMMDD` |
| `--until` | 오늘 | 종료일 `YYYYMMDD` |
| `--overlap-days` | `3` | 마지막 성공일 이전 며칠을 다시 확인할지 |
| `--no-index` | 끔 | Milvus 반영 생략 |

cron 예시 (매일 새벽 3시):

```cron
0 3 * * * cd /home/<사용자>/lawlens && ~/.local/bin/uv run python -m scripts.sync daily >> logs/sync.log 2>&1
```

### reindex — Milvus 재색인

API를 호출하지 않고 PostgreSQL과 Milvus의 차이만 맞춥니다. `--no-index`로 적재했거나 Milvus 반영이 실패했을 때 씁니다.

```bash
uv run python -m scripts.sync reindex         # 차이만
uv run python -m scripts.sync reindex --all   # 전체 재색인 (스키마·임베딩 모델 변경 시)
```

---

## ingest.py — 원본 수집

API 응답을 `data/raw/{target}/` 에 JSON으로 저장만 합니다(DB·Milvus 반영 없음). 이미 받은 문서는 건너뜁니다. 법령은 `sync.py`를 쓰고, 이 파일은 판례처럼 아직 파서가 없는 데이터를 미리 받아 둘 때 씁니다.

```bash
uv run python -m scripts.ingest --target prec --query 손해배상 --max-pages 1
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--target` | `eflaw` | `prec`(판례), `eflaw`(법령) 등 |
| `--query` | 없음 (전체) | 검색어 |
| `--nw` | `3` | `eflaw` 전용. 1: 연혁, 2: 시행예정, 3: 현행 |
| `--max-pages` | 없음 | 최대 페이지 수 |
| `--overwrite` | 끔 | 이미 받은 문서도 다시 받기 |

국세청 출처 판례는 API가 본문을 HTML로만 제공해서 실패로 집계됩니다(정상).

---

## init_milvus.py — Milvus collection 생성

`law_chunks` collection과 인덱스를 만듭니다. `sync.py`도 실행할 때 없으면 자동으로 만들기 때문에, 보통은 스키마를 바꿨을 때만 `--drop`과 함께 씁니다.

```bash
uv run python -m scripts.init_milvus
uv run python -m scripts.init_milvus --drop   # 기존 collection 삭제 후 재생성 → 이후 sync reindex --all
```

---

## eval.py — 검색 평가 (구현 예정)

`eval/datasets/*.jsonl`의 질문으로 검색해 Recall@k, MRR을 계산할 예정입니다. 지금은 실행하면 "아직 구현되지 않았습니다"를 출력하고 종료합니다.
