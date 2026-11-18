# 검색 평가

평가셋 질문으로 검색을 돌려서 정답 조문이 몇 위에 나오는지 측정합니다. 검색을 바꿀 때마다 같은 평가셋으로 다시 돌려 이전 결과와 비교합니다. 평가 설계의 배경은 [README 10장](../README.md#10-평가)에 있습니다.

평가 질의를 만드는 방법(A~D)과 정답 정하는 방식은 [golden/README.md](golden/README.md) 에 정리했습니다.

모든 명령은 프로젝트 폴더(`~/lawlens`)에서 실행합니다.

## 빠른 시작

```bash
# 0. 준비: docker compose 로 Milvus · PostgreSQL 이 떠 있어야 함
docker compose ps

# 1. 300문항만 뽑아 빠르게 (처음 약 9분, 같은 질의를 다시 돌리면 몇 초)
uv run python -m scripts.eval run eval/datasets/expc.jsonl --sample 300

# 2. 전체 (1,918문항, 처음 약 1시간)
uv run python -m scripts.eval run eval/datasets/expc.jsonl

# 결과 보기
cat eval/reports/$(ls eval/reports | tail -1)/report.md
```

시간이 오래 걸리면 터미널을 닫아도 계속 돌도록 백그라운드로 실행합니다.

```bash
nohup uv run python -m scripts.eval run eval/datasets/expc.jsonl > logs/eval.log 2>&1 &
```

```bash
tail -f logs/eval.log
```

## 평가 실행 — `run`

```bash
uv run python -m scripts.eval run <평가셋.jsonl> [<평가셋.jsonl> ...] [옵션]
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--modes` | `hybrid,dense,sparse` | 비교할 검색 방식. 쉼표로 조합 (예: `--modes hybrid`) |
| `--sample` | 없음 (전체) | 평가셋마다 무작위로 이만큼만 뽑기. 빠른 확인용 |
| `--seed` | `0` | `--sample` 난수 시드. 같으면 같은 문항이 뽑혀 결과를 비교할 수 있음 |
| `--out` | `eval/reports` | 결과를 저장할 폴더 |
| `--cache` | `eval/cache/embeddings.sqlite` | 질의 임베딩 캐시 파일 |
| `--refresh-cache` | 꺼짐 | 저장된 임베딩을 무시하고 새로 계산해 덮어씀 |
| `--no-cache` | 꺼짐 | 임베딩을 파일에 저장하지 않음 (한 번 실행하는 동안만 재사용) |

검색 방식:

| 방식 | 설명 |
|---|---|
| `hybrid` | dense + sparse 결과를 RRF로 합침 (검색 화면 기본값) |
| `dense` | bge-m3 dense 벡터(의미)만 사용 |
| `sparse` | bge-m3 sparse 가중치(키워드)만 사용 |

예시:

```bash
# hybrid 만, 100문항
uv run python -m scripts.eval run eval/datasets/expc.jsonl --modes hybrid --sample 100

# 평가셋 두 개를 함께 (B 를 만든 뒤)
uv run python -m scripts.eval run eval/datasets/expc.jsonl eval/datasets/synth.jsonl

# 모델 파일을 바꿨을 때 임베딩 다시 계산
uv run python -m scripts.eval run eval/datasets/expc.jsonl --refresh-cache
```

진행 상황은 10문항마다 출력됩니다.

```
expc: 10/300  경과 0.3분, 남은 시간 약 8.7분
```

검색 서버와 같은 bge-m3 모델을 따로 불러오므로 메모리가 2GB 정도 더 필요합니다. 검색 서버가 떠 있어도 함께 돌릴 수 있습니다.

### 질의 임베딩 캐시

평가 시간의 약 95%가 질의 임베딩입니다. CPU에서 질의당 0.7~2.3초가 걸리고, Milvus 검색은 80ms 안팎입니다.
그래서 한 번 임베딩한 질의는 `eval/cache/embeddings.sqlite`에 저장해 두고 다시 씁니다.
- hybrid · dense · sparse가 같은 임베딩을 함께 씁니다.
- 다음 평가 때도 그대로 씁니다. 처음 한 번만 임베딩하고, 그다음부터는 Milvus 검색만 합니다.
- 저장 키는 모델명 + `max_length` + 질의 원문입니다. 그래서 다른 모델로 바꾸면 자동으로 새로 임베딩합니다. 모델명은 같은데 모델 파일만 바뀌었다면 `--refresh-cache`를 씁니다.
- 처음 임베딩할 때 걸린 시간도 함께 저장해서 지연시간(p50 · p95)에 더합니다. 그래서 캐시를 써도 지연시간은 실제 검색 한 번과 같은 기준으로 나옵니다.
- 크기는 질의당 약 9KB입니다(전체 1,918문항이면 약 18MB). git 에는 올라가지 않습니다.

캐시가 효과를 보는 실험은 임베딩 이후 단계를 바꾸는 경우입니다(RRF 값, 후보 수, 필터, 결과 묶기, 리랭커).
임베딩 모델이나 질의 전처리를 바꾸면 새로 임베딩하므로 처음 실행과 같은 시간이 걸립니다.

## 결과

`eval/reports/<실행 시각>/` 폴더에 저장됩니다(git 에는 올라가지 않음).

| 파일 | 내용 |
|---|---|
| `report.md` | 평가셋 · 질의 유형 · 검색 방식별 지표 표, 실행 설정 |
| `items.jsonl` | 문항별 질의, 정답, 상위 10개 결과, 지표. 오답 분석용 |
| `config.json` | 실행 설정: 평가셋 · 문항 수, 방식, 임베딩 모델, 시드, git 커밋, 캐시 경로 |

`config.json`의 `git` 값에 `-dirty`가 붙어 있으면 커밋하지 않은 변경이 있는 상태에서 돌린 결과입니다.

### 지표

정답 단위는 **조문**(법령 + 조문번호)입니다. 같은 조문의 청크 여러 개는 하나로 묶어 순위를 매깁니다.
질의에 "법령명 제N조"가 들어 있어 조문 바로가기로 찾은 조문은 맨 앞 순위로 칩니다. 화면에 보이는 순서와 같습니다.

| 지표 | 뜻 | 볼 때 |
|---|---|---|
| `recall@5/10/20` | 정답 조문 중 상위 k 안에 든 비율 | 답변 생성(RAG)에 넘길 후보를 충분히 찾는지 |
| `hit@10` | 정답이 하나라도 상위 10 안에 든 비율 | 최소한 하나는 찾는지 |
| `mrr@10` | 첫 정답 순위의 역수 평균 (1위=1, 2위=0.5, 10위 밖=0) | 정답이 얼마나 위에 나오는지 |
| `ndcg@10` | 핵심(2) · 관련(1) 등급까지 반영한 순위 품질 | 핵심 조문을 관련 조문보다 위에 두는지 |
| `hit@1` | 1위가 정답인 비율 | 사용자가 첫 결과만 볼 때 |
| `p50_ms` · `p95_ms` | 질의당 검색 시간(임베딩 포함)의 중앙값 · 95% | 속도 |

표의 `유형`:
- `전체`: 평가셋의 모든 문항입니다.
- 그 아래 줄: 질의 유형별로 나눈 결과입니다.

| 유형 | 평가셋 | 설명 |
|---|---|---|
| `expc_masked` | C | 법령명 · 조문번호를 가린 질의 — **주 지표** |
| `expc_raw` | C | 원문 그대로 (법령명 · 조문번호 포함) |
| `situation` | B | 일반인이 상황을 설명하는 질문 |
| `term` | B | 용어 · 개념 질문 |

**읽는 법**
- 절대값보다 **같은 평가셋 · 같은 시드에서 바꾸기 전과 후의 차이**를 봅니다.
- 문항이 적은 유형은 표본 오차가 큽니다(145문항이면 약 ±4%p). 그보다 작은 차이는 문항 수를 늘려 다시 확인합니다.
- 정답이 불완전합니다. 실제로 관련 있는 조문이 정답 목록에 없을 수 있어서 점수가 실제보다 낮게 나올 수 있습니다. 오답은 `items.jsonl`에서 직접 확인합니다.

## 평가셋 만들기

평가셋은 `eval/datasets/`에 있습니다.

| 파일 | 평가셋 | 문항 |
|---|---|---|
| `expc.jsonl` | C. 법령해석례 | 1,918 (해석례 959건 × masked/raw) |
| `synth.jsonl` | B. 합성 | 아직 없음 (`ANTHROPIC_API_KEY` 필요) |

### C. 법령해석례 — `build-expc`

법제처 법령해석례의 질의요지를 질문으로 쓰고, 안건명 · 회답 · 질의요지에 인용된 조문을 정답으로 씁니다.

```bash
# 1) 원본 받기 (최근 순으로 10페이지 = 약 1,000건, data/raw/expc/)
uv run python -m scripts.ingest --target expc --sort ddes --max-pages 10

# 2) 평가셋으로 변환
uv run python -m scripts.eval build-expc
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--raw-dir` | `data/raw/expc` | 법령해석례 원본 폴더 |
| `--min-date` | `20150101` | 이 날짜(YYYYMMDD) 이후 해석례만. 조문 번호가 바뀌었을 가능성을 줄임 |
| `--output` | `eval/datasets/expc.jsonl` | 저장 경로 |

- 안건명 · 회답에 나온 조문은 핵심(2), 질의요지에만 나온 조문은 관련(1)입니다.
- 현행 조문만 정답으로 남깁니다.

### B. 합성 — `build-synth`

현행 조문을 법령 종류별 비율로 뽑습니다. Claude 가 각 조문에 대해 일반인 질문과 용어 질문을 만듭니다.
법령명, 조문번호, 원문 구절(5어절 이상)이 들어간 질문은 걸러 냅니다.

```bash
# .env 에 ANTHROPIC_API_KEY=... 를 넣은 뒤
uv run python -m scripts.eval build-synth --n 300
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--n` | `300` | 뽑을 조문 수 (조문당 질문 최대 2개) |
| `--seed` | `lawlens-eval-v1` | 같으면 같은 조문이 뽑힘 |
| `--workers` | `4` | 동시 요청 수 |
| `--output` | `eval/datasets/synth.jsonl` | 저장 경로 |

- 유료입니다. 조문 하나당 API 요청이 1번입니다.
- 조문 본문(공개 법령)이 Anthropic API로 전송됩니다.

### 정답 다듬기 — `pool` · `judge`

평가셋의 정답을 모든 소스에 공통인 방식으로 다시 정합니다. 배경은 [golden/README.md](golden/README.md)에 있습니다.

1. **후보 모으기 (`pool`)**
   - 상황(같은 해석례의 user · expert 질의)마다 검색 방식별 상위 결과와 기존 정답(seed)을 합칩니다.
   - 후보 순서는 섞습니다. 판정자가 순위나 seed 여부를 짐작하지 못하게 하기 위해서입니다.
2. **등급 판정 (`judge`)**
   - 후보마다 2(핵심) / 1(관련) / 0(무관)을 매깁니다.
   - 판정 결과로 정답을 바꾼 평가셋 `*.judged.jsonl`을 만듭니다.
3. **두 판정 합치기 (`judge-merge`)**
   - 서로 다른 판정기 두 개의 결과를 규칙으로 합칩니다. 0 ↔ 2로 크게 갈린 조문만 사람이 정합니다.
4. **사람 확인 (`judge-sample` · `judge-agree`)**
   - 표본을 사람이 판정해 판정기와의 일치도(kappa)를 봅니다.

```bash
uv run python -m scripts.eval pool eval/datasets/expc_rewrite.jsonl --depth 10
# → expc_rewrite.pool.jsonl

uv run python -m scripts.eval judge eval/datasets/expc_rewrite.pool.jsonl.gz
# Claude API 로 판정 (유료). 미리 판정한 파일이 있으면 --answers <파일>
# → .judgments.jsonl (후보별 등급), .judged.jsonl (새 정답 평가셋), .judge.md (검토 문서)

uv run python -m scripts.eval judge eval/datasets/expc_rewrite.pool.jsonl.gz --backend codex --tag codex --workers 3
# Codex CLI 로 판정 (로그인한 계정 사용량) → .codex.judgments.jsonl 등. 기존 판정을 덮어쓰지 않음

uv run python -m scripts.eval judge-merge eval/datasets/expc_rewrite.pool.jsonl.gz \
  --a eval/datasets/expc_rewrite.claude.judgments.jsonl --b eval/datasets/expc_rewrite.codex.judgments.jsonl \
  --overrides eval/datasets/expc_rewrite.judge_overrides.json --out eval/datasets/expc_rewrite.merged_answers.json
uv run python -m scripts.eval judge eval/datasets/expc_rewrite.pool.jsonl.gz --answers eval/datasets/expc_rewrite.merged_answers.json

uv run python -m scripts.eval judge-sample eval/datasets/expc_rewrite.pool.jsonl.gz --n 90
# → .human.jsonl 의 grade 를 0 · 1 · 2 로 채운 뒤
uv run python -m scripts.eval judge-agree eval/datasets/expc_rewrite.human.jsonl
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `pool --depth` | `20` | 검색 방식 · 질의마다 상위 몇 개를 모을지. 판정량이 이 값에 거의 비례합니다 (시범 50건: 10 → 1,486개, 20 → 2,813개) |
| `pool --need` | `auto` | 판정자가 보는 상황 설명. `expc`(해석례 안건명 · 질의요지 · 회답, 조문 번호는 지움) / `query`(질의 그대로) |
| `judge --answers` | 없음 | `{상황: {"2": [키], "1": [키]}}` JSON. 적지 않은 후보는 0. 키는 `law_id:조문번호` |
| `judge --backend` | `claude` | `--answers` 가 없을 때 판정기. `claude`(Anthropic API, 키 필요) / `codex`(`codex exec`, 빈 임시 폴더 · 읽기 전용 샌드박스 · 응답 스키마 고정) |
| `judge --tag` | 없음 | 결과 파일 이름에 붙일 표시. 다른 판정기 결과를 따로 둘 때 씁니다 |
| `judge-merge --overrides` | 없음 | 0 ↔ 2로 갈린 조문을 사람이 정한 `{상황: {키: 등급}}`. 없으면 1로 두고 미해결로 출력합니다 |

- pool 원본(`*.pool.jsonl`)은 커서 커밋하지 않고 gzip 압축본(`*.pool.jsonl.gz`)만 커밋합니다. 법령 sync · 날짜 · 검색 코드가 바뀌면 다시 만든 pool 이 판정 당시와 달라지므로, 판정에 쓴 pool 을 보관하기 위해서입니다. `judge` · `judge-sample` 은 압축본을 그대로 읽습니다.
- 판정되지 않은 상황, 관련 조문이 하나도 없는 상황은 `.judged.jsonl`에서 빠집니다. 원래 정답은 `meta.seeds`에 남습니다.
- `--depth 10`으로 모았으면 11~20위 결과는 판정받지 못해 오답으로 계산됩니다. 그래서 recall@20은 낮게 나올 수 있습니다.
- API 판정 시 상황 설명과 조문 본문(공개 자료)이 Anthropic API(`claude`) 또는 OpenAI(`codex`)로 전송됩니다.
- `judge` 는 결과 파일에 이미 판정된 상황을 다시 판정하지 않습니다. 사용량 한도 등으로 중간에 실패하면 같은 명령을 다시 실행하면 됩니다. Codex 상황별 로그는 `.cache/codex_judge/`에 남습니다.
- `judge-merge` 규칙: 같으면 그대로 / 1 차이는 seed 면 높은 쪽, 아니면 B / 0 ↔ 2 는 overrides. B 에는 조문 전문을 읽은 판정기를 둡니다.

### 조정용 · 최종 확인용 분할 — `split`

```bash
uv run python -m scripts.eval split eval/datasets/expc_rewrite.judged.jsonl --test 60
# → .judged.dev.jsonl (튜닝 · 비교용), .judged.test.jsonl (최종 확인용)
```

- 상황 단위로 나눕니다. 같은 해석례의 `user` · `expert` 질의는 항상 같은 쪽에 들어갑니다.
- 시드(`--seed`, 기본 `lawlens-split-v1`)가 같으면 항상 같은 분할입니다.
- 리랭커 등 검색 개선은 `.dev`로만 비교하고, `.test`는 마지막 확인에만 씁니다. `.test`를 보면서 고치면 그 셋에 맞춰져서 점수를 믿을 수 없게 됩니다.

### 평가셋 형식

한 줄에 한 문항인 JSON Lines입니다. 직접 만든 평가셋도 이 형식이면 `run`으로 돌릴 수 있습니다.

```json
{"id": "my-001", "query": "전세금을 못 돌려받으면 집을 경매에 넘길 수 있나요?", "type": "situation",
 "gold": [{"law_name": "주택임대차보호법", "article_no": "3의2", "grade": 2}]}
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `id` | ✓ | 문항 ID |
| `query` | ✓ | 질문 |
| `gold` | ✓ | 정답 조문 목록. `law_name` 또는 `law_id`, `article_no`(예: `3의2`), `grade`(2 = 핵심, 1 = 관련, 기본 1) |
| `type` | | 질의 유형. 결과 표에서 유형별로 나눠 집계 (기본 `general`) |
| `as_of` | | 기준일 `YYYY-MM-DD`. 없으면 오늘 시행 중인 법령 |
| `kinds` | | 법령 종류 필터 (`헌법`, `법률`, `대통령령`, `총리령·부령`, `규칙`) |
| `source`, `meta` | | 출처 · 부가 정보 (집계에는 안 씀) |

- 법령명은 띄어쓰기를 무시하고 맞춥니다.
- 이름으로 찾지 못한 정답이 있으면 실행할 때 알려 주고, 그 정답은 빼고 계산합니다.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 연결 오류 (Milvus · PostgreSQL) | `docker compose ps`로 컨테이너가 떠 있는지 |
| `법령명을 찾지 못한 정답 N개` | 법령명 오타이거나, 폐지 등으로 DB에 없는 법령 |
| 처음 실행이 오래 걸림 | 정상입니다. 질의 임베딩 때문이며, 같은 질의는 다음부터 캐시를 씁니다 |
| 결과가 이전과 다름 | `--sample`·`--seed`가 같은지, 데이터 동기화(sync)로 법령이 바뀌지 않았는지 `config.json`에서 확인 |
| `Anthropic 인증 정보가 없습니다` | `.env`에 `ANTHROPIC_API_KEY`가 있는지 (B 만들 때만 필요) |
