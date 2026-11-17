"""FastAPI 앱.  실행: uv run uvicorn lawlens.api.main:app

http://localhost:8000/        검색 화면
http://localhost:8000/docs    API 문서
"""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from lawlens import __version__
from lawlens.api.deps import get_search_service
from lawlens.api.routers import search

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger(__name__)

_model_state = {"status": "loading", "error": ""}


def _warm_up() -> None:
    """bge-m3 로딩(수십 초)을 서버 시작 직후 백그라운드에서 끝내 둔다."""
    try:
        get_search_service().searcher  # noqa: B018 - 지연 생성 트리거
        _model_state["status"] = "ready"
    except Exception as e:  # 검색 요청 때 다시 시도된다
        log.exception("검색 모델 로딩 실패")
        _model_state.update(status="error", error=str(e))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    yield


app = FastAPI(title="LawLens", version=__version__, lifespan=lifespan)
app.include_router(search.router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "model": _model_state["status"]}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
