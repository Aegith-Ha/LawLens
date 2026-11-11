"""FastAPI 앱.  실행: uv run uvicorn lawlens.api.main:app --reload"""

from fastapi import FastAPI

from lawlens import __version__
from lawlens.api.routers import search

app = FastAPI(title="LawLens", version=__version__)
app.include_router(search.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
