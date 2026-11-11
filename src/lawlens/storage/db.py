"""SQLAlchemy 엔진 · 세션."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lawlens.config import get_settings
from lawlens.storage.schema import Base


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """테이블이 없으면 만든다. (스키마 변경 관리는 이후 Alembic 도입 예정)"""
    Base.metadata.create_all(engine or get_engine())
