"""환경 변수(.env) 기반 설정."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 국가법령정보 Open API
    law_api_oc: str = ""
    law_api_base_url: str = "https://www.law.go.kr/DRF"
    law_api_timeout: float = 30.0
    law_api_min_interval: float = 0.2

    # 저장소
    database_url: str = "postgresql+psycopg://lawlens:lawlens@localhost:5432/lawlens"
    data_dir: Path = Path("data")

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"

    # 임베딩
    embedding_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024

    # LLM
    llm_model: str = "claude-sonnet-5-5"
    anthropic_api_key: SecretStr | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
