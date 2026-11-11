"""Milvus collection 생성.

예) uv run python -m scripts.init_milvus [--drop]
"""

import typer

from lawlens.config import get_settings
from lawlens.indexing.milvus_schema import LAW_COLLECTION, ensure_law_collection
from lawlens.indexing.milvus_store import get_milvus_client

app = typer.Typer(add_completion=False)


@app.command()
def main(drop: bool = typer.Option(False, help="기존 collection 삭제 후 재생성")) -> None:
    settings = get_settings()
    client = get_milvus_client(settings)
    typer.echo(f"Milvus server: {client.get_server_version()}")
    created = ensure_law_collection(client, settings.dense_dim, drop=drop)
    typer.echo(f"{LAW_COLLECTION}: {'created' if created else 'already exists'}")


if __name__ == "__main__":
    app()
