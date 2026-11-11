"""검색 평가: eval/datasets/*.jsonl 의 질문으로 검색해 Recall@k, MRR 계산.

데이터 형식 (한 줄에 하나):
{"question": "...", "relevant": ["{법령ID}:{article_no}", ...]}
(버전이 바뀌어도 정답이 유지되도록 version_key 대신 법령ID 기준)

TODO(Phase 2): HybridSearcher 연결.
"""

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    typer.echo("아직 구현되지 않았습니다 (HybridSearcher 연결 필요)", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
