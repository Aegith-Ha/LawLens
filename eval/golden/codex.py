"""Codex CLI (`codex exec`) 를 JSON 응답 함수처럼 쓴다. 로그인한 ChatGPT 계정의 사용량을 쓴다.

도구를 쓰지 않도록 빈 임시 폴더에서 읽기 전용 샌드박스로 실행하고, 응답은 JSON 스키마로 고정한다.
계정 정보(~/.codex)는 codex 가 직접 읽는다. 이 모듈은 읽지 않는다.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def run_json(
    text: str,
    schema: dict[str, Any],
    model: str | None = None,
    timeout: int = 900,
    log: Path | None = None,
) -> Any:
    """text 를 프롬프트로 보내고, schema 에 맞는 JSON 응답을 파싱해 돌려준다."""
    with tempfile.TemporaryDirectory() as tmp:
        schema_path, out = Path(tmp, "schema.json"), Path(tmp, "out.json")
        schema_path.write_text(json.dumps(schema), "utf-8")
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "-C", tmp, "--output-schema", str(schema_path), "-o", str(out),
               "--color", "never"]  # fmt: skip
        if model:
            cmd += ["-m", model]
        r = subprocess.run(
            [*cmd, "-"], input=text, capture_output=True, text=True, timeout=timeout
        )  # fmt: skip
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex 종료 코드 {r.returncode}: {r.stderr[-500:]}")
        return json.loads(out.read_text("utf-8"))
