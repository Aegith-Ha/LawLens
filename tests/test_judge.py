"""정답 다듬기 (pool · judge) 테스트 (DB · Milvus · Claude 없이)."""

import json
from pathlib import Path

import pytest

from eval.dataset import EvalItem, Gold
from eval.golden.judge import agreement, apply, from_answers, kappa, sample_for_human, seed_table
from eval.golden.pool import Candidate, PoolCase, case_id, group_cases, strip_article_refs


def _item(id_: str, gold: list[Gold]) -> EvalItem:
    return EvalItem(id=id_, query="q", gold=gold, type=id_.rsplit("-", 1)[1], source="expc")


def _pool() -> list[PoolCase]:
    cands = [
        Candidate("L1:2", "L1", "2", "가법", "[가법 제2조]", seed=2, found_by=["hybrid/user@1"]),
        Candidate("L1:3", "L1", "3", "가법", "[가법 제3조]", seed=1),
        Candidate("L2:5", "L2", "5", "나법", "[나법 제5조]", found_by=["dense/expert@4"]),
    ]
    return [PoolCase("expc-1", "상황", [{"id": "expc-1-user", "type": "expc_user", "query": "q"}],
                     {"L1:2": 2, "L1:3": 1}, cands)]  # fmt: skip


def test_case_id_groups_user_and_expert() -> None:
    a = _item("expc-1-user", [])
    b = _item("expc-1-expert", [])
    assert case_id(a) == case_id(b) == "expc-1"
    assert list(group_cases([a, b])) == ["expc-1"]
    c = EvalItem(id="x", query="q", gold=[], meta={"case": "s-9"})
    assert case_id(c) == "s-9"


def test_strip_article_refs() -> None:
    s = "「건축법」 제2조제1항제11호 나목 및 같은 법 제45조의2에 따른"
    out = strip_article_refs(s)
    assert "제2조" not in out and "제45조" not in out and "건축법" in out


def test_from_answers_defaults_to_zero(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"expc-1": {"2": ["L2:5"], "1": ["L1:2"]}}), "utf-8")
    j = from_answers(f)(_pool()[0])
    assert j == {"L2:5": 2, "L1:2": 1, "L1:3": 0}
    f.write_text(json.dumps({"expc-1": {"2": ["L9:9"]}}), "utf-8")
    with pytest.raises(KeyError):
        from_answers(f)(_pool()[0])
    f.write_text("{}", "utf-8")
    assert from_answers(f)(_pool()[0]) is None  # 아직 판정 안 한 상황


def test_apply_replaces_gold_and_drops_unjudged() -> None:
    seed = [Gold("2", "가법", "L1", 2), Gold("3", "가법", "L1", 1)]
    items = [_item("expc-1-user", seed), _item("expc-2-user", seed)]
    judged = {"expc-1": {"L1:2": 1, "L1:3": 0, "L2:5": 2}}
    out, stats = apply(items, _pool(), judged)
    assert stats == {"items": 2, "kept": 1, "unjudged": 1, "no_relevant": 0}
    assert out[0].gains() == {"L2:5": 2, "L1:2": 1}
    assert len(out[0].meta["seeds"]) == 2
    _, stats = apply(items[:1], _pool(), {"expc-1": {"L1:2": 0, "L1:3": 0, "L2:5": 0}})
    assert stats["no_relevant"] == 1


def test_seed_table() -> None:
    t = seed_table(_pool(), {"expc-1": {"L1:2": 2, "L1:3": 0, "L2:5": 1}})
    assert t["seed 2"][2] == 1 and t["seed 1"][0] == 1 and t["seed 아님"][1] == 1


def test_kappa_and_agreement() -> None:
    assert kappa([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)
    assert kappa([0, 0, 2, 2], [2, 2, 0, 0]) < 0
    # 2차 가중: 0↔1 불일치는 0↔2 보다 덜 나쁘다 (가중 없으면 둘이 같다)
    a, near, far = [0, 1, 2, 0, 1, 2], [1, 1, 2, 0, 1, 2], [2, 1, 2, 0, 1, 2]
    assert kappa(a, near) == pytest.approx(kappa(a, far))
    assert kappa(a, near, weighted=True) > kappa(a, far, weighted=True)
    judged = {"expc-1": {"L1:2": 2, "L1:3": 0, "L2:5": 1}}
    rows = sample_for_human(_pool(), judged, n=3)
    assert len(rows) == 3 and all(r["grade"] is None for r in rows)
    for r in rows:
        r["grade"] = judged["expc-1"][r["key"]]
    res = agreement(rows, judged)
    assert res["n"] == 3 and res["exact"] == 1.0


def test_pool_load_gzip(tmp_path: Path) -> None:
    import gzip

    from eval.golden.pool import load, save

    raw = tmp_path / "x.pool.jsonl"
    save(_pool(), raw)
    gz = tmp_path / "x.pool.jsonl.gz"
    gz.write_bytes(gzip.compress(raw.read_bytes()))
    assert load(gz) == load(raw) == _pool()
