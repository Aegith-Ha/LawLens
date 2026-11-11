from eval.metrics import recall_at_k, reciprocal_rank


def test_metrics() -> None:
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, {"b", "z"}, k=2) == 0.5
    assert reciprocal_rank(retrieved, {"c"}) == 1 / 3
    assert reciprocal_rank(retrieved, {"z"}) == 0.0
