"""Evaluation harness, offline: deterministic gates hold; LLM grader skipped without a key."""

import evaluation


def test_deterministic_gates_are_perfect_offline():
    rep = evaluation.evaluate()
    d = rep["deterministic"]
    assert d["retrieval_hit_rate_at_k"] == 1.0
    assert d["numeric_grounding_rate"] == 1.0
    assert rep["version"] == evaluation.EVAL_VERSION


def test_qa_grader_skipped_offline():
    rep = evaluation.evaluate()
    assert rep["use_llm"] is False
    assert "qa_correctness" not in rep


def test_references_are_computed_from_data_not_hardcoded():
    # The total-sales reference must carry the real figure (so gold can't drift).
    cases = {c["id"]: c for c in evaluation.build_eval_set()}
    assert "1383220" in cases["total"]["reference"]
    assert "Widget A" in cases["best_product"]["reference"]
