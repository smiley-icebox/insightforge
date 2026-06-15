"""Evaluation — the brief's QAEvalChain plus deterministic gates over a versioned set.

Two tracks (same discipline as the rest of the build):
  Deterministic (always, no key): retrieval precision (did the right stat block get
    retrieved?) and numeric grounding (does every figure in the answer trace to the
    retrieved statistics?).
  LLM-graded (USE_LLM): LangChain QAEvalChain scores each answer against a reference.

The references are COMPUTED from analytics, not hard-coded — so the gold answers can never
drift from the data. The set is small, fixed, and checked in; report against it every run.
"""

import analytics
import config
import grounding
import llm
import rag

EVAL_VERSION = "2026-06-14-v1"


def build_eval_set() -> list[dict]:
    """Questions paired with references computed from the data (always the true figures)."""
    ov = analytics.overview()
    prod = analytics.sales_by_product()
    region = analytics.sales_by_region()
    stats = analytics.statistical_measures()
    seg = analytics.customer_segmentation()
    top_age = seg["by_age"].sort_values("n", ascending=False).iloc[0]
    # Peak among FULL months (mirrors the time block's partial-period guard).
    months = analytics.sales_by_period(freq="month")
    full = months[months["n"] >= 0.5 * months["n"].median()]
    peak = full.loc[full["total_sales"].idxmax()]
    return [
        {"id": "total", "question": "What were the total sales across all records?",
         "reference": f"Total sales is {ov['total_sales']}.", "expected_stat": "overview"},
        {"id": "best_product", "question": "Which product had the highest total sales?",
         "reference": f"{prod.iloc[0]['Product']} with {int(prod.iloc[0]['total_sales'])} in total sales.",
         "expected_stat": "product"},
        {"id": "best_region", "question": "Which region had the highest total sales?",
         "reference": f"{region.iloc[0]['Region']} with {int(region.iloc[0]['total_sales'])} in total sales.",
         "expected_stat": "region"},
        {"id": "median", "question": "What is the median sale value?",
         "reference": f"The median sale value is {stats['Sales']['median']}.",
         "expected_stat": "stats"},
        {"id": "std", "question": "What is the standard deviation of sales?",
         "reference": f"The standard deviation of sales is {stats['Sales']['std']}.",
         "expected_stat": "stats"},
        {"id": "age_band", "question": "Which customer age band has the most customers?",
         "reference": f"The {top_age['AgeBand']} age band, with {int(top_age['n'])} customers.",
         "expected_stat": "demographics"},
        {"id": "trend", "question": "What was the peak sales month?",
         "reference": f"The peak full month was {peak['YearMonth']} with "
                      f"{int(peak['total_sales'])} in sales.",
         "expected_stat": "time"},
    ]


def _deterministic(results: list[dict], k: int = 2) -> dict:
    n = len(results)
    # Hit-rate@k (recall@k): did the expected stat block appear among the top-k retrieved?
    # NOT precision (with 1 relevant block and k>1, precision can't reach 1.0) — named
    # honestly so a 1.0 isn't mistaken for perfect precision.
    retr_hits = sum(1 for r in results if r["case"]["expected_stat"] in r["out"]["stats_used"])
    grounded = 0
    for r in results:
        # Allowed = computed stats + retrieved source BODIES (mirrors the rag gate).
        allowed = "\n\n".join(d.page_content for d in r["stat_docs"] + r["source_docs"])
        ok, _ = grounding.is_numerically_grounded(r["out"]["answer"], allowed)
        grounded += int(ok)
    return {
        "retrieval_hit_rate_at_k": round(retr_hits / n, 3) if n else 1.0,
        "k": k,
        # NOTE: offline the answer IS the stat-block text, so this is grounded-by-
        # construction (~1.0). The gate's real teeth are negative-tested on the LLM path
        # in test_rag_llm.py (a fabricated figure is rejected); here it's a sanity floor.
        "numeric_grounding_rate": round(grounded / n, 3) if n else 1.0,
        "n": n,
    }


def _parse_qa_grade(raw: str) -> bool:
    import re
    t = (raw or "").strip().lower()
    # QAEvalChain emits "GRADE: CORRECT/INCORRECT" — prefer that line. Fall back to a
    # negative-aware check so "not correct" / "incorrect" aren't mis-read as CORRECT.
    m = re.search(r"grade:\s*(correct|incorrect)", t)
    if m:
        return m.group(1) == "correct"
    if "incorrect" in t or "not correct" in t:
        return False
    return "correct" in t


def _grade_qa(results: list[dict]) -> dict:
    try:
        try:
            from langchain_classic.evaluation.qa import QAEvalChain
        except ImportError:
            from langchain.evaluation.qa import QAEvalChain
        grader = QAEvalChain.from_llm(llm.chat_model(256))
        examples = [{"query": r["case"]["question"], "answer": r["case"]["reference"]} for r in results]
        preds = [{"result": r["out"]["answer"]} for r in results]
        graded = grader.evaluate(examples, preds, question_key="query",
                                 answer_key="answer", prediction_key="result")
        verdicts = [_parse_qa_grade(g.get("results") or g.get("text") or "") for g in graded]
        return {"correct_rate": round(sum(verdicts) / len(verdicts), 3), "n": len(verdicts)}
    except Exception as e:
        return {"correct_rate": None, "n": len(results), "error": str(e)[:120]}


def evaluate() -> dict:
    cases = build_eval_set()
    results = []
    for c in cases:
        stat_docs, source_docs = rag.retrieve(c["question"])
        out = rag.answer(c["question"])
        results.append({"case": c, "out": out, "stat_docs": stat_docs, "source_docs": source_docs})
    report = {"version": EVAL_VERSION, "use_llm": config.USE_LLM, "cases": len(cases),
              "deterministic": _deterministic(results)}
    if config.USE_LLM:
        report["qa_correctness"] = _grade_qa(results)
    return report


def _print_report(r: dict) -> None:
    d = r["deterministic"]
    print(f"\n=== InsightForge Eval ({r['version']}) ===")
    print(f"cases: {r['cases']}   use_llm: {r['use_llm']}\n")
    print("-- deterministic gates --")
    print(f"  retrieval_hit_rate@{d['k']}   : {d['retrieval_hit_rate_at_k']}  (n={d['n']})")
    print(f"  numeric_grounding_rate  : {d['numeric_grounding_rate']}  (n={d['n']}, "
          "grounded-by-construction offline; gate negative-tested in test_rag_llm.py)")
    if r.get("qa_correctness"):
        q = r["qa_correctness"]
        print("\n-- LLM-graded (QAEvalChain) --")
        print(f"  qa_correctness          : {q.get('correct_rate')}  (n={q['n']})"
              + (f"  error={q['error']}" if q.get("error") else ""))
    print()


def main() -> None:
    _print_report(evaluate())


if __name__ == "__main__":
    main()
