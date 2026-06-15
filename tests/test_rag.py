"""The RAG pipeline, offline (deterministic path): grounded-by-construction answers,
correct provenance, and the no-grounding refusal."""

import glob
import os

import pytest

import config
import grounding
import rag

_HAS_PDFS = bool(glob.glob(os.path.join(config.SOURCES_DIR, "*.pdf")))


def test_offline_answer_is_grounded_and_uses_right_stats():
    out = rag.answer("which product sells best?")
    assert out["used_llm"] is False
    assert "product" in out["stats_used"]
    assert "Widget A" in out["answer"]
    # every number in the deterministic answer traces to stats + cited sources
    stat_docs, source_docs = rag.retrieve("which product sells best?")
    allowed = "\n".join(d.page_content for d in stat_docs + source_docs)
    assert grounding.is_numerically_grounded(out["answer"], allowed)[0]


@pytest.mark.skipif(not _HAS_PDFS, reason="no source PDFs present (not redistributed publicly)")
def test_offline_answer_carries_source_citations_when_doc_rag_on():
    if not config.USE_DOC_RAG:
        return
    out = rag.answer("what should we do to improve sales?")
    assert out["citations"]                      # cited recommendations attached in code
    assert all(c["source"] and c["page"] for c in out["citations"])


def test_off_topic_question_refuses_rather_than_guessing():
    out = rag.answer("zzqx wxyz nonsense qqqq")
    assert out["answer"] == config.NO_GROUNDING_MESSAGE
    assert out["grounded"] is False
