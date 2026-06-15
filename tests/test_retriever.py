"""The two custom retrievers: stats blocks (quantitative) and cited PDF passages."""

import glob
import os

import pytest

import config
from retriever import DocRetriever, StatsRetriever

_HAS_PDFS = bool(glob.glob(os.path.join(config.SOURCES_DIR, "*.pdf")))


def test_stats_retriever_maps_questions_to_right_blocks():
    sr = StatsRetriever(k=1)
    assert sr.invoke("which product is best?")[0].metadata["id"] == "product"
    assert sr.invoke("sales by region")[0].metadata["id"] == "region"
    assert sr.invoke("age and gender of customers")[0].metadata["id"] == "demographics"


def test_stats_retriever_returns_no_block_for_gibberish():
    assert StatsRetriever(k=2).invoke("zzqx wxyz qqqq") == []


@pytest.mark.skipif(not _HAS_PDFS, reason="no source PDFs present (not redistributed publicly)")
def test_doc_retriever_returns_cited_passages():
    docs = DocRetriever(k=3).invoke("business intelligence best practices for dashboards")
    assert docs, "expected at least one source passage"
    d = docs[0]
    assert d.metadata["kind"] == "source"
    assert d.metadata.get("source") and d.metadata.get("page")  # citation present
