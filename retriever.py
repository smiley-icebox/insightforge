"""The two custom retrievers — the heart of the dual-RAG design.

  StatsRetriever  — over the deterministic statistic blocks (analytics.stat_blocks). This
                    is the brief's "custom retriever to extract relevant statistics": a
                    question is matched to the computed stat block(s) that answer it, so
                    the LLM is grounded in real numbers it didn't compute.

  DocRetriever    — over chunks of the BI literature PDFs (TF-IDF + FAISS). This powers
                    the qualitative, CITED recommendations layer — going beyond the brief.

Both are LangChain BaseRetrievers (so they drop into LCEL chains) and both are fully
offline + deterministic (TF-IDF, no embeddings API, no torch). The FAISS index over the
documents is built once from the source PDFs and cached on disk.
"""

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import analytics
import config

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) >= 3}


# --- quantitative: retrieve the relevant computed statistics -----------------
class StatsRetriever(BaseRetriever):
    """Match a question to the computed stat block(s) that answer it (TF-IDF cosine over
    title+tags+text). Small corpus (a handful of blocks), so plain cosine — no FAISS
    needed. Returns the blocks as Documents; the chain grounds the LLM in them."""

    k: int = 2

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        blocks = analytics.stat_blocks()
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [f"{b['title']} {b['tags']} {b['text']}" for b in blocks]
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(corpus)
        q = vec.transform([query or ""])
        scores = (mat @ q.T).toarray().ravel()
        order = scores.argsort()[::-1][: self.k]
        return [Document(page_content=blocks[i]["text"],
                         metadata={"title": blocks[i]["title"], "id": blocks[i]["id"],
                                   "kind": "stat", "score": float(scores[i])})
                for i in order if scores[i] > 0]


# --- qualitative: retrieve cited passages from the BI source PDFs ------------
def _pdf_chunks() -> list[dict]:
    """Extract + chunk the source PDFs into ~paragraph passages with source metadata."""
    from pypdf import PdfReader
    chunks = []
    if not os.path.isdir(config.SOURCES_DIR):
        return chunks
    for fname in sorted(os.listdir(config.SOURCES_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(config.SOURCES_DIR, fname)
        try:
            reader = PdfReader(path)
        except Exception:
            continue
        for pageno, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            # split into reasonably sized passages on blank lines / sentence runs
            for para in re.split(r"\n\s*\n", text):
                para = re.sub(r"\s+", " ", para).strip()
                if len(para) >= 120:                       # skip headers/page-furniture
                    chunks.append({"text": para[:1200], "source": fname[:-4], "page": pageno})
    return chunks


class DocRetriever(BaseRetriever):
    """TF-IDF + FAISS retrieval over the BI literature, for cited recommendations. Built
    lazily from the source PDFs and cached on the instance (rebuilt from source, never
    drifts). Returns Documents carrying source + page for citation."""

    k: int = 3

    def _ensure(self):
        if getattr(self, "_built", False):
            return
        import faiss
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        chunks = _pdf_chunks()
        object.__setattr__(self, "_chunks", chunks)
        if not chunks:
            object.__setattr__(self, "_built", True)
            return
        vec = TfidfVectorizer(stop_words="english", max_features=4096)
        mat = vec.fit_transform([c["text"] for c in chunks]).astype("float32").toarray()
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        object.__setattr__(self, "_vec", vec)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_np", np)
        object.__setattr__(self, "_faiss", faiss)
        object.__setattr__(self, "_built", True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        self._ensure()
        chunks = getattr(self, "_chunks", [])
        if not chunks or not getattr(self, "_vec", None):
            return []
        q = self._vec.transform([query or ""]).astype("float32").toarray()
        if q.sum() == 0:
            return []
        self._faiss.normalize_L2(q)
        scores, idx = self._index.search(q, min(self.k, len(chunks)))
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i >= 0 and score > 0.03:
                c = chunks[i]
                out.append(Document(page_content=c["text"],
                                    metadata={"source": c["source"], "page": c["page"],
                                              "kind": "source", "score": float(score)}))
        return out
