# Qualitative source corpus (BYO PDFs)

The **qualitative RAG** layer (`DocRetriever`) grounds recommendations in business-
intelligence literature. The capstone shipped a few third-party academic PDFs for this —
they are **copyrighted and not redistributed** in this public repo.

To enable the qualitative layer, drop any BI-related PDFs into this folder, e.g.:

```
data/sources/
  some-bi-whitepaper.pdf
  your-industry-report.pdf
```

They're indexed (TF-IDF + FAISS) at first use and cited by filename + page. **Nothing else
depends on them** — the quantitative assistant (the core, "the LLM never does math") works
fully without any PDFs here; the recommendation layer simply stays quiet until you add some.
Set `USE_DOC_RAG=0` to disable the layer entirely.
