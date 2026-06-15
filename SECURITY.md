# Security Policy

InsightForge is an **educational / portfolio project** (an Applied GenAI capstone). It
runs over a **synthetic sales dataset** and public BI literature — there are no real
customers, credentials, or secrets in this repository.

## Reporting a vulnerability

If you find a security issue you'd still like to report, please use **GitHub's private
vulnerability reporting** (the repository's **Security** tab → *Report a vulnerability*)
rather than opening a public issue. I'll respond as time allows — this is a personal
project, not a maintained service.

## What *is* handled (the load-bearing controls)

- **The LLM never invents a number.** Every statistic is computed deterministically by
  pandas (`analytics.py`); a custom retriever feeds the relevant figures to the LLM, and a
  post-generation **numeric-grounding gate** (`grounding.py`) rejects any answer whose
  figures don't trace to the retrieved statistics — falling back to the deterministic
  rendering. Backed by tests.
- **Citations are attached in code, not authored by the LLM.** Qualitative
  recommendations are grounded in retrieved passages from the source PDFs, cited with
  source + page from the retriever's metadata (`rag.py`) — the model can't fabricate a
  source.
- **No secrets committed.** `.env` is gitignored; the Anthropic API key lives only
  locally. The app runs fully offline (`USE_LLM=0`) with no key.
- **Read-only data path.** The CSV is loaded read-only; there is no write path, no user
  accounts, and no PII beyond synthetic age/gender columns.

## Known, intentional limitations (by design, for a demo)

Documented so they aren't mistaken for oversights; a production deployment would close
each one:

- **Retrieval is TF-IDF, not semantic embeddings.** Deterministic, offline, and
  CI-friendly (no torch), which suits a demo and keeps tests reproducible. Production
  would use a semantic embedding model + a managed vector store for better recall on
  paraphrased questions.
- **Single CSV, in-process.** Fine for a demo; production would use a warehouse/connector
  and per-tenant access controls.
- **Memory is an in-process conversation buffer.** No cross-session persistence or
  summarization (documented in the README).
- **The numeric gate is a high-recall heuristic.** It catches fabricated or LLM-computed
  figures by value-matching with a small rounding tolerance; it is not a formal proof of
  correctness (the underlying figures are correct by construction from pandas). It checks a
  figure's *provenance*, not its binding to the right label — correct attribution is the
  LLM's job, constrained by the exact-quote prompt.
- **Numeric grounding is numeric-only; the qualitative source corpus is trusted.** The gate
  doesn't filter prose, and the BI source PDFs are trusted operator input (no upload path —
  `SOURCES_DIR` is fixed). An adversarial PDF could inject a qualitative instruction; out of
  scope for a curated single-operator demo.
