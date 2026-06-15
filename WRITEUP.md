# InsightForge — Project Writeup

Enabling AI-Powered Business Intelligence for Organizations
Advanced GenAI Capstone · STAR format

---

## Situation

Organizations accumulate sales and customer data but struggle to turn it into answers —
especially smaller teams without a BI analyst. LLMs make natural-language data Q&A
tempting, but the naive version is dangerous: ask a model about a spreadsheet and it will
confidently report numbers it made up. For a *business intelligence* tool, a wrong figure
is worse than no figure.

## Task

Build InsightForge: a BI assistant using LangChain, RAG, and LLMs that analyzes business
data, generates insights and recommendations, and visualizes them — with QAEvalChain
evaluation and a Streamlit UI. Then take it past the brief toward something trustworthy
enough to actually use.

## Action

### The central design decision: the LLM never does math

Every number is computed deterministically by **pandas** (`analytics.py`) — sales by time
period, product and regional analysis (with shares and gaps), customer segmentation by
demographics, and statistical measures (mean, median, std, quartiles). A **custom
retriever** (`retriever.py`) maps a question to the relevant computed stat blocks, and the
LLM is asked only to **phrase** them using the exact figures. A post-generation
**numeric-grounding gate** (`grounding.py`) then verifies that every significant number in
the answer traces to a retrieved statistic (within a small rounding tolerance); if one
doesn't, the deterministic rendering ships instead. The model is fenced out of arithmetic.

To keep rich comparative answers *grounded* rather than falling back, the stats engine
**precomputes the derived figures** users ask for — each product/region's share of total
and gap to the leader — so the LLM quotes them instead of subtracting. This was a real
finding during the build: the first live run had the model computing "Widget A outsells
Widget D by 48,381," which the gate correctly rejected; precomputing the gap made the same
answer ship, grounded.

### Two grounded RAG layers (beyond the brief)

- **Quantitative RAG** — the custom statistics retriever above (the brief's "custom
  retriever to extract relevant statistics").
- **Qualitative RAG** — a second retriever (TF-IDF + FAISS) over the provided BI
  literature PDFs, so recommendations are **grounded in and cited from real sources**
  (source + page), not invented. This makes the assistant useful for the "so what?", not
  just the "what."

### The rest of the brief

- **Chained prompts** (`rag.py`): a structured system prompt + retrieved stats + cited
  sources + conversation context, engineered to force exact figures and refuse when the
  data doesn't contain the answer.
- **Memory** (`memory.py`): a bounded conversation buffer so follow-ups resolve against
  earlier turns.
- **QAEvalChain** (`evaluation.py`): LangChain's QA grader over a versioned eval set whose
  **gold answers are computed from the data** (so they can't drift), alongside
  deterministic gates (retrieval hit-rate@k, numeric grounding).
- **Visualizations** (`viz.py`): the four required charts, all driven by the same
  analytics layer so a chart can never disagree with an answer.
- **Streamlit UI** (`app.py`): a grounded chat (each answer carries an inline "how I
  answered this" provenance panel) plus a visual dashboard.

## Result

- All required capabilities work end-to-end. On the versioned eval set: **retrieval
  hit-rate@k 1.0, numeric grounding 1.0**, and LangChain **QAEvalChain correctness 1.0**.
- **28 automated tests pass with no API call** — analytics figures, the grounding gate,
  both retrievers, the RAG path (incl. a stubbed-LLM test proving fabricated numbers fall
  back), memory, eval, and the charts are all verifiable offline.
- The assistant answers comparative questions with exact, grounded figures (shares, gaps,
  the insight that the top-selling product isn't the highest-satisfaction one) and cites BI
  sources for recommendations — and **refuses** rather than guessing when the data can't
  answer.

## Engineering decisions worth calling out

| Decision | Why |
|----------|-----|
| LLM never computes a number | A BI tool's core failure mode is a confident wrong figure; pandas owns the math. |
| Numeric-grounding gate on the final answer | Makes "no fabricated numbers" an enforced invariant, not a hope. |
| Precompute shares + gaps | Lets comparative answers stay grounded instead of falling back. |
| Citations attached in code | The model can't invent a source; provenance is verifiable. |
| Gold eval answers computed from data | The references can't drift from the dataset. |
| Charts driven by the same analytics layer | A visualization can never contradict an answer. |

## What production would still demand (documented, not built)

- **Semantic embeddings + a managed vector store** behind both retrievers (TF-IDF today —
  deterministic and CI-friendly, but weaker on paraphrase).
- A **warehouse/connector** instead of a single in-process CSV, with per-tenant access.
- **Persisted, summarized memory** per user instead of an in-process buffer.
- A larger eval set and an independent judge for the qualitative recommendations.
