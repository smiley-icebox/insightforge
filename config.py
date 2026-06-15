"""Central configuration for InsightForge.

Everything tunable lives here: the model + resilience knobs, feature flags, data/source
paths, and the canonical wording for the no-data-no-answer guardrail. Keys come from the
environment (a local .env if present) — never hard-coded.

WHY one config module: the guarantee that distinguishes this BI assistant from a toy —
"the LLM never invents a number; every figure traces to a computed statistic" — depends
on a few strings and flags (the grounding refusal, the model, the feature gates). Keeping
them in one auditable place means a reviewer can see the controls without reading logic.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # picks up .env in this folder if present; no-op otherwise

# --- Model -------------------------------------------------------------------
LLM_MODEL = "claude-sonnet-4-6"
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 30
LLM_MAX_RETRIES = 2

# Whether the LLM phrases answers. Off => deterministic extractive answers, so the app +
# tests run with no API key. Set USE_LLM=0 to disable.
USE_LLM = os.getenv("USE_LLM", "1") not in ("0", "false", "False", "")
# Whether the qualitative document-RAG (over the BI source PDFs) is active. The
# quantitative stats RAG always runs; this adds cited recommendations on top.
USE_DOC_RAG = os.getenv("USE_DOC_RAG", "1") not in ("0", "false", "False", "")

# --- Data --------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(_HERE, "data", "sales_data.csv")
SOURCES_DIR = os.path.join(_HERE, "data", "sources")   # BI literature PDFs (doc-RAG)

# The columns we rely on (a schema contract — load() validates against these).
EXPECTED_COLUMNS = (
    "Date", "Product", "Region", "Sales",
    "Customer_Age", "Customer_Gender", "Customer_Satisfaction",
)

# --- Guardrail wording (the most important text in the system) ---------------
# Shown when a question can't be answered from the computed statistics or the cited
# sources. Better to say "I don't have that" than to let the LLM invent a figure.
NO_GROUNDING_MESSAGE = (
    "I can only answer from the sales data and the cited BI sources, and I don't have "
    "the figures to answer that confidently. Try asking about sales by product, region, "
    "time period, or customer demographics."
)

# Appended to answers that include qualitative recommendations drawn from the sources.
RECOMMENDATION_DISCLAIMER = (
    "_Recommendations are general BI guidance drawn from the cited sources, not financial "
    "advice; validate against your own context before acting._"
)

# --- Customer segmentation bands (analytics) --------------------------------
# Half-open + contiguous [lo, hi): no gaps at the boundaries (29.5 lands in 18-29) and an
# open top edge so any age >= 60 is "60+" — no fractional age or outlier falls to "unknown".
AGE_BANDS = [(18, 30, "18-29"), (30, 45, "30-44"), (45, 60, "45-59"), (60, float("inf"), "60+")]
