"""Streamlit UI — chat with the BI assistant + a visual dashboard.

Two tabs: "Ask" is a grounded chat (each answer carries an inline "how I answered this"
panel — the stats retrieved, the sources cited, and whether it shipped grounded), and
"Dashboard" shows the four required charts plus headline metrics. Memory persists across
turns within a session. Everything routes through rag.answer, so the numeric-grounding
guarantee holds in the UI exactly as in the tests.

Run: streamlit run app.py   (offline if USE_LLM=0; live with an API key)
"""

import streamlit as st

import analytics
import config
import evaluation
import rag
import viz
from memory import ConversationMemory

st.set_page_config(page_title="InsightForge — BI Assistant", page_icon="📊", layout="wide")

EXAMPLES = [
    "Which product sells best, and how does it compare to the others?",
    "Which region has the highest sales?",
    "Summarize the monthly sales trend.",
    "Break down customers by age and gender.",
    "What's the median and standard deviation of sales?",
    "What does the data suggest we should focus on?",
]


@st.cache_resource
def _overview():
    return analytics.overview()


def _memory() -> ConversationMemory:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
        st.session_state.history = []
    return st.session_state.memory


def sidebar():
    ov = _overview()
    st.sidebar.title("📊 InsightForge")
    st.sidebar.caption("A grounded BI assistant — every figure traces to the data or a "
                       "cited source; the LLM never invents a number.")
    st.sidebar.markdown("### Dataset")
    st.sidebar.caption(f"{ov['rows']} records · {ov['date_min']} → {ov['date_max']}")
    st.sidebar.metric("Total sales", f"{ov['total_sales']:,}")
    st.sidebar.metric("Mean sale", ov["mean_sales"])
    st.sidebar.metric("Mean satisfaction", f"{ov['mean_satisfaction']} / 5")
    with st.sidebar.expander("🔬 Under the hood"):
        st.caption(f"LLM: {'on' if config.USE_LLM else 'off (deterministic)'} · "
                   f"doc-RAG: {'on' if config.USE_DOC_RAG else 'off'}")
        if st.button("Run evaluation"):
            with st.spinner("Running eval set…"):
                st.session_state["eval_report"] = evaluation.evaluate()
        rep = st.session_state.get("eval_report")   # persists across reruns
        if rep:
            d = rep["deterministic"]
            st.metric(f"Retrieval hit-rate@{d['k']}", d["retrieval_hit_rate_at_k"])
            st.metric("Numeric grounding", d["numeric_grounding_rate"])
            if rep.get("qa_correctness"):
                st.metric("QAEvalChain correctness", rep["qa_correctness"].get("correct_rate"))


def _render_turn(turn):
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        cites = turn.get("citations") or []
        if cites:
            st.caption("Sources: " + " · ".join(f"{c['source']} (p.{c['page']})" for c in cites))
        with st.expander("🔎 How I answered this"):
            st.markdown(f"**Statistics retrieved:** {', '.join(turn['stats_used']) or '—'}")
            st.markdown(f"**Sources cited:** {', '.join(turn['sources_used']) or '—'}")
            st.markdown(f"**Numerically grounded:** {'yes' if turn['grounded'] else 'fell back to deterministic'}")


def tab_ask():
    st.subheader("Ask InsightForge")
    mem = _memory()
    cols = st.columns(3)
    pending = None
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, use_container_width=True, key=f"ex_{i}"):
            pending = ex
    for turn in st.session_state.get("history", []):
        _render_turn(turn)
    typed = st.chat_input("Ask about sales, products, regions, customers…")
    q = typed or pending
    if q:
        with st.spinner("Analyzing…"):
            out = rag.answer(q, memory=mem)
        st.session_state.history.append({"q": q, **out})
        st.rerun()


def tab_dashboard():
    st.subheader("Dashboard")
    ov = _overview()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales", f"{ov['total_sales']:,}")
    c2.metric("Median sale", ov["median_sales"])
    c3.metric("Std dev", ov["std_sales"])
    c4.metric("Mean age", ov["mean_age"])
    left, right = st.columns(2)
    with left:
        st.pyplot(viz.sales_trend())
        st.pyplot(viz.regional_analysis())
    with right:
        st.pyplot(viz.product_performance())
        st.pyplot(viz.customer_demographics())


def main():
    sidebar()
    st.title("📊 InsightForge — Business Intelligence Assistant")
    ask, dash = st.tabs(["💬 Ask", "📊 Dashboard"])
    with ask:
        tab_ask()
    with dash:
        tab_dashboard()


if __name__ == "__main__":
    main()
