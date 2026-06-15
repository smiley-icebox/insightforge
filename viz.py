"""Visualizations — the four charts the brief requires, each driven by the deterministic
analytics layer (so a chart can never disagree with a number in an answer).

  sales_trend            — sales trends over time (monthly line)
  product_performance    — product performance comparison (bar)
  regional_analysis      — regional analysis (bar)
  customer_demographics  — demographics & segmentation (age bands + gender)

Each returns a Matplotlib Figure for Streamlit's st.pyplot. Figures are built via
matplotlib.figure.Figure() directly (not pyplot), so nothing is added to pyplot's global
registry — no figure leaks across Streamlit reruns, and it imports cleanly headless.
"""

from matplotlib.figure import Figure   # construct figures OFF the global pyplot registry

import analytics

_PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#db2777"]


def _fig(w=7, h=3.5):
    # Figure() (not plt.subplots) means the figure is never added to pyplot's global
    # registry — so Streamlit reruns can't leak figures / hit matplotlib's memory warning.
    return Figure(figsize=(w, h))


def sales_trend():
    m = analytics.sales_by_period(freq="month")
    fig = _fig(8, 3.5)
    ax = fig.subplots()
    ax.plot(m["YearMonth"], m["total_sales"], marker="o", color=_PALETTE[0], linewidth=2)
    ax.set_title("Sales trend over time (monthly total)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total sales")
    step = max(1, len(m) // 12)
    ax.set_xticks(range(0, len(m), step))
    ax.set_xticklabels(m["YearMonth"][::step], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def product_performance():
    p = analytics.sales_by_product()
    fig = _fig()
    ax = fig.subplots()
    ax.bar(p["Product"], p["total_sales"], color=_PALETTE[: len(p)])
    ax.set_title("Product performance (total sales)")
    ax.set_xlabel("Product")
    ax.set_ylabel("Total sales")
    fig.tight_layout()
    return fig


def regional_analysis():
    r = analytics.sales_by_region()
    fig = _fig()
    ax = fig.subplots()
    ax.bar(r["Region"], r["total_sales"], color=_PALETTE[: len(r)])
    ax.set_title("Regional analysis (total sales)")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total sales")
    fig.tight_layout()
    return fig


def customer_demographics():
    seg = analytics.customer_segmentation()
    age, gender = seg["by_age"], seg["by_gender"]
    fig = _fig(9, 3.5)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.bar(age["AgeBand"], age["n"], color=_PALETTE[0])
    ax1.set_title("Customers by age band")
    ax1.set_xlabel("Age band")
    ax1.set_ylabel("Customers")
    ax2.bar(gender["Customer_Gender"], gender["n"], color=_PALETTE[1])
    ax2.set_title("Customers by gender")
    ax2.set_xlabel("Gender")
    ax2.set_ylabel("Customers")
    fig.tight_layout()
    return fig


CHARTS = {
    "Sales trend over time": sales_trend,
    "Product performance": product_performance,
    "Regional analysis": regional_analysis,
    "Customer demographics": customer_demographics,
}
