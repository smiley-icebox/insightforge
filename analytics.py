"""Deterministic statistics engine — pandas computes every number, full stop.

This is the spine of InsightForge: the LLM is NEVER asked to do arithmetic. Each public
function returns a structured result, and `stat_blocks()` renders the whole set into
titled, tagged text blocks. The custom retriever (retriever.py) indexes those blocks and
hands the relevant ones to the LLM, which only PHRASES them — so a number can appear in an
answer only if pandas computed it here first. That is what makes the assistant trustworthy
on figures (the #1 failure mode of "ask the LLM about your spreadsheet" tools).

Coverage maps 1:1 to the brief's "advanced data summary": sales by time period, product
and regional analysis, customer segmentation by demographics, and statistical measures
(median, standard deviation, quartiles).
"""

import pandas as pd

import data as data_module


def _df() -> pd.DataFrame:
    return data_module.load()


def _round(x) -> float:
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return float("nan")


# --- structured analyses (the numbers) ---------------------------------------
def overview(df: pd.DataFrame | None = None) -> dict:
    df = _df() if df is None else df
    s = df["Sales"]
    return {
        "rows": int(len(df)),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "total_sales": int(s.sum()),
        "mean_sales": _round(s.mean()),
        "median_sales": _round(s.median()),
        "std_sales": _round(s.std()),
        "min_sales": int(s.min()),
        "max_sales": int(s.max()),
        "mean_satisfaction": _round(df["Customer_Satisfaction"].mean()),
        "mean_age": _round(df["Customer_Age"].mean()),
    }


def sales_by_period(df: pd.DataFrame | None = None, freq: str = "month") -> pd.DataFrame:
    df = _df() if df is None else df
    key = {"month": "YearMonth", "year": "Year"}.get(freq, "YearMonth")
    g = df.groupby(key)["Sales"].agg(["sum", "mean", "count"]).reset_index()
    g.columns = [key, "total_sales", "mean_sales", "n"]
    g["mean_sales"] = g["mean_sales"].round(2)   # match the 2dp the other analytics use
    return g


def sales_by_product(df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _grouped(df, "Product")


def sales_by_region(df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _grouped(df, "Region")


def _grouped(df, col) -> pd.DataFrame:
    df = _df() if df is None else df
    g = df.groupby(col).agg(
        total_sales=("Sales", "sum"),
        mean_sales=("Sales", "mean"),
        median_sales=("Sales", "median"),
        std_sales=("Sales", "std"),
        n=("Sales", "count"),
        mean_satisfaction=("Customer_Satisfaction", "mean"),
    ).reset_index().sort_values("total_sales", ascending=False)
    # Derived comparison figures PRECOMPUTED here so the LLM can quote them rather than
    # do arithmetic (share of total, gap to the leader) — keeps comparative answers
    # grounded under the numeric gate.
    grand = g["total_sales"].sum()
    leader = g["total_sales"].iloc[0]
    g["share_pct"] = (g["total_sales"] / grand * 100).round(2)
    g["gap_to_leader"] = (leader - g["total_sales"]).astype(int)
    for c in ("mean_sales", "median_sales", "std_sales", "mean_satisfaction"):
        g[c] = g[c].round(2)
    return g


def customer_segmentation(df: pd.DataFrame | None = None) -> dict:
    df = _df() if df is None else df
    by_age = df.groupby("AgeBand").agg(
        n=("Sales", "count"), mean_sales=("Sales", "mean"),
        mean_satisfaction=("Customer_Satisfaction", "mean")).round(2).reset_index()
    by_gender = df.groupby("Customer_Gender").agg(
        n=("Sales", "count"), mean_sales=("Sales", "mean"),
        mean_satisfaction=("Customer_Satisfaction", "mean")).round(2).reset_index()
    return {"by_age": by_age, "by_gender": by_gender}


def statistical_measures(df: pd.DataFrame | None = None) -> dict:
    df = _df() if df is None else df
    out = {}
    for col in ("Sales", "Customer_Age", "Customer_Satisfaction"):
        s = df[col]
        out[col] = {
            "mean": _round(s.mean()), "median": _round(s.median()),
            "std": _round(s.std()), "min": _round(s.min()), "max": _round(s.max()),
            "q1": _round(s.quantile(0.25)), "q3": _round(s.quantile(0.75)),
        }
    return out


# --- rendering: structured numbers -> retrievable text blocks ----------------
def _tbl(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def stat_blocks(df: pd.DataFrame | None = None) -> list[dict]:
    """The computed statistics rendered as titled, tagged text blocks. The retriever
    indexes these; the LLM is grounded in the ones it retrieves. Each block's `text` is
    pure computed fact — no prose, no interpretation."""
    df = _df() if df is None else df
    ov = overview(df)
    months = sales_by_period(df, "month")
    years = sales_by_period(df, "year")
    # Guard against partial boundary buckets: a half-recorded final month/year would make
    # peak/trough/growth a truncation artifact (e.g. a fake -82% "collapse"), the exact
    # mistake the grounding thesis exists to prevent. Use only "full" months (record count
    # >= half the median) for trend endpoints, and flag any partial year in the table.
    med_n = months["n"].median()
    full = months[months["n"] >= 0.5 * med_n]
    peak = full.loc[full["total_sales"].idxmax()]
    trough = full.loc[full["total_sales"].idxmin()]
    first, last = full.iloc[0], full.iloc[-1]   # first/last FULL month, chronological
    growth = _round((last["total_sales"] - first["total_sales"]) / first["total_sales"] * 100)
    partial_years = years[years["n"] < 360]["Year"].tolist()
    partial_note = (f" (note: {', '.join(str(y) for y in partial_years)} cover partial "
                    f"years — fewer recorded days)") if partial_years else ""
    seg = customer_segmentation(df)
    stats = statistical_measures(df)
    prod, region = sales_by_product(df), sales_by_region(df)

    blocks = [
        {"id": "overview", "title": "Overall sales summary",
         "tags": "overview total sales summary average mean median revenue how much overall",
         "text": (f"Rows: {ov['rows']}. Date range: {ov['date_min']} to {ov['date_max']}. "
                  f"Total sales: {ov['total_sales']}. Mean sale: {ov['mean_sales']}, "
                  f"median: {ov['median_sales']}, std dev: {ov['std_sales']}, "
                  f"min: {ov['min_sales']}, max: {ov['max_sales']}. "
                  f"Mean customer satisfaction: {ov['mean_satisfaction']} (1-5). "
                  f"Mean customer age: {ov['mean_age']}.")},
        {"id": "time", "title": "Sales by time period (trend)",
         "tags": "time period month monthly yearly trend over time growth seasonal peak best worst month year when",
         # Concise on purpose: yearly totals + peak/trough + a PRECOMPUTED growth figure,
         # so a phrased trend summary stays grounded (and the fallback stays readable). The
         # full monthly series drives the chart, not this text.
         "text": (f"Yearly total sales:\n{_tbl(years)}{partial_note}\n"
                  f"Among full months — peak: {peak['YearMonth']} ({int(peak['total_sales'])}); "
                  f"lowest: {trough['YearMonth']} ({int(trough['total_sales'])}). "
                  f"First full month {first['YearMonth']}: {int(first['total_sales'])}; "
                  f"last full month {last['YearMonth']}: {int(last['total_sales'])} "
                  f"({growth}% change, first-to-last full month).")},
        {"id": "product", "title": "Sales by product",
         "tags": "product widget which product best worst top selling performance compare products",
         "text": (f"Sales by product (sorted by total):\n{_tbl(prod)}\n"
                  f"Best-selling product: {prod.iloc[0]['Product']} "
                  f"({int(prod.iloc[0]['total_sales'])}). "
                  f"Lowest: {prod.iloc[-1]['Product']} ({int(prod.iloc[-1]['total_sales'])}).")},
        {"id": "region", "title": "Sales by region",
         "tags": "region regional north south east west area geography where location best region",
         "text": (f"Sales by region (sorted by total):\n{_tbl(region)}\n"
                  f"Top region: {region.iloc[0]['Region']} "
                  f"({int(region.iloc[0]['total_sales'])}). "
                  f"Lowest: {region.iloc[-1]['Region']} ({int(region.iloc[-1]['total_sales'])}).")},
        {"id": "demographics", "title": "Customer segmentation by demographics",
         "tags": "customer demographics age gender segmentation segment who buyers satisfaction by age by gender",
         "text": (f"By age band:\n{_tbl(seg['by_age'])}\n\nBy gender:\n{_tbl(seg['by_gender'])}")},
        {"id": "stats", "title": "Statistical measures",
         "tags": "statistics statistical measures median standard deviation std variance quartile spread distribution",
         "text": ("Mean / median / std / quartiles per metric:\n"
                  + "\n".join(f"{k}: {v}" for k, v in stats.items()))},
    ]
    return blocks
