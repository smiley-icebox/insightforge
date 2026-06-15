"""Auto-surfaced key insights — a deterministic, grounded summary for the landing screen.

Computed straight from analytics (no LLM), so the app opens with the few things actually
worth noticing in the data — top product/region with shares, the peak month, and the
satisfaction-vs-sales tension — every figure traceable, consistent with the rest of the
system. This is the "access insights" part of the brief, made the first thing you see.
"""

import analytics


def key_insights() -> list[str]:
    ov = analytics.overview()
    prod = analytics.sales_by_product()
    region = analytics.sales_by_region()
    months = analytics.sales_by_period(freq="month")
    peak = months.loc[months["total_sales"].idxmax()]

    top_p = prod.iloc[0]
    top_r = region.iloc[0]
    best_sat = prod.sort_values("mean_satisfaction", ascending=False).iloc[0]

    out = [
        f"**Top product:** {top_p['Product']} — {top_p['share_pct']}% of sales "
        f"({int(top_p['total_sales']):,}), ahead of the lowest by "
        f"{int(prod.iloc[-1]['gap_to_leader']):,}.",
        f"**Top region:** {top_r['Region']} — {top_r['share_pct']}% of sales "
        f"({int(top_r['total_sales']):,}).",
        f"**Peak month:** {peak['YearMonth']} at {int(peak['total_sales']):,} in sales.",
        f"**Overall:** {ov['total_sales']:,} total sales across {ov['rows']:,} records "
        f"(mean {ov['mean_sales']}, median {ov['median_sales']}).",
    ]
    # The "best-seller isn't best-rated" angle is only worth surfacing if the satisfaction
    # gap is MEANINGFUL — on this data the spread is ~0.16 on a 1-5 scale (statistical
    # noise), so presenting it as an opportunity would be dressing noise as a finding. Gate
    # on a real margin; stay quiet otherwise.
    SATISFACTION_MARGIN = 0.25
    top_p_sat = prod[prod["Product"] == top_p["Product"]]["mean_satisfaction"].iloc[0]
    if (best_sat["Product"] != top_p["Product"]
            and best_sat["mean_satisfaction"] - top_p_sat >= SATISFACTION_MARGIN):
        out.append(
            f"**Worth a look:** {best_sat['Product']} has the highest satisfaction "
            f"({best_sat['mean_satisfaction']}) but isn't the top seller — a possible "
            f"under-marketed opportunity.")
    return out
