"""Deterministic analytics — the numbers everything else is grounded in. Values are
regression-pinned against the committed sales_data.csv."""

import analytics


def test_overview_headline_figures():
    ov = analytics.overview()
    assert ov["rows"] == 2500
    assert ov["total_sales"] == 1383220
    assert ov["median_sales"] == 552.5
    assert 259 < ov["std_sales"] < 261        # ~260.1


def test_sales_by_product_sorted_with_shares_and_gaps():
    p = analytics.sales_by_product()
    assert list(p["Product"]) == sorted(p["Product"])  # 4 products present (A-D)
    assert p.iloc[0]["total_sales"] >= p.iloc[-1]["total_sales"]  # sorted desc by total
    assert p.iloc[0]["gap_to_leader"] == 0                        # leader has zero gap
    assert abs(p["share_pct"].sum() - 100.0) < 0.1               # shares sum to 100%
    assert p.iloc[0]["Product"] == "Widget A"


def test_sales_by_region_has_four_regions():
    r = analytics.sales_by_region()
    assert set(r["Region"]) == {"North", "South", "East", "West"}
    assert r.iloc[0]["total_sales"] >= r.iloc[-1]["total_sales"]


def test_statistical_measures_present():
    s = analytics.statistical_measures()
    for col in ("Sales", "Customer_Age", "Customer_Satisfaction"):
        assert {"mean", "median", "std", "q1", "q3"} <= set(s[col])


def test_stat_blocks_cover_required_analyses():
    ids = {b["id"] for b in analytics.stat_blocks()}
    assert {"overview", "time", "product", "region", "demographics", "stats"} == ids
