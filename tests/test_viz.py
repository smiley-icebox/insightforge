"""Visualizations render headless (Agg backend) — smoke test the four required charts."""

import viz


def test_all_required_charts_render():
    assert set(viz.CHARTS) == {
        "Sales trend over time", "Product performance",
        "Regional analysis", "Customer demographics",
    }
    for name, fn in viz.CHARTS.items():
        fig = fn()
        assert fig.axes, f"{name} produced no axes"
