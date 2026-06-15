"""Landing key-insights — deterministic, grounded summary lines."""

import insights


def test_key_insights_are_present_and_grounded():
    lines = insights.key_insights()
    assert len(lines) >= 4
    blob = " ".join(lines)
    assert "Widget A" in blob                 # the real top product
    assert "1,383,220" in blob                # the real total, formatted
    assert "%" in blob                        # shares surfaced


def test_peak_month_is_pinned_to_the_data():
    # S6: pin the user-facing peak figure (peak is a max, so a partial month never wins).
    blob = " ".join(insights.key_insights())
    assert "2028-04" in blob and "20,387" in blob


def test_satisfaction_tension_not_surfaced_when_margin_is_noise():
    # The satisfaction spread is ~0.16 on a 1-5 scale (below the 0.25 margin) — statistical
    # noise, so it must NOT be dressed up as an opportunity.
    blob = " ".join(insights.key_insights())
    assert "Worth a look" not in blob
