"""Landing key-insights — deterministic, grounded summary lines."""

import insights


def test_key_insights_are_present_and_grounded():
    lines = insights.key_insights()
    assert len(lines) >= 4
    blob = " ".join(lines)
    assert "Widget A" in blob                 # the real top product
    assert "1,383,220" in blob                # the real total, formatted
    assert "%" in blob                        # shares surfaced


def test_satisfaction_tension_insight_surfaces():
    # Widget D is the highest-satisfaction product but not the top seller — the insight
    # generator should flag that opportunity.
    blob = " ".join(insights.key_insights())
    assert "Worth a look" in blob and "Widget D" in blob
