"""The numeric-grounding gate — the enforcement behind 'the LLM never invents a number'."""

from grounding import is_numerically_grounded, numbers


def test_grounded_exact_and_rounded():
    allowed = "Total sales: 1383220. Mean sale: 553.29, median: 552.5."
    assert is_numerically_grounded("Total sales is 1,383,220.", allowed)[0]   # comma form
    assert is_numerically_grounded("The mean is about 553.", allowed)[0]       # rounded within tol


def test_invented_number_is_rejected():
    allowed = "Mean sale: 553.29, median: 552.5."
    ok, bad = is_numerically_grounded("Total sales reached 999999.", allowed)
    assert ok is False and bad == 999999.0


def test_small_structural_integers_are_skipped():
    # counts / scale points aren't claims about the data
    assert is_numerically_grounded("There are 4 products on a 1 to 5 scale.", "")[0]


def test_single_digit_percentage_and_multiplier_are_NOT_skipped():
    # the highest-frequency fabricated BI figure — must be caught even though it's < 10
    allowed = "Total sales: 1383220. Mean sale: 553.29."
    assert is_numerically_grounded("Sales grew 9% year over year.", allowed)[0] is False
    assert is_numerically_grounded("North outsells South by 5x.", allowed)[0] is False


def test_multi_dot_doi_or_section_number_does_not_pool_a_spurious_figure():
    # a DOI/section like 2020.03.24010 must not seed a fake "24010" into the allowed pool
    allowed = "Mean sale: 553.29.\n[ref] doi 2020.03.24010 section 4.2.1"
    assert is_numerically_grounded("Projected sales reached 24,010 units.", allowed)[0] is False


def test_number_from_a_cited_source_counts_as_grounded():
    allowed = "Mean sale: 553.29.\n[BI approaches p.6] Gartner Group, 2007 ranked BI #1."
    assert is_numerically_grounded("Per Gartner (2007), BI was top-ranked.", allowed)[0]


def test_numbers_parser_handles_commas_and_decimals():
    assert numbers("1,383,220 and 552.5 and 27.13%") == [1383220.0, 552.5, 27.13]
