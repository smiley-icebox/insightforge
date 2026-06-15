"""Numeric grounding — the enforcement behind "the LLM never invents a number."

After the LLM phrases an answer, every SIGNIFICANT number in it must trace to a figure in
the retrieved statistics (within a small rounding tolerance). If one doesn't, the answer is
rejected and the deterministic rendering ships instead.

Three sharp edges, all deliberate:
  - "Significant" = has a decimal, OR is >= 10, OR is immediately followed by % / x /
    percent. A BARE small integer (counts like "4 products", the 1-5 satisfaction scale)
    is structural and skipped — but "9%" or "5x" is a claim and is NOT skipped (that was a
    real bypass: single-digit percentages are the most common fabricated BI figure).
  - The number regex matches a whole numeric run, so a multi-dot token like a DOI/section
    number ("2020.03.24010") fails to parse and is ignored rather than being split into a
    spurious "24010" that a fabricated figure could match.
  - Tolerance is tight (0.1% relative, or 0.5 absolute) so a fabricated value can't match a
    DIFFERENT real figure in a dense cluster of means; exact quotes (the prompt demands
    them) match exactly, and light rounding ("553" for 553.29) is absorbed by tol_abs.
"""

import re

# A full numeric run. Notes on the sharp edges:
#  - (?<!\d) so a '-' is a sign only when NOT preceded by a digit — "2020-03" yields
#    2020 and 3, not a spurious -3 (dates), while "-82.6%" keeps its sign.
#  - the \.\d+ alternative parses a leading-dot decimal (".5" -> 0.5, not 5).
#  - a multi-dot run ("2020.03.24010") matches whole and fails float() below, so it's
#    dropped rather than split into a spurious figure.
_NUM = r"(?<!\d)-?(?:\d[\d,.]*\d|\.\d+|\d)"
_NUM_RE = re.compile(_NUM)
# A number immediately followed by a percent/multiplier marker -> always a claim.
_CLAIM_SUFFIX_RE = re.compile(rf"({_NUM})\s*(%|percent|x\b|×)", re.IGNORECASE)


def numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        cleaned = m.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue   # multi-dot / malformed run -> not a quantity
    return out


def _claim_values(text: str) -> set[float]:
    vals = set()
    for raw, _suffix in _CLAIM_SUFFIX_RE.findall(text or ""):
        try:
            vals.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return vals


def is_numerically_grounded(answer: str, allowed_text: str,
                            tol_rel: float = 0.001, tol_abs: float = 0.01):
    """Return (ok, offending_number). A number is grounded if it's within tol of a figure
    in `allowed_text`. Bare small integers (<10) are structural and skipped — UNLESS they
    carry a % / x suffix, which makes them a claim that must be grounded.

    tol_abs is 0.01 (figures are rendered to 2dp): tight enough that a fabricated value
    can't match a DIFFERENT real figure in a dense cluster — e.g. satisfaction means span
    only ~0.16 on the 1-5 scale, so a ±0.5 floor would have let any of them match any
    other. tol_rel still absorbs light rounding of large figures."""
    allowed = numbers(allowed_text)
    claims = _claim_values(answer)
    for n in numbers(answer):
        structural = abs(n) < 10 and float(n).is_integer() and n not in claims
        if structural:
            continue
        if not any(abs(n - a) <= max(tol_abs, abs(a) * tol_rel) for a in allowed):
            return False, n
    return True, None
