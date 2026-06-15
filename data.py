"""Data loading + structuring — the single source of truth for every number.

The brief says the data is pre-cleaned, so this is deliberately thin: load the CSV,
validate the schema (fail loud if a relied-on column is missing), coerce types, and add
the derived time columns (year, month, year-month) that the analytics layer groups by.
A cached singleton means the 2,500-row frame is parsed once per process.

Everything downstream (analytics, the stats retriever, the charts) reads THIS frame, so
there is exactly one place where the data enters the system.
"""

import functools

import pandas as pd

import config


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"sales data is missing expected columns: {missing}")
    df = df[list(config.EXPECTED_COLUMNS)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])                       # a row with no date can't be analyzed
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Customer_Age"] = pd.to_numeric(df["Customer_Age"], errors="coerce")
    df["Customer_Satisfaction"] = pd.to_numeric(df["Customer_Satisfaction"], errors="coerce")
    # Derived grouping keys (computed once here, not re-derived in each analysis).
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["AgeBand"] = df["Customer_Age"].apply(_age_band)
    return df


def _age_band(age) -> str:
    if pd.isna(age):
        return "unknown"
    for lo, hi, label in config.AGE_BANDS:
        if lo <= age < hi:           # half-open [lo, hi): contiguous, no boundary gaps
            return label
    return "unknown"


@functools.lru_cache(maxsize=1)
def load(path: str | None = None) -> pd.DataFrame:
    """The structured sales frame (cached per process). Pass a path in tests for isolation."""
    return _load(path or config.DATA_PATH)


def reset_cache() -> None:
    """Drop the cached frame (tests that swap the data path call this)."""
    load.cache_clear()
