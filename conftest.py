"""Shared fixtures — every test runs offline (USE_LLM off, so no API key) against the
committed sales_data.csv (read-only, so no isolation needed). The LLM synthesis path is
covered separately with a stub model."""

import pytest

import config
import data


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(config, "USE_LLM", False)
    data.reset_cache()
    yield
    data.reset_cache()
