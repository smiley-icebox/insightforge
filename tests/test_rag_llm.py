"""The LLM synthesis path with a STUB model — exercises the numeric-grounding gate that
only runs when USE_LLM is on, with no API key."""

import config
import llm
import rag


class _Msg:
    def __init__(self, text): self.content = text


class _Model:
    def __init__(self, text): self._t = text
    def invoke(self, _messages): return _Msg(self._t)


def _use_llm(monkeypatch, text):
    monkeypatch.setattr(config, "USE_LLM", True)
    monkeypatch.setattr(llm, "chat_model", lambda *a, **k: _Model(text))


def test_grounded_llm_answer_ships(monkeypatch):
    _use_llm(monkeypatch, "Widget A leads with 375235 in total sales, a 27.13% share.")
    out = rag.answer("which product sells best?")
    assert out["grounded"] is True
    assert "375235" in out["answer"]


def test_invented_number_falls_back_to_deterministic(monkeypatch):
    # The model fabricates a figure not in the stats — the gate must reject it.
    _use_llm(monkeypatch, "Total sales skyrocketed to 9,999,999 last quarter!")
    out = rag.answer("what were total sales?")
    assert out["grounded"] is False
    assert "9,999,999" not in out["answer"] and "9999999" not in out["answer"]


def test_empty_llm_output_falls_back(monkeypatch):
    _use_llm(monkeypatch, "")
    out = rag.answer("which region sells most?")
    assert out["grounded"] is False
    assert "Region" in out["answer"]  # deterministic stat block shipped
