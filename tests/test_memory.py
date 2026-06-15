"""Conversation memory — bounded buffer + context rendering."""

from memory import ConversationMemory


def test_context_empty_then_populated():
    m = ConversationMemory()
    assert m.context() == ""
    m.add("which product is best?", "Widget A")
    ctx = m.context()
    assert "which product is best?" in ctx and "Widget A" in ctx


def test_buffer_is_bounded():
    m = ConversationMemory(max_turns=2)
    for i in range(5):
        m.add(f"q{i}", f"a{i}")
    ctx = m.context()
    assert "q4" in ctx and "q3" in ctx and "q0" not in ctx   # only the last 2 kept


def test_clear():
    m = ConversationMemory()
    m.add("q", "a")
    m.clear()
    assert m.context() == ""
