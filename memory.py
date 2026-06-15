"""Conversation memory — retain context across turns (the brief's "memory integration").

A small, explicit conversation buffer: the last N (question, answer) turns, rendered into
a compact context string that's injected into the RAG prompt. This lets follow-ups resolve
against earlier turns ("...and by region?" after a product question) and keeps the
assistant coherent across a session, without an unbounded or PII-heavy store.

Deliberately simple and in-process (Streamlit holds one per session). A production system
would persist per-user with summarization; that trade-off is documented in the README.
"""


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []

    def add(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))
        self._turns = self._turns[-self.max_turns:]

    def context(self) -> str:
        """Recent turns as prompt context (most recent last). Empty string if none."""
        if not self._turns:
            return ""
        lines = []
        for q, a in self._turns:
            lines.append(f"Earlier question: {q}\nEarlier answer: {a[:300]}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns = []
