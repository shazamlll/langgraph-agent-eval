"""Swappable LLM layer for the bookstore agent.

Default: :class:`StubResponder` — a deterministic, rule-based responder that
runs fully offline with no API key. Every behaviour the eval set exercises is
implemented here, so the graph, the trajectory logging, and the eval runner
can be tested end to end without any external service.

:class:`RealLLMAdapter` is a scaffold for a real provider. It is NOT wired
up: it reads an API key from the environment but its methods raise
``NotImplementedError`` until a follow-up wires in an actual provider call.
See README "Remaining work".
"""

from __future__ import annotations

import os
import re

from catalog import all_keywords

# Genre aliases the stub understands, mapped to canonical genre names.
GENRE_ALIASES = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "science fiction": "science fiction",
    "fantasy": "fantasy",
    "mystery": "mystery",
    "mysteries": "mystery",
    "detective": "mystery",
    "romance": "romance",
    "romantic": "romance",
    "history": "history",
    "historical": "history",
    "thriller": "thriller",
    "thrillers": "thriller",
    "biography": "biography",
    "memoir": "biography",
    "horror": "horror",
}

_PREFERENCE_RE = re.compile(
    r"\b(like|love|prefer|enjoy|into|fan of)\b", re.IGNORECASE
)
_RECOMMEND_RE = re.compile(r"\b(recommend|suggest|what should i read|pick)\b", re.IGNORECASE)
_SEARCH_RE = re.compile(
    r"\b(find|search|looking for|show me|do you have|any books|got anything)\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(r"^(hi|hello|hey)\b", re.IGNORECASE)
_PRICE_RE = re.compile(
    r"\b(?:under|less than|below|cheaper than)\s*\$?(\d+(?:\.\d+)?)", re.IGNORECASE
)
_RATING_RE = re.compile(r"\b(highly rated|top rated|best rated|best-rated)\b", re.IGNORECASE)


def _extract_genre(text: str) -> str | None:
    lowered = text.lower()
    for alias, genre in GENRE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return genre
    return None


def _extract_keywords(text: str) -> list[str]:
    lowered = text.lower()
    vocab = all_keywords()
    found = {kw for kw in vocab if re.search(rf"\b{re.escape(kw)}\b", lowered)}
    # "about X" topics: keep the topic even when it is not in the catalog,
    # so a search for an absent topic correctly returns no results instead
    # of falling back to "everything".
    about = re.search(r"\babout\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)", lowered)
    if about:
        topic = re.sub(r"\s+books?$", "", about.group(1).strip())
        if topic:
            found.add(topic)
    return sorted(found)


def _extract_max_price(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if match:
        return float(match.group(1))
    if re.search(r"\bcheap\b", text, re.IGNORECASE):
        return 15.0
    return None


def _extract_min_rating(text: str) -> float | None:
    if _RATING_RE.search(text):
        return 4.5
    return None


class StubResponder:
    """Deterministic, rule-based stand-in for an LLM.

    Intent classification and response generation are pure functions of the
    input text plus the agent state passed in, so runs are reproducible and
    the eval scores are stable.
    """

    def classify_intent(self, text: str) -> dict:
        """Classify one user turn.

        Returns a dict with ``intent`` (one of ``preference``, ``recommend``,
        ``search``, ``greeting``, ``chat``) plus extracted slots: ``genre``,
        ``keywords``, ``max_price``, ``min_rating``.
        """
        genre = _extract_genre(text)
        keywords = _extract_keywords(text)
        max_price = _extract_max_price(text)
        min_rating = _extract_min_rating(text)

        if _PREFERENCE_RE.search(text) and (genre or keywords):
            intent = "preference"
        elif _RECOMMEND_RE.search(text):
            intent = "recommend"
        elif _SEARCH_RE.search(text) or genre or keywords:
            intent = "search"
        elif _GREETING_RE.search(text.strip()):
            intent = "greeting"
        else:
            intent = "chat"
        return {
            "intent": intent,
            "genre": genre,
            "keywords": keywords,
            "max_price": max_price,
            "min_rating": min_rating,
        }

    def respond(
        self,
        *,
        intent: str,
        results: list[dict],
        preferences: dict,
        user_text: str,
        search_genre: str | None = None,
    ) -> str:
        """Build the agent's reply from the intent, tool results, and memory."""
        if intent == "greeting":
            return (
                "Hi! I'm the bookstore assistant. I can search our catalog, "
                "recommend books, and remember your preferences. "
                "What are you in the mood for?"
            )
        if intent == "preference":
            genre = preferences.get("genre")
            topics = preferences.get("topics") or []
            if genre:
                return (
                    f"Noted — I'll remember you like {genre}. "
                    "Ask me to recommend something any time and I'll pick "
                    f"from {genre} first."
                )
            if topics:
                return (
                    f"Noted — I'll remember you're interested in "
                    f"{', '.join(topics)}. Ask me to recommend something any "
                    "time and I'll keep that in mind."
                )
            return "Noted — I'll remember that."
        if intent in ("search", "recommend"):
            return self._results_reply(intent, results, preferences, search_genre)
        return (
            "I can help you find books, recommend something based on your "
            "taste, or remember your preferences. Try asking me to find a "
            "book or recommend one!"
        )

    def _results_reply(
        self, intent: str, results: list[dict], preferences: dict,
        search_genre: str | None = None,
    ) -> str:
        genre = preferences.get("genre")
        if not results:
            return (
                "I don't have any books matching that in our catalog right "
                "now. We carry science fiction, fantasy, mystery, romance, "
                "history, thriller, biography, and horror — want me to "
                "suggest something from one of those?"
            )
        lines = []
        for book in results:
            lines.append(
                f"- {book['title']} by {book['author']} — "
                f"${book['price']:.2f}, rated {book['rating']}/5"
            )
        listing = "\n".join(lines)
        if intent == "recommend" and genre:
            return (
                f"Since you like {genre}, I'd recommend "
                f"{results[0]['title']} by {results[0]['author']} — "
                f"${results[0]['price']:.2f}, rated {results[0]['rating']}/5. "
                f"Here are the top {genre} picks:\n{listing}"
            )
        if intent == "recommend":
            return (
                "Here are our top-rated books:\n"
                f"{listing}\n"
                "Tell me a genre you like and I can narrow it down."
            )
        # A search reply is labelled by the genre actually searched,
        # not by the remembered preference (they can differ).
        if intent == "search" and search_genre:
            return f"Here are some {search_genre} books I found:\n{listing}"
        return f"Here are some books I found:\n{listing}"


class RealLLMAdapter:
    """Scaffold for a real LLM provider. NOT wired up.

    Reads ``LLM_API_KEY`` (and optionally ``LLM_BASE_URL`` / ``LLM_MODEL``)
    from the environment, but every call raises ``NotImplementedError`` until
    a follow-up implements the provider request. This keeps the repo runnable
    offline while showing exactly where the real model plugs in.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def classify_intent(self, text: str) -> dict:
        raise NotImplementedError(
            "RealLLMAdapter is a scaffold: implement classify_intent() with "
            "a real provider call (prompt the model to return intent + slots "
            "as JSON). The stub responder covers this offline."
        )

    def respond(self, **kwargs) -> str:
        raise NotImplementedError(
            "RealLLMAdapter is a scaffold: implement respond() with a real "
            "provider call. The stub responder covers this offline."
        )


def get_llm():
    """Return the active LLM layer.

    Uses the real adapter only when ``USE_REAL_LLM=1`` is set; otherwise the
    deterministic stub. The real adapter is a scaffold and will raise if
    actually called — that is intentional until it is wired up.
    """
    if os.environ.get("USE_REAL_LLM") == "1":
        return RealLLMAdapter()
    return StubResponder()


__all__ = [
    "StubResponder",
    "RealLLMAdapter",
    "get_llm",
]
