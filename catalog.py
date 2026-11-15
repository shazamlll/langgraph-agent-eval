"""In-memory book catalog for the demo bookstore assistant.

All books are invented for this demonstration. No real titles, authors, or
company data are used anywhere in this repository.
"""

from __future__ import annotations

# Genre vocabulary used by the stub responder and the search tool.
GENRES = [
    "science fiction",
    "fantasy",
    "mystery",
    "romance",
    "history",
    "thriller",
    "biography",
    "horror",
]

BOOKS = [
    {
        "id": "b01",
        "title": "The Salt Road",
        "author": "Mara Ellison",
        "genre": "science fiction",
        "price": 14.99,
        "rating": 4.6,
        "in_stock": True,
        "keywords": ["space", "colony", "mars"],
    },
    {
        "id": "b02",
        "title": "Orbit of Echoes",
        "author": "Dario Venn",
        "genre": "science fiction",
        "price": 18.99,
        "rating": 4.8,
        "in_stock": True,
        "keywords": ["space", "station", "orbit"],
    },
    {
        "id": "b03",
        "title": "The Long Silence",
        "author": "Ines Kaur",
        "genre": "science fiction",
        "price": 12.99,
        "rating": 4.2,
        "in_stock": True,
        "keywords": ["first contact", "signal"],
    },
    {
        "id": "b04",
        "title": "The Cartographer's Daughter",
        "author": "Tomas Rill",
        "genre": "fantasy",
        "price": 16.99,
        "rating": 4.7,
        "in_stock": True,
        "keywords": ["maps", "magic", "kingdom"],
    },
    {
        "id": "b05",
        "title": "Ember Crown",
        "author": "Sana Iqbal",
        "genre": "fantasy",
        "price": 12.99,
        "rating": 4.5,
        "in_stock": True,
        "keywords": ["dragons", "magic", "war"],
    },
    {
        "id": "b06",
        "title": "The Quiet Witness",
        "author": "Helen Marsh",
        "genre": "mystery",
        "price": 13.99,
        "rating": 4.9,
        "in_stock": True,
        "keywords": ["detective", "murder", "london"],
    },
    {
        "id": "b07",
        "title": "A Study in Smoke",
        "author": "Edwin Cole",
        "genre": "mystery",
        "price": 11.99,
        "rating": 4.3,
        "in_stock": True,
        "keywords": ["detective", "victorian", "fog"],
    },
    {
        "id": "b08",
        "title": "Letters from Lisbon",
        "author": "Ana Duarte",
        "genre": "romance",
        "price": 10.99,
        "rating": 4.4,
        "in_stock": True,
        "keywords": ["love", "letters", "lisbon"],
    },
    {
        "id": "b09",
        "title": "The Spice Merchants",
        "author": "Jonas Berg",
        "genre": "history",
        "price": 15.99,
        "rating": 4.5,
        "in_stock": True,
        "keywords": ["trade", "spice", "venice"],
    },
    {
        "id": "b10",
        "title": "The Last Ferry",
        "author": "Nadia Petrova",
        "genre": "thriller",
        "price": 17.99,
        "rating": 4.7,
        "in_stock": True,
        "keywords": ["island", "ferry", "storm"],
    },
    {
        "id": "b11",
        "title": "Ada in the Margins",
        "author": "Priya Nair",
        "genre": "biography",
        "price": 19.99,
        "rating": 4.6,
        "in_stock": True,
        "keywords": ["mathematician", "letters", "victorian"],
    },
    {
        "id": "b12",
        "title": "The Hollow House",
        "author": "Victor Anselm",
        "genre": "horror",
        "price": 9.99,
        "rating": 4.1,
        "in_stock": True,
        "keywords": ["haunted", "house", "ghost"],
    },
]


def _matches(book: dict, keywords: list[str]) -> bool:
    """True if any keyword matches the book's title, author, or keywords."""
    haystack = " ".join(
        [book["title"], book["author"], book["genre"], *book["keywords"]]
    ).lower()
    return any(kw.lower() in haystack for kw in keywords)


def search_books(
    genre: str | None = None,
    keywords: list[str] | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    limit: int = 5,
) -> list[dict]:
    """Search the in-memory catalog.

    Results are sorted by rating (highest first). Returns plain dicts so the
    output is easy to log, serialize, and assert on in tests.
    """
    results = list(BOOKS)
    if genre:
        results = [b for b in results if b["genre"] == genre.lower()]
    if keywords:
        results = [b for b in results if _matches(b, keywords)]
    if max_price is not None:
        results = [b for b in results if b["price"] <= max_price]
    if min_rating is not None:
        results = [b for b in results if b["rating"] >= min_rating]
    results.sort(key=lambda b: b["rating"], reverse=True)
    return results[:limit]


def get_book(book_id: str) -> dict | None:
    """Return a single book by id, or None."""
    for book in BOOKS:
        if book["id"] == book_id:
            return book
    return None


def all_keywords() -> set[str]:
    """Vocabulary of searchable topic keywords (used by the stub responder)."""
    vocab: set[str] = set()
    for book in BOOKS:
        vocab.update(k.lower() for k in book["keywords"])
        vocab.update(w.lower() for w in book["title"].split() if len(w) > 3)
    return vocab
