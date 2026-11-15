"""Tests for the synthetic catalog and its LangChain tool wrapper."""

from agent import search_catalog_tool
from catalog import get_book, search_books


def test_search_by_genre_sorted_by_rating():
    results = search_books(genre="mystery")
    assert [b["title"] for b in results] == ["The Quiet Witness", "A Study in Smoke"]
    assert all(b["genre"] == "mystery" for b in results)


def test_search_by_keyword():
    results = search_books(keywords=["space"])
    titles = {b["title"] for b in results}
    assert titles == {"Orbit of Echoes", "The Salt Road"}


def test_search_by_price_filter():
    results = search_books(genre="fantasy", max_price=15.0)
    assert [b["title"] for b in results] == ["Ember Crown"]


def test_search_by_min_rating():
    results = search_books(min_rating=4.8)
    assert {b["title"] for b in results} == {"Orbit of Echoes", "The Quiet Witness"}


def test_search_no_match_returns_empty():
    assert search_books(keywords=["cooking"]) == []
    assert search_books(genre="science fiction", max_price=1.0) == []


def test_search_default_is_top_rated():
    results = search_books()
    assert results[0]["title"] == "The Quiet Witness"  # highest rated
    assert len(results) == 5  # default limit


def test_get_book():
    assert get_book("b06")["title"] == "The Quiet Witness"
    assert get_book("nope") is None


def test_tool_wrapper_invoke():
    results = search_catalog_tool.invoke({"genre": "thriller"})
    assert [b["title"] for b in results] == ["The Last Ferry"]
    assert search_catalog_tool.invoke({})[0]["title"] == "The Quiet Witness"
