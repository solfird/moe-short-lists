import pytest

import earlydamage as ed


def test_head_and_rest_split_on_words() -> None:
    text = "one two three four five"
    assert ed.head(text, 3) == "one two three"
    assert ed.rest(text, 3) == "four five"
    assert ed.rest("one two", 3) == ""


def test_number_counts_use_the_article() -> None:
    article = "The deputy, 73, shot a man in 2015."
    assert ed.number_counts("A 33-year-old deputy shot a man in 2015.", article) == (1, 2)
    assert ed.number_counts("A deputy shot a man.", article) == (0, 0)


def test_pooled_rate_ignores_items_with_no_number() -> None:
    assert ed.pooled_rate([(1, 2), (0, 0), (0, 2)]) == pytest.approx(0.25)
    assert ed.pooled_rate([(0, 0)]) is None


def test_paired_ci_of_a_clear_difference_is_above_zero() -> None:
    masked = [(1, 2)] * 40
    base = [(0, 2)] * 40
    lo, hi = ed.paired_ci(masked, base, ed.pooled_rate, n=300)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_head_over_half_copied() -> None:
    article = "the quick brown fox jumps over the lazy dog near the river bank today " * 2
    copied = "the quick brown fox jumps over the lazy dog near the river bank today and more words follow here now"
    assert ed.head_copied(copied, article, 16) is True
    fresh = "a summary in new words that shares nothing at all with the source text here"
    assert ed.head_copied(fresh, article, 16) is False


def test_clauses() -> None:
    assert ed.clause_a(0.30, 0.03, (0.10, 0.40)) is True
    assert ed.clause_a(0.05, 0.03, (0.01, 0.04)) is False
    assert ed.clause_a(0.30, 0.03, (-0.01, 0.40)) is False
    assert ed.clause_b(0.25, (0.18, 0.31)) is True
    assert ed.clause_b(0.15, (0.10, 0.20)) is False


def test_verdict() -> None:
    assert ed.verdict({"gemma": True, "qwen36": True}, {"nemotron": True, "qwen36": True}) == "early"
    assert ed.verdict({"gemma": False, "qwen36": False}, {"nemotron": False, "qwen36": False}) == "not early"
    assert ed.verdict({"gemma": True, "qwen36": False}, {"nemotron": True, "qwen36": True}) == "split"
