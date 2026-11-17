"""copyaudit.py: how much of a summary is lifted from its article, pure parts."""

import copyaudit as ca


def test_copied_share_counts_words_in_runs_of_eight_or_more():
    article = "one two three four five six seven eight nine ten eleven twelve"
    assert ca.copied_share("one two three four five six seven eight", article) == 1.0
    assert ca.copied_share("one two three four five six seven nope", article) == 0.0
    half = "one two three four five six seven eight a b c d e f g h"
    assert ca.copied_share(half, article) == 0.5


def test_copied_share_ignores_case_and_punctuation_and_short_answers():
    article = "One, two; three four five six seven eight."
    assert ca.copied_share("one two three four five six seven eight", article) == 1.0
    assert ca.copied_share("one two", article) == 0.0


def test_first_copy_position_is_the_share_of_the_article_before_the_first_run():
    article = " ".join(f"w{i}" for i in range(100))
    answer = " ".join(f"w{i}" for i in range(50, 60))
    assert ca.first_copy_position(answer, article) == 0.5
    assert ca.first_copy_position("nothing here at all", article) is None
