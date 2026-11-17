"""baseline.py: copying the article's opening, and how the jury's verdicts depend on its members, pure parts."""

import pytest

import baseline as bl


def test_lead_sentences_takes_the_first_sentences():
    a = "One two. Three four! Five six? Seven."
    assert bl.lead_sentences(a, 3) == "One two. Three four! Five six?"
    assert bl.lead_sentences("No stop here", 3) == "No stop here"


def test_lead_words_keeps_the_first_words():
    assert bl.lead_words("a b c d", 2) == "a b"
    assert bl.lead_words("a  b", 9) == "a b"


def test_article_of_strips_the_instruction():
    p = "Summarize this news article in 2 or 3 sentences.\n\nBody one.\n\nBody two."
    assert bl.article_of(p) == "Body one.\n\nBody two."


def test_fleiss_kappa_perfect_and_chance():
    assert bl.fleiss_kappa([[True, True, True], [False, False, False]]) == pytest.approx(1.0)
    # Every item split the same way: agreement below chance.
    assert bl.fleiss_kappa([[True, True, False]] * 4) < 0


def test_fleiss_kappa_is_undefined_when_all_votes_match():
    assert bl.fleiss_kappa([[True, True, True]] * 3) is None


def test_accept_rules():
    yes = {"supported": True, "condensed": True, "covers": True}
    no = {"supported": False, "condensed": True, "covers": True}
    assert bl.accepted([yes, yes, no], "majority") is True
    assert bl.accepted([yes, yes, no], "unanimous") is False
    assert bl.accepted([yes, no, no], "majority") is False


def test_majority_is_per_question_not_per_juror():
    a = {"supported": False, "condensed": True, "covers": True}
    b = {"supported": True, "condensed": False, "covers": True}
    c = {"supported": True, "condensed": True, "covers": False}
    assert bl.accepted([a, b, c], "majority") is True


def test_point_biserial():
    assert bl.point_biserial([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) > 0.9
    assert bl.point_biserial([0.5, 0.5], [True, False]) is None
    assert bl.point_biserial([0.1, 0.9], [True, True]) is None


def test_union_rouge_l_scores_each_reference_sentence_against_all_answer_sentences():
    ref = "a b c\nd e f"
    # Whole-text LCS keeps only one order; the union per reference sentence finds both.
    assert bl.rouge_l_union(ref, "d e f. a b c.") == pytest.approx(1.0)
    assert bl.rouge_l_union(ref, "x y z.") == 0.0


def test_lcs_length():
    assert bl.lcs_length(list("abcd"), list("axbd")) == 3


def test_foreign_numbers_are_digit_strings_absent_from_the_article():
    article = "He was 73 years old and earned 1,200 dollars in 2015."
    assert bl.foreign_numbers("He was 33 and earned 1200 in 2015.", article) == (["33"], 3)
    assert bl.foreign_numbers("No numbers here.", article) == ([], 0)


def test_foreign_numbers_do_not_match_inside_a_longer_number():
    assert bl.foreign_numbers("It cost 12.", "It cost 1200.") == (["12"], 1)


def test_foreign_names_skip_the_first_word_of_a_sentence():
    article = "Robert Bates spoke in Tulsa."
    assert bl.foreign_names("Bates spoke. Then Bates met Harris in Tulsa.", article) == ["Harris"]
    assert bl.foreign_names("Harris spoke.", article) == []
