"""failures.py: common metric and failure types, from stored token ids."""

import pytest

import failures as fl


def test_lcs_f1_identical_is_one_and_disjoint_is_zero():
    assert fl.lcs_f1([1, 2, 3], [1, 2, 3]) == 1.0
    assert fl.lcs_f1([1, 2, 3], [4, 5]) == 0.0


def test_lcs_f1_survives_a_one_token_shift():
    a, b = [9, 1, 2, 3, 4], [1, 2, 3, 4]
    assert fl.lcs_f1(a, b) == pytest.approx(2 * 0.8 * 1.0 / 1.8)


def test_lcs_f1_of_two_empty_answers_is_one_and_of_one_empty_is_zero():
    assert fl.lcs_f1([], []) == 1.0
    assert fl.lcs_f1([], [1]) == 0.0


def test_is_loop_finds_a_repeated_tail_block():
    assert fl.is_loop([5, 6, 7] + [1, 2, 3] * 8)
    assert fl.is_loop([4] * 24)


def test_is_loop_needs_the_whole_tail():
    assert not fl.is_loop(list(range(40)))
    assert not fl.is_loop([1, 2, 3] * 3)  # shorter than the tail
    assert not fl.is_loop([1, 2, 3] * 8 + [9])


def test_failure_type_order():
    assert fl.failure_type([], "", 120) == "empty"
    assert fl.failure_type([7, 8], " \n", 120) == "empty"
    assert fl.failure_type([1, 2] * 60, "ab" * 60, 120) == "loop"
    assert fl.failure_type(list(range(120)), "text", 120) == "limit"
    assert fl.failure_type([1, 2, 3], "text", 120) == "other"


def test_first_token_agrees():
    assert fl.first_token_agrees([1, 2], [1, 3])
    assert not fl.first_token_agrees([2], [1])
    assert not fl.first_token_agrees([], [1])


def test_echoes_instruction():
    assert fl.echoes_instruction(
        "Summarize this news article in three sentences. Then",
        "Summarize this news article in three sentences.\n\nBody",
    )
    assert not fl.echoes_instruction(
        "A storm hit the coast.", "Summarize this news article in three sentences.\n\nBody"
    )


def test_gap_ci_is_above_zero_for_a_clear_gap_and_keeps_the_means():
    g = fl.gap([0.8] * 50 + [0.9] * 50, [0.1] * 50 + [0.2] * 50)
    assert g["mean"] == pytest.approx(0.7)
    assert 0.6 < g["lo"] < g["hi"] < 0.8


def test_gap_refuses_an_empty_side():
    with pytest.raises(ValueError):
        fl.gap([], [0.1])


def test_quartiles_are_by_unmasked_length_and_cover_every_item():
    lengths = {i: i for i in range(8)}
    q = fl.length_quartiles(lengths)
    assert [q[i] for i in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_verdict_holds_only_with_all_three_parts_on_both_models():
    good = {
        "a_gap": {"mean": 0.2, "lo": 0.1, "hi": 0.3},
        "b_excess": {"mean": 0.3, "lo": 0.15, "hi": 0.4},
        "gap_clean": {"mean": 0.05, "lo": 0.0, "hi": 0.1},
    }
    assert fl.verdict({"gemma": good, "qwen36": good}) == "holds"
    weak_b = good | {"b_excess": {"mean": 0.05, "lo": 0.01, "hi": 0.1}}
    assert fl.verdict({"gemma": good, "qwen36": weak_b}) == "fails"
    no_shrink = good | {"gap_clean": {"mean": 0.15, "lo": 0.1, "hi": 0.2}}
    assert fl.verdict({"gemma": no_shrink, "qwen36": good}) == "fails"
    no_gap = good | {"a_gap": {"mean": 0.0, "lo": -0.1, "hi": 0.1}}
    assert fl.verdict({"gemma": no_gap, "qwen36": good}) == "metric effect: gemma"


def test_spearman_handles_ties_and_direction():
    assert fl.spearman([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(0.8944, abs=1e-3)
    assert fl.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_is_zero_when_one_side_is_constant():
    assert fl.spearman([1, 2, 3], [1, 1, 1]) == 0.0


def test_spearman_ci_brackets_the_estimate():
    x = list(range(200))
    y = [float(i > 100) for i in x]
    c = fl.spearman_ci(x, y)
    assert c["lo"] < c["mean"] < c["hi"] and c["lo"] > 0


def test_h24_verdict_needs_two_of_three_negative_on_both_models():
    neg, zero = {"lo": -0.3, "hi": -0.1}, {"lo": -0.1, "hi": 0.1}
    assert fl.h24_verdict({"a": {"x": neg, "y": neg, "z": zero}, "b": {"x": neg, "y": zero, "z": neg}}) == "holds"
    assert fl.h24_verdict({"a": {"x": neg, "y": zero, "z": zero}, "b": {"x": neg, "y": neg, "z": neg}}) == "fails"
