"""curriculum.py: do experts share a common curriculum (H45 to H48, T3)? Pure parts."""

import numpy as np

import curriculum as cu


def test_arith_items_are_fixed_and_correct():
    a, b = cu.arith_short(50), cu.arith_short(50)
    assert a == b and len(a) == 50
    for prompt, gold in a[:10]:
        assert "number only" in prompt
        x = prompt.split("What is ")[1].split("?")[0]
        assert eval(x) == int(gold)  # noqa: S307


def test_arith_long_lists_twenty_four_numbers_and_sums_them():
    prompt, gold = cu.arith_long(5)[0]
    nums = [int(x) for x in prompt.split(":")[1].split(".")[0].split(",")]
    assert len(nums) == 24 and sum(nums) == int(gold)


def test_exposed_layers_are_evenly_spaced_and_counted():
    got = cu.exposed_layers(30, 13)
    assert len(got) == 13 and len(set(got)) == 13 and min(got) == 0 and max(got) <= 29
    assert max(np.diff(sorted(got))) <= 3


def test_jaccard_of_top_sets():
    a, b = np.array([3.0, 2, 1, 0]), np.array([0.0, 2, 3, 1])
    assert cu.top_jaccard(a, b, 2) == 1 / 3
    assert cu.top_jaccard(a, a, 2) == 1.0


def test_shared_energy_is_one_for_identical_and_low_for_opposed():
    same = np.ones((4, 3))
    assert abs(cu.shared_energy(same) - 1.0) < 1e-9
    assert cu.shared_energy(np.array([[1.0, 0], [-1.0, 0]])) < 1e-9


def test_pair_cosine_averages_distinct_pairs():
    x = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
    assert abs(cu.pair_cosine(x) - 1 / 3) < 1e-9


def test_token_type():
    assert cu.token_type("word") == "word" and cu.token_type(" 42") == "number"
    assert cu.token_type(",") == "punct" and cu.token_type("<start_of_turn>") == "template"


def test_h46_reading():
    at, off = {"mean": 0.95, "lo": 0.9, "hi": 1.0}, {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    assert cu.h46_reading(at, off) == "length"
    assert cu.h46_reading(off, off) == "ability"
    assert cu.h46_reading(at, at) == "word_problems"


def test_arith_long_takes_how_many_numbers_to_add():
    prompt, gold = cu.arith_long(5, 8)[0]
    nums = [int(x) for x in prompt.split(":")[1].split(".")[0].split(",")]
    assert len(nums) == 8 and sum(nums) == int(gold)
