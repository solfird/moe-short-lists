import pytest

import humanref


def test_summeval_label_bands() -> None:
    assert humanref.summeval_label(5.0) is True
    assert humanref.summeval_label(4.0) is False
    assert humanref.summeval_label(1.0) is False
    assert humanref.summeval_label(4.67) is None
    assert humanref.summeval_label(4.33) is None


def test_xsum_label_needs_two_clean_annotators() -> None:
    assert humanref.xsum_label({"a": [-1], "b": [-1], "c": [1]}) is True
    assert humanref.xsum_label({"a": [-1], "b": [0, 1], "c": [1]}) is False
    assert humanref.xsum_label({"a": [-1], "b": [-1], "c": [-1]}) is True


def test_sample_is_balanced_and_repeatable() -> None:
    rows = [{"k": n, "human": n % 3 == 0} for n in range(60)] + [{"k": 99, "human": None}]
    a = humanref.sample(rows, 5, seed=0)
    assert [r["human"] for r in a] == [True] * 5 + [False] * 5
    assert a == humanref.sample(rows, 5, seed=0)
    assert len({r["k"] for r in a}) == 10


def test_agreement_counts_each_class_apart() -> None:
    human = [True, True, True, True, False, False]
    verdict = [True, True, True, False, False, True]
    got = humanref.agreement(human, verdict)
    assert got["passed"] == pytest.approx(0.75)
    assert got["caught"] == pytest.approx(0.5)
    assert got["balanced"] == pytest.approx(0.625)


def test_balanced_ci_brackets_the_estimate() -> None:
    human = [True] * 50 + [False] * 50
    verdict = [True] * 40 + [False] * 10 + [False] * 45 + [True] * 5
    lo, hi = humanref.balanced_ci(human, verdict, n=500)
    assert lo < 0.85 < hi


def test_gate() -> None:
    assert humanref.gate({"summeval": 0.80, "xsum": 0.75}) == "holds"
    assert humanref.gate({"summeval": 0.80, "xsum": 0.60}) == "fails"
    assert humanref.gate({"summeval": 0.80, "xsum": 0.70}) == "neither"
