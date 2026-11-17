"""promptlen.py: arms of the prompt length test (H25)."""

import argparse

import pytest

import promptlen as pl
import short


def test_cut_keeps_the_prompt_form_and_scales_the_article():
    out = pl.summarize_items(["x" * 4000], ["ref"], 0.25)
    assert out == [("summarize", 0, "Summarize this news article in 2 or 3 sentences.\n\n" + "x" * 750, "ref")]


def test_padding_goes_before_an_unchanged_classify_prompt():
    plain = ("classify", 3, "Classify this.\n\nNews", "World")
    task, i, prompt, ref = pl.pad_item(plain, "y" * 4000)
    assert (task, i, ref) == ("classify", 3, "World")
    assert prompt == "y" * 3000 + "\n\nClassify this.\n\nNews"


def test_arm_tags_are_distinct_and_refuse_unknown_arms():
    assert pl.tag("gemma", "c8") == "gemma_pl_c8"
    assert len({pl.tag("gemma", a) for a in pl.ARMS}) == len(pl.ARMS)
    with pytest.raises(ValueError):
        pl.tag("gemma", "nope")


def test_paired_diff_matches_items_and_brackets_the_mean():
    a = {i: 1.0 if i % 2 else 0.0 for i in range(100)}
    b = dict.fromkeys(range(50, 150), 0.0)
    d = pl.paired_diff(a, b)
    assert d["n"] == 50
    assert d["mean"] == pytest.approx(0.5)
    assert d["lo"] < 0.5 < d["hi"]


def test_paired_diff_refuses_no_overlap():
    with pytest.raises(ValueError):
        pl.paired_diff({1: 0.0}, {2: 0.0})


@pytest.mark.parametrize(
    ("cut", "pad", "want"),
    [
        ((0.3, 0.1, 0.5), (-0.3, -0.5, -0.1), "holds"),
        ((0.3, 0.1, 0.5), (-0.1, -0.2, 0.0), "partial"),
        ((0.1, 0.05, 0.2), (-0.3, -0.5, -0.1), "fails"),
        ((0.3, -0.01, 0.5), (-0.3, -0.5, -0.1), "fails"),
    ],
)
def test_verdict(cut, pad, want):
    row = {
        "cut": dict(zip(("mean", "lo", "hi"), cut, strict=True)),
        "pad": dict(zip(("mean", "lo", "hi"), pad, strict=True)),
    }
    good = {
        "cut": {"mean": 0.3, "lo": 0.1, "hi": 0.5},
        "pad": {"mean": -0.3, "lo": -0.5, "hi": -0.1},
    }
    assert pl.verdict({"a": row, "b": good}) == want


@pytest.mark.data
def test_full_cut_and_plain_classify_match_the_short_suite():
    full = pl.arm_items("c1", 5)
    want = [x for x in short.items(5) if x[0] == "summarize"]
    assert full == want


def test_cmd_run_builds_the_arm_before_it_replaces_the_suite(monkeypatch):
    """The padded arm reads short.items to build itself, so that read must see the original suite."""
    plain = [("classify", 0, "p", "r")]
    monkeypatch.setattr(pl.short, "items", lambda n: plain)
    monkeypatch.setattr(pl, "arm_items", lambda arm, n: [(*x[:2], "padded " + x[2], x[3]) for x in pl.short.items(n)])
    seen = []
    monkeypatch.setattr(pl.short, "cmd_run", lambda a: seen.extend(pl.short.items(a.n)))
    pl.cmd_run(argparse.Namespace(model_tag="gemma", arm="pad", frac=0.125, n=1))
    assert seen == [("classify", 0, "padded p", "r")]
    assert pl.short.items(1) == plain
