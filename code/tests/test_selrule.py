"""selrule.py: prompt set chosen from the end of the prompt (H26, H27)."""

import argparse

import mlx.core as mx
import numpy as np
import pytest

import segroute as sr
import selrule as sl


def probs(rows):
    return mx.array(np.array(rows, dtype=np.float32))


HEAD = [0.5, 0.5, 0.0, 0.0]
TAIL = [0.0, 0.0, 0.5, 0.5]


def controller(tail):
    c = sl.TailController(1, 4, tail)
    c.mode, c.k, c.s = "mask", 2, 2
    return c


def test_prompt_set_comes_from_the_tail_only():
    c = controller(lambda: 1)
    c.observe(0, probs([HEAD, HEAD, TAIL]))
    assert c.allowed[0] == {2, 3}
    assert c.loads[0] == 2


def test_a_tail_longer_than_the_prompt_uses_the_whole_prompt():
    c = controller(lambda: 99)
    c.observe(0, probs([HEAD, HEAD, TAIL]))
    assert c.allowed[0] == {0, 1}


@pytest.mark.parametrize("tail", [None, 1])
def test_decode_steps_behave_as_in_the_base_controller(tail):
    base = sr.Controller(1, 4)
    base.mode, base.k, base.s = "mask", 2, 2
    c = controller(lambda: tail)
    for ctl in (base, c):
        ctl.observe(0, probs([TAIL]))
        for row in (HEAD, HEAD, HEAD):
            ctl.observe(0, probs([row]))
    assert c.allowed == base.allowed and c.loads == base.loads and c.reselects == base.reselects


def test_no_tail_is_the_base_rule():
    c = controller(lambda: None)
    c.observe(0, probs([HEAD, HEAD, TAIL]))
    assert c.allowed[0] == {0, 1}


def test_unmasked_mode_is_untouched():
    c = sl.TailController(1, 4, lambda: 1)
    assert c.observe(0, probs([HEAD])) is None


def test_repeat_arm_fills_with_the_item_before_the_unchanged_prompt():
    head = "Classify this. Reply with the label only.\n\n"
    plain = ("classify", 3, head + "abc", "World")
    task, i, prompt, ref = sl.repeat_item(plain, 7)
    assert (task, i, ref) == ("classify", 3, "World")
    assert prompt == "abcabca\n\n" + head + "abc"


def test_after_arm_puts_the_filler_after_the_prompt():
    plain = ("classify", 0, "p", "r")
    assert sl.after_item(plain, "y" * 4000, 5) == ("classify", 0, "p\n\nyyyyy", "r")


def test_tag_names_the_arm():
    assert sl.tag("gemma", "sel") == "gemma_pl_sel"
    assert sl.tag("gemma", "tail64") == "gemma_tail64"
    with pytest.raises(ValueError):
        sl.tag("gemma", "nope")


D = {"mean": 0.3, "lo": 0.1, "hi": 0.5}
LOW = {"mean": 0.1, "lo": 0.05, "hi": 0.2}


@pytest.mark.parametrize(
    ("sel", "rep", "agree", "want"),
    [
        (D, LOW, 0.85, {"selection": True, "content": False, "suffices": True, "verdict": "selection"}),
        (D, LOW, 0.5, {"selection": True, "content": False, "suffices": False, "verdict": "selection"}),
        (LOW, D, 0.5, {"selection": False, "content": True, "suffices": False, "verdict": "content"}),
        (D, D, 0.5, {"selection": True, "content": True, "suffices": False, "verdict": "both"}),
        (LOW, LOW, 0.5, {"selection": False, "content": False, "suffices": False, "verdict": "fails"}),
    ],
)
def test_h26_clauses(sel, rep, agree, want):
    models = {
        "gemma": {"sel": sel, "rep": rep, "sel_first": agree, "plain_first": 0.9},
        "qwen36": {"sel": sel, "rep": rep, "sel_first": agree, "plain_first": 0.9},
    }
    assert sl.h26_clauses(models) == want


def test_h26_needs_both_models():
    good = {"sel": D, "rep": LOW, "sel_first": 0.5, "plain_first": 0.9}
    bad = {"sel": LOW, "rep": LOW, "sel_first": 0.5, "plain_first": 0.9}
    assert sl.h26_clauses({"gemma": good, "qwen36": bad})["verdict"] == "fails"


UP = {"mean": -0.06, "lo": -0.09, "hi": -0.02}  # intact minus arm in kept_difference: the arm keeps more
FLAT = {"mean": -0.01, "lo": -0.05, "hi": 0.03}


@pytest.mark.parametrize(
    ("row", "want"),
    [
        ([UP, UP, FLAT], "holds"),
        ([UP, FLAT, FLAT], "fails"),
        ([{"mean": -0.04, "lo": -0.08, "hi": -0.01}, UP, FLAT], "fails"),
    ],
)
def test_h27_verdict(row, want):
    assert sl.h27_verdict({"gemma": row, "qwen36": [UP, UP, UP]}) == want


def test_sel_arm_sets_the_tail_to_the_plain_prompt_length(monkeypatch):
    """The set comes from the plain prompt tokens, per padded item, and the patches do not outlive the run."""
    plain = [("classify", 0, "abc", "r"), ("summarize", 0, "x", "r")]
    padded = [("classify", 0, "filler\n\nabc", "r")]
    monkeypatch.setattr(sl.short, "items", lambda n: plain)
    monkeypatch.setattr(sl, "_arm_items", lambda arm, n: padded)
    monkeypatch.setattr(sl.sr, "chat", lambda tok, text: list(text))
    seen = {}

    def fake_run(a):
        ctl = sl.sr.install(FakeModel())
        for _, _, prompt, _ in sl.short.items(a.n):
            ids = sl.sr.chat(None, prompt)
            seen[prompt] = (len(ids), ctl.tail())
        seen["tag"] = a.tag

    monkeypatch.setattr(sl.short, "cmd_run", fake_run)
    monkeypatch.setattr(sl, "install", lambda model, tail, rule: sl.TailController(1, 4, tail, rule))
    orig_install = sl.sr.install
    sl.cmd_run(argparse.Namespace(model_tag="gemma", arm="sel", frac=0.125, n=1))
    assert seen["filler\n\nabc"] == (len("filler\n\nabc"), 3)
    assert seen["tag"] == "gemma_pl_sel"
    assert sl.sr.install is orig_install and sl.short.items(1) == plain


def test_masked_only_arms_refuse_an_unmasked_run():
    with pytest.raises(SystemExit):
        sl.cmd_run(argparse.Namespace(model_tag="gemma", arm="sel", frac=0, n=1))


class FakeModel:
    pass


def test_head_rule_chooses_from_the_first_tokens():
    c = sl.TailController(1, 4, lambda: 1, rule="head")
    c.mode, c.k, c.s = "mask", 2, 2
    c.observe(0, probs([HEAD, TAIL, TAIL]))
    assert c.allowed[0] == {0, 1}


def test_head_arm_tag():
    assert sl.tag("qwen36", "head64") == "qwen36_head64"


A = [0.9, 0.1, 0.0, 0.0]
Z = [0.1, 0.9, 0.0, 0.0]
M = [0.0, 0.0, 0.5, 0.5]


def ends(tail):
    c = sl.TailController(1, 4, lambda: tail, rule="ends")
    c.mode, c.k, c.s = "mask", 2, 2
    return c


def test_ends_rule_chooses_from_both_ends_of_the_prompt():
    c = ends(2)
    c.observe(0, probs([A, M, M, M, Z]))
    assert c.allowed[0] == {0, 1}


def test_ends_rule_does_not_count_a_token_twice_on_a_short_prompt():
    c = ends(4)
    c.observe(0, probs([M, A, M]))
    assert c.allowed[0] == {2, 3}


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        sl.TailController(1, 4, lambda: 1, rule="middle")


def test_ends_arm_tag_and_nemotron_budget():
    assert sl.tag("nemotron", "ends64") == "nemotron_ends64"
    assert sl.BUDGETS == {"gemma": 16, "nemotron": 16, "qwen36": 32}


def test_at_bar_needs_every_task():
    good = {"mean": 0.95, "lo": 0.91, "hi": 0.99}
    near = {"mean": 0.88, "lo": 0.84, "hi": 0.91}
    low = {"mean": 0.80, "lo": 0.75, "hi": 0.85}
    assert sl.at_bar([good, near, good])
    assert not sl.at_bar([good, low, good])


def test_h29_needs_two_models_at_the_bar():
    assert sl.h29_verdict({"a": True, "b": True, "c": False}) == "holds"
    assert sl.h29_verdict({"a": True, "b": False, "c": False}) == "fails"


def test_h29_rule_arm_of_kept_with_ci():
    base = {i: 1.0 for i in range(20)}
    arm = {i: 0.5 for i in range(20)}
    k = sl.kept_ci(arm, base)
    assert k["mean"] == pytest.approx(0.5) and k["lo"] == pytest.approx(0.5) and k["hi"] == pytest.approx(0.5)


def test_disjoint_arm_tags():
    assert sl.tag("gemma", "whole", 300) == "gemma_d300"
    assert sl.tag("gemma", "ends64", 300) == "gemma_d300_ends64"
    assert sl.tag("gemma", "ends64") == "gemma_ends64"


def test_whole_arm_only_runs_on_disjoint_items():
    with pytest.raises(ValueError):
        sl.tag("gemma", "whole")


def test_rule_arm_refuses_an_unmasked_run_on_disjoint_items():
    with pytest.raises(SystemExit):
        sl.cmd_run(argparse.Namespace(model_tag="gemma", arm="ends64", frac=0, n=1, start=300))


def test_disjoint_whole_run_passes_items_from_start_and_keeps_no_tail(monkeypatch):
    seen = {}

    def fake_run(a):
        seen["tag"] = a.tag
        seen["items"] = sl.short.items(a.n)
        ctl = sl.sr.install(FakeModel())
        seen["tail"] = ctl.tail()

    calls = []
    monkeypatch.setattr(sl.short, "items", lambda n, start=0: calls.append((n, start)) or [("qa", start, "p", None)])
    monkeypatch.setattr(sl.short, "cmd_run", fake_run)
    monkeypatch.setattr(sl, "install", lambda model, tail, rule: sl.TailController(1, 4, tail, rule))
    orig = sl.short.items
    sl.cmd_run(argparse.Namespace(model_tag="gemma", arm="whole", frac=0, n=1, start=300))
    assert seen == {"tag": "gemma_d300", "items": [("qa", 300, "p", None)], "tail": None}
    assert calls == [(1, 300)]
    assert sl.short.items is orig


def test_h30_needs_two_models_at_the_bar_and_the_rule_above_whole():
    up = {"mean": 0.1, "lo": 0.02, "hi": 0.2}
    flat = {"mean": 0.0, "lo": -0.1, "hi": 0.1}
    row = {"at_bar": True, "diffs": [flat, up, flat]}
    miss = {"at_bar": False, "diffs": [flat, flat, flat]}
    assert sl.h30_verdict({"a": row, "b": row, "c": miss}) == "holds"
    assert sl.h30_verdict({"a": row, "b": miss, "c": miss}) == "fails"
    assert sl.h30_verdict({"a": row, "b": {**row, "diffs": [flat] * 3}, "c": miss}) == "fails"


def test_frontier_is_the_smallest_budget_at_the_bar():
    rows = {
        16: {"at_bar": True, "clean": False},
        8: {"at_bar": False, "clean": False},
        24: {"at_bar": True, "clean": True},
    }
    assert sl.frontier(rows, "at_bar") == 16
    assert sl.frontier(rows, "clean") == 24
    assert sl.frontier({8: {"at_bar": False}}, "at_bar") is None


def test_h31_needs_gemma_at_3_16_and_one_model_at_3_32():
    n = {"gemma": 128, "nemotron": 128, "qwen36": 256}
    assert sl.h31_verdict({"gemma": 24, "nemotron": 12, "qwen36": 32}, n) == "holds"
    assert sl.h31_verdict({"gemma": 24, "nemotron": 16, "qwen36": 24}, n) == "holds"
    assert sl.h31_verdict({"gemma": 32, "nemotron": 8, "qwen36": 16}, n) == "fails"
    assert sl.h31_verdict({"gemma": 24, "nemotron": 16, "qwen36": 32}, n) == "fails"
    assert sl.h31_verdict({"gemma": None, "nemotron": 8, "qwen36": None}, n) == "fails"


def test_loop_rate_of_one_task():
    recs = [{"task": "qa", "gen_ids": [7] * 40}, {"task": "qa", "gen_ids": [1, 2]}, {"task": "x", "gen_ids": [7] * 40}]
    assert sl.loop_rate(recs, "qa") == 0.5


def test_repeat_cells_are_the_h31_frontiers_not_already_run():
    assert sl.REPEAT_CELLS == {"gemma": (24,), "qwen36": (24, 48)}


def test_a_rule_arm_without_a_delta_refuses_the_unmasked_run():
    a = argparse.Namespace(arm="ends64", frac=0, delta=None, model_tag="gemma", n=1, start=0)
    with pytest.raises(SystemExit):
        sl.cmd_run(a)


def test_identical_share():
    a = [{"task": "qa", "i": 0, "gen_ids": [1]}, {"task": "qa", "i": 1, "gen_ids": [2]}]
    b = [{"task": "qa", "i": 0, "gen_ids": [1]}, {"task": "qa", "i": 1, "gen_ids": [3]}]
    assert sl.identical_share(a, a) == 1.0
    assert sl.identical_share(a, b) == 0.5


def test_article_rule_chooses_only_article_tokens():
    ctl = sl.TailController(1, 4, lambda: 4, "article")
    ctl.span = (1, 3)
    p = probs([HEAD, TAIL, TAIL, HEAD, HEAD])
    assert np.array(ctl._chosen(p, 4)).tolist() == np.array(p[1:3]).tolist()


def test_article_rule_takes_both_ends_of_a_long_article():
    ctl = sl.TailController(1, 4, lambda: 2, "article")
    ctl.span = (1, 5)
    p = probs([HEAD, TAIL, HEAD, HEAD, TAIL, HEAD])
    assert np.array(ctl._chosen(p, 2)).tolist() == [TAIL, TAIL]


def test_reserve_bonus_marks_the_top_experts_of_each_span():
    p = probs([[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4]])
    bonus = np.array(sl.reserve_bonus(p, [(0, 1), (2, 3)], 1))
    assert bonus[0] > 0 and bonus[2] > 0 and bonus[1] == 0 and bonus[3] == 0


def test_reserve_bonus_with_two_per_span():
    p = probs([[0.5, 0.3, 0.2, 0.0]])
    bonus = np.array(sl.reserve_bonus(p, [(0, 1)], 2))
    assert bonus[0] > 0 and bonus[1] > 0 and bonus[2] == 0


def test_reserved_experts_enter_the_set_ahead_of_heavier_ones():
    ctl = sl.TailController(1, 4, lambda: 64, "ends")
    ctl.mode, ctl.k, ctl.s = "mask", 2, 32
    ctl.reserved, ctl.reserve_m = [(0, 1)], 1
    p = probs([[0.0, 0.0, 0.0, 1.0], [0.6, 0.4, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]])
    ctl.observe(0, p)
    assert 3 in ctl.allowed[0] and 0 in ctl.allowed[0]
