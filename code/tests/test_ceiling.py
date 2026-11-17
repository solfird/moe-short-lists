"""ceiling.py: ceiling test, a fixed set per layer from the unmasked answer (H33)."""

import argparse

import mlx.core as mx
import numpy as np

import ceiling as oc
import segroute as sr


def test_chosen_tokens_are_both_prompt_ends_and_the_answer():
    assert oc.chosen_tokens(100, 103) == [*range(32), *range(68, 100), 100, 101, 102]


def test_a_short_prompt_is_chosen_whole():
    assert oc.chosen_tokens(5, 7) == [0, 1, 2, 3, 4, 5, 6]


def test_no_answer_leaves_the_prompt_ends():
    assert oc.chosen_tokens(64, 64) == list(range(64))


class FakeModel:
    """Two layers, three tokens, four experts: token 2 (the answer) wants experts 2 and 3 in layer 1."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.seen = []

    def __call__(self, x):
        self.seen.append(x.shape[1])
        rows = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]
        for layer in range(2):
            p = mx.array(np.array(rows, dtype=np.float32))
            self.ctl.observe(layer, p if layer == 1 else p[:, ::-1])


def test_fix_sets_masks_each_layer_from_a_full_pass_and_stops_reselection():
    ctl = sr.Controller(2, 4)
    ctl.k, ctl.s = 2, 32
    model = FakeModel(ctl)
    oc.fix_sets(ctl, model, [1, 2], [3], [2])
    assert model.seen == [3]
    assert ctl.mode == "mask" and ctl.s is None
    assert ctl.allowed[1] == {2, 3}
    assert ctl.allowed[0] == {0, 1}
    assert ctl.loads == [2, 2]
    assert ctl.reselects == [0, 0]


def test_a_fixed_set_does_not_move_during_generation():
    ctl = sr.Controller(1, 4)
    ctl.k, ctl.s = 2, 1
    ctl.mode = "log"
    ctl.observe(0, mx.array(np.array([[0.5, 0.5, 0, 0]], dtype=np.float32)))
    ctl.reset()
    ctl.mode, ctl.s = "mask", None
    ctl._select(0, np.array([1.0, 1.0, 0, 0]))
    for _ in range(5):
        ctl.observe(0, mx.array(np.array([[0, 0, 0.5, 0.5]], dtype=np.float32)))
    assert ctl.allowed[0] == {0, 1} and ctl.loads == [2]


def test_verdict_needs_two_models_at_the_bar():
    assert oc.h33_verdict({"a": True, "b": True, "c": False}) == "holds"
    assert oc.h33_verdict({"a": True, "b": False, "c": False}) == "fails"


def test_run_uses_the_stored_unmasked_answer_of_each_prompt(monkeypatch):
    seen = {}
    ctl = object()

    def fake_fix(c, model, ids, gen, idx=None):
        seen["fix"] = (c, ids, gen)

    def fake_run(a):
        seen["tag"] = a.tag
        got = sr.install("model")
        assert got is ctl
        ids = sr.chat(None, "prompt one")
        seen["gen"] = sr.generate("model", None, ids, 8, None)

    monkeypatch.setattr(oc.short, "items", lambda n: [("qa", 7, "prompt one", None)])
    monkeypatch.setattr(oc.short, "read_records", lambda p: [{"task": "qa", "i": 7, "gen_ids": [9, 9]}])
    monkeypatch.setattr(oc.short, "cmd_run", fake_run)
    monkeypatch.setattr(oc, "fix_sets", fake_fix)
    monkeypatch.setattr(oc.sr, "install", lambda model: ctl)
    monkeypatch.setattr(oc.sr, "chat", lambda tok, text: [1, 2])
    monkeypatch.setattr(oc.sr, "generate", lambda m, t, ids, mx_, cfg: [5])
    oc.cmd_run(argparse.Namespace(model_tag="gemma", k=12, n=1))
    assert seen["tag"] == "gemma_ceiling"
    assert seen["fix"] == (ctl, [1, 2], [9, 9])
    assert seen["gen"] == [5]
