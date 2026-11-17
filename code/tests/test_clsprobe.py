"""clsprobe.py: first-token label scores on classify (H38a), pure parts."""

import numpy as np
import pytest

import clsprobe as cp


def test_article_span_is_what_differs_from_the_empty_prompt():
    empty = [1, 2, 3, 8, 9]
    ids = [1, 2, 3, 40, 41, 42, 8, 9]
    assert cp.article_span(ids, empty) == (3, 6)


def test_article_span_with_an_article_that_repeats_the_suffix():
    assert cp.article_span([1, 2, 8, 9, 8, 9], [1, 2, 8, 9]) == (2, 4)


def test_offset_share_is_one_for_a_constant_shift_and_near_zero_for_noise():
    constant = np.tile(np.array([1.0, -1.0, 0.5, -0.5]), (200, 1))
    assert cp.offset_share(constant) == pytest.approx(1.0)
    noise = np.random.default_rng(0).normal(size=(2000, 4))
    assert cp.offset_share(noise) < 0.02


def test_offset_share_ignores_a_per_item_level():
    x = np.tile(np.array([1.0, -1.0, 0.0, 0.0]), (10, 1)) + np.arange(10)[:, None]
    assert cp.offset_share(x) == pytest.approx(1.0)


def test_rotate_list_moves_the_first_label_last():
    p = "Classify this text into one of: a, b, c. Reply with the label only.\n\nT"
    assert "one of: b, c, a. Reply" in cp.rotate_list(p, ("a", "b", "c"))


def test_emotion_prompt_lists_the_labels_in_the_given_order():
    assert cp.emotion_prompt("hello").startswith(
        "Classify this text into one of: sadness, joy, love, anger, fear, surprise."
    )
    assert cp.emotion_prompt("hello").endswith("\n\nhello")


def test_emotion_cells_have_their_own_paths():
    assert cp.cell_path("gemma", 12, "x", "ends", False, "emotion").name == "clsprobe_emotion_gemma_k12_x_ends.jsonl"
    assert cp.cell_path("gemma", 0, "", "", True, "emotion").name == "clsprobe_emotion_gemma_k0_rot.jsonl"
    assert cp.cell_path("gemma", 12, "x", "ends", False).name == "clsprobe_gemma_k12_x_ends.jsonl"


def test_rotate_list_by_n_cycles_the_labels():
    p = "one of: a, b, c, d. Reply"
    assert "b, c, d, a." in cp.rotate_list(p, ("a", "b", "c", "d"), 1)
    assert "c, d, a, b." in cp.rotate_list(p, ("a", "b", "c", "d"), 2)
    assert "d, a, b, c." in cp.rotate_list(p, ("a", "b", "c", "d"), 3)
    assert cp.rotate_list(p, ("a", "b", "c", "d"), 0) == p


def test_cell_path_names_each_order_and_keeps_the_h38a_names():
    assert cp.cell_path("g", 12, "x", "ends", False).name == "clsprobe_g_k12_x_ends.jsonl"
    assert cp.cell_path("g", 12, "x", "ends", True).name == "clsprobe_g_k12_x_ends_rot.jsonl"
    assert cp.cell_path("g", 12, "x", "ends", 1).name == "clsprobe_g_k12_x_ends_rot.jsonl"
    assert cp.cell_path("g", 12, "x", "ends", 2).name == "clsprobe_g_k12_x_ends_rot2.jsonl"
    assert cp.cell_path("g", 0, "", "", 3).name == "clsprobe_g_k0_rot3.jsonl"


def test_find_spans_takes_the_first_occurrence_of_any_spelling():
    ids = [9, 1, 7, 2, 3, 7, 1, 5]
    spans = cp.find_spans(ids, {"A": [[1]], "B": [[8], [2, 3]]})
    assert spans == {"A": (1, 2), "B": (3, 5)}


def test_find_spans_refuses_a_missing_label():
    with pytest.raises(ValueError, match="B"):
        cp.find_spans([1, 2], {"B": [[8]]})


def test_rule_parts():
    assert cp.rule_parts("ends") == ("ends", 0)
    assert cp.rule_parts("labels2") == ("ends", 2)
    assert cp.rule_parts("article") == ("article", 0)


def test_order_summary_uses_each_order():
    assert cp.order_summary([0.9, 0.7, 0.8, 0.8]) == pytest.approx({"mean": 0.8, "worst": 0.7, "gap": 0.2})


def test_offset_share_of_no_change_is_zero():
    assert cp.offset_share(np.zeros((5, 4))) == 0.0


def test_h40_needs_every_order_at_the_bar_on_every_model():
    good = {"mean": 0.95, "lo": 0.91, "hi": 0.99}
    by_ci = {"mean": 0.87, "lo": 0.82, "hi": 0.92}
    bad = {"mean": 0.80, "lo": 0.75, "hi": 0.85}
    assert cp.h40_verdict({"a": [good, by_ci], "b": [good, good]}) == {
        "verdict": "holds",
        "at_bar": {"a": True, "b": True},
    }
    assert cp.h40_verdict({"a": [good, bad], "b": [good, good]}) == {
        "verdict": "fails",
        "at_bar": {"a": False, "b": True},
    }


def _ci(mean, lo, hi):
    return {"mean": mean, "lo": lo, "hi": hi}


def test_cell_path_marks_a_later_start_and_keeps_the_old_names():
    assert cp.cell_path("q", 12, "x", "ends", False, "", 300).name == "clsprobe_q_k12_x_ends.jsonl"
    assert cp.cell_path("q", 12, "x", "ends", True, "", 600).name == "clsprobe_q_k12_x_ends_rot_d600.jsonl"
    assert cp.cell_path("q", 0, "", "", False, "emotion", 600).name == "clsprobe_emotion_q_k0_d600.jsonl"


def test_emotion_window_moves_with_the_start():
    assert cp.emotion_window(300) == (0, 300)
    assert cp.emotion_window(600) == (300, 600)
