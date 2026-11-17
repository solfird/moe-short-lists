"""mathrule.py: GSM8K under the first paper's selection rule (H41), pure parts."""

import mathrule as mr


def rec(i, ok, pred="1", n_gen=5):
    return {"i": i, "ok": ok, "pred": pred, "n_gen": n_gen}


def test_kept_is_accuracy_over_unmasked_accuracy_on_matched_items():
    base = [rec(0, True), rec(1, True), rec(2, False), rec(3, True)]
    arm = [rec(0, True), rec(1, False), rec(2, False)]
    assert abs(mr.kept(arm, base)["mean"] - 0.5) < 1e-9


def test_h41_holds_on_two_of_three_at_the_first_budget():
    at = {"mean": 0.95, "lo": 0.9, "hi": 1.0}
    off = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    assert mr.h41_verdict({"a": at, "b": at, "c": off}) == "holds"
    assert mr.h41_verdict({"a": at, "b": off, "c": off}) == "fails"


def test_smallest_at_bar_takes_the_lowest_budget_and_none_when_all_miss():
    at = {"mean": 0.95, "lo": 0.9, "hi": 1.0}
    off = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    assert mr.smallest_at_bar({24: off, 32: at, 48: at}) == 32
    assert mr.smallest_at_bar({24: off, 32: off}) is None


def test_baseline_check_compares_prediction_and_length_on_the_rerun_items():
    stored = [rec(0, True), rec(1, True, n_gen=9), rec(2, True)]
    assert mr.identical_share([rec(0, True), rec(1, True, n_gen=9)], stored) == 1.0
    assert mr.identical_share([rec(0, True), rec(1, True, n_gen=8)], stored) == 0.5


def test_paths_name_the_rule_and_keep_the_stored_baseline():
    assert mr.cell_tag("gemma") == "gemma_ends64"
    assert mr.run_path("gemma", 24).name == "gemma_ends64_gsm8k_k24_s32.jsonl"
    assert mr.base_path("qwen36").name == "qwen36_gsm8k_k0_s32.jsonl"
