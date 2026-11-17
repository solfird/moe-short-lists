import whyreport as wr


def ci(mean: float, lo: float | None = None, hi: float | None = None) -> dict[str, float]:
    return {"mean": mean, "lo": mean - 0.05 if lo is None else lo, "hi": mean + 0.05 if hi is None else hi}


def test_h47_margin_is_mean_gap_over_layers():
    rows = [{"same_token": 0.30, "other_token_floor": 0.05}, {"same_token": 0.10, "other_token_floor": 0.05}]
    assert abs(wr.h47_margin(rows) - 0.15) < 1e-9


def test_h47_holds_only_when_every_model_clears_the_margin():
    big = [{"same_token": 0.5, "other_token_floor": 0.1}]
    small = [{"same_token": 0.12, "other_token_floor": 0.1}]
    assert wr.h47_verdict({"a": big, "b": big, "c": big}) == "holds"
    assert wr.h47_verdict({"a": big, "b": big, "c": small}) == "fails"


def test_h48_holds_on_two_of_three_token_function_models():
    fn = {"across_tasks_same_type": 0.5, "across_types_same_task": 0.3}
    subj = {"across_tasks_same_type": 0.2, "across_types_same_task": 0.4}
    assert wr.h48_verdict({"a": fn, "b": fn, "c": subj}) == "holds"
    assert wr.h48_verdict({"a": fn, "b": subj, "c": subj}) == "fails"


def test_h45_needs_qa_and_classify_at_bar_on_two_models():
    ok = {"qa_f1": ci(0.95), "cls_acc": ci(0.92)}
    qa_low = {"qa_f1": ci(0.60), "cls_acc": ci(0.99)}
    assert wr.h45_verdict({"a": ok, "b": ok, "c": qa_low}) == "holds"
    assert wr.h45_verdict({"a": ok, "b": qa_low, "c": qa_low}) == "fails"


def test_h46_reads_short_against_longest_valid_arm():
    acc = {"short": 0.99, "mid": 0.95, "long": 0.40}
    assert wr.longest_valid(acc) == "mid"
    assert wr.longest_valid({"short": 0.99, "mid": 0.5, "long": 0.3}) is None
    assert wr.longest_valid({"short": 0.99, "mid": 0.95, "long": 0.85}) == "long"


def test_h43_needs_masked_rise_unmasked_flat_and_gate():
    assert wr.h43_holds(masked_rise=0.15, unmasked_rise=0.01, gate=True)
    assert not wr.h43_holds(masked_rise=0.15, unmasked_rise=0.03, gate=True)
    assert not wr.h43_holds(masked_rise=0.08, unmasked_rise=0.0, gate=True)
    assert not wr.h43_holds(masked_rise=0.15, unmasked_rise=0.0, gate=False)


def test_t3_holds_when_two_paired_cis_are_above_zero():
    up, flat = ci(0.05, 0.01, 0.09), ci(0.01, -0.02, 0.04)
    assert wr.t3_verdict([up, up, flat]) == "holds"
    assert wr.t3_verdict([up, flat, flat]) == "fails"


def test_h55_holds_when_the_prompt_rule_beats_the_fixed_set_on_two_tasks_on_two_models():
    win, tie = {"lo": 0.01}, {"lo": -0.02}
    two = [win, win, tie]
    one = [win, tie, tie]
    assert wr.h55_verdict({"a": two, "b": two, "c": one}) == "holds"
    assert wr.h55_verdict({"a": two, "b": one, "c": one}) == "fails"


def test_h56_is_specific_only_when_the_routed_rise_is_under_half_on_every_measure_that_held():
    always = {"loops": 0.30, "at_cap": 0.56}
    assert wr.h56_specific({"loops": 0.10, "at_cap": 0.20}, always, ["loops", "at_cap"])
    assert not wr.h56_specific({"loops": 0.10, "at_cap": 0.30}, always, ["loops", "at_cap"])
    assert wr.h56_specific({"loops": 0.10, "at_cap": 0.30}, always, ["loops"])


def test_h60_holds_when_the_prompt_rule_beats_gemmas_fixed_list_on_two_tasks():
    win, tie = {"lo": 0.01}, {"lo": -0.02}
    assert wr.h60_verdict([win, win, tie]) == "holds"
    assert wr.h60_verdict([win, tie, tie]) == "fails"
