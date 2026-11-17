"""jury_c.py: the two-juror panel on items 300 to 399, pure parts."""

import jury_c as jc

YES = {"supported": True, "condensed": True, "covers": True}
NO = {"supported": False, "condensed": True, "covers": True}


def test_accepted_needs_both_jurors_yes_on_every_question():
    assert jc.accepted_by_both(YES, YES) is True
    assert jc.accepted_by_both(YES, NO) is False


def test_h52_smallest_budget_at_the_bar():
    at = {"mean": 0.95, "lo": 0.9, "hi": 1.0}
    under = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    assert jc.smallest_at_bar({"48": under, "64": at}) == 64
    assert jc.smallest_at_bar({"48": under, "64": under}) is None
    assert jc.smallest_at_bar({"64": at, "128": at}) == 64


def test_panel_cells_cover_every_arm():
    cells = jc.panel_cells()
    assert ("gemma", 0) in cells and ("gemma", 24) in cells and ("gemma", 64) in cells
    assert ("qwen36", 128) in cells and ("nemotron", 16) in cells
    assert len(cells) == 12


def test_agreement_with_the_three_juror_majority():
    assert jc.agreement({1: True, 2: False}, {1: True, 2: True, 3: False}) == 0.5
