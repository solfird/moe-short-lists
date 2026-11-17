"""jury.py: blind model jury on summaries (H44), pure parts."""

import pytest

import jury as jy


def test_parse_verdict_reads_the_three_answers_inside_extra_text():
    text = 'Here: {"supported": "yes", "condensed": "No", "covers": "yes"} done'
    assert jy.parse_verdict(text) == {"supported": True, "condensed": False, "covers": True}


def test_parse_verdict_refuses_missing_or_odd_answers():
    assert jy.parse_verdict('{"supported": "yes", "condensed": "no"}') is None
    assert jy.parse_verdict('{"supported": "maybe", "condensed": "no", "covers": "yes"}') is None
    assert jy.parse_verdict("no json") is None


def test_majority_needs_two_of_three_on_each_question_and_accepts_on_all_three():
    y = {"supported": True, "condensed": True, "covers": True}
    n = {"supported": False, "condensed": True, "covers": True}
    assert jy.majority([y, y, n]) == {"supported": True, "condensed": True, "covers": True, "accepted": True}
    assert jy.majority([y, n, n])["supported"] is False
    assert jy.majority([y, n, n])["accepted"] is False


def test_cost_uses_list_prices_per_token():
    assert abs(jy.call_cost("luna", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}) - 1.40) < 1e-9


def test_shuffle_is_fixed_and_hides_the_cell():
    rows = [{"cell": "a", "i": n, "article": "x", "answer": "y"} for n in range(20)]
    one, two = jy.shuffled(rows), jy.shuffled(rows)
    assert [r["id"] for r in one] == [r["id"] for r in two]
    assert [r["i"] for r in one] != list(range(20))
    assert "gemma" not in jy.juror_prompt({"article": "gemma text is here", "answer": "z"}).split("ARTICLE")[0]


def test_h44_verdict_needs_two_of_three():
    at = {"mean": 0.95, "lo": 0.9, "hi": 1.0}
    off = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    assert jy.h44_verdict({"a": at, "b": at, "c": off}) == "holds"
    assert jy.h44_verdict({"a": at, "b": off, "c": off}) == "fails"


def test_reply_text_is_empty_when_the_router_returns_no_choices():
    assert jy.reply_text({"choices": []}) == ""
    assert jy.reply_text({"choices": [{"message": {"content": None}}]}) == ""
    assert jy.reply_text({"choices": [{"message": {"content": "x"}}]}) == "x"


def test_each_round_has_its_own_files_items_and_cap():
    jy.use_round("b")
    assert jy.JURY.name == "jury_b" and jy.ITEMS.parent.name == "jury_b"
    assert jy.START == 0 and jy.CELLS == {"gemma": 32, "nemotron": 32, "qwen36": 64} and jy.CAP_USD == 1.5
    jy.use_round("a")
    assert jy.JURY.name == "jury" and jy.START == 300 and jy.CELLS["gemma"] == 24 and jy.CAP_USD == 2.0


def test_votes_read_the_chosen_claude_file(tmp_path, monkeypatch):
    monkeypatch.setattr(jy, "JURY", tmp_path)
    yes = {"supported": True, "condensed": True, "covers": True}
    for name in ("luna", "flash", "claude"):
        (tmp_path / f"{name}.jsonl").write_text(
            '{"id": 1, "verdict": {"supported": false, "condensed": true, "covers": true}}\n'
        )
    (tmp_path / "claude_blind.jsonl").write_text('{"id": 1, "verdict": ' + __import__("json").dumps(yes) + "}\n")
    monkeypatch.setattr(jy, "CLAUDE", "claude_blind")
    assert jy._votes()["claude"][1] == yes
    assert jy.out_name("jury_h44.json") == "jury_h44_blind.json"
    monkeypatch.setattr(jy, "CLAUDE", "claude")
    assert jy._votes()["claude"][1]["supported"] is False
    assert jy.out_name("jury_h44.json") == "jury_h44.json"


def test_key_is_read_from_the_environment_only(monkeypatch):
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        jy._key()
    monkeypatch.setenv("REQUESTY_API_KEY", " abc ")
    assert jy._key() == "abc"
