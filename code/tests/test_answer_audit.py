"""tools/answer_audit.py: damage the kept score does not see."""

from tools import answer_audit as aa


def test_names_and_numbers_skip_sentence_openers():
    assert aa.names_and_numbers("The dog Theia ran. She met Bob at 12:30, in 2014.") == {
        "theia",
        "bob",
        "12:30",
        "2014",
    }


def test_supported_share_counts_names_and_numbers_found_in_the_prompt():
    assert aa.supported_share("The dog Theia met Bob in 2014.", "A stray named Theia, 2014 report.") == 2 / 3


def test_supported_share_is_none_when_the_answer_names_nothing():
    assert aa.supported_share("Business", "any prompt") is None


def test_summary_of_one_task():
    un = {("qa", 0): {"gen_ids": [1, 2], "out": "x"}, ("qa", 1): {"gen_ids": [3], "out": "y"}}
    recs = [
        {"task": "qa", "i": 0, "gen_ids": [1, 2], "out": "x"},
        {"task": "qa", "i": 1, "gen_ids": [7] * 40, "out": "Zed Zed"},
    ]
    prompts = {("qa", 0): "p", ("qa", 1): "p"}
    row = aa.summarize_task(recs, un, prompts)
    assert row["n"] == 2
    assert row["loop"] == 0.5
    assert row["same"] == 0.5
    assert row["lcs"] == 0.5
    assert row["supported"] == 0.0
