import final

YES = {"supported": True, "condensed": True, "covers": True}
NO = {"supported": False, "condensed": True, "covers": True}


def test_summary_prompt_is_the_suite_prompt_with_the_text_cut_to_3000():
    p = final.summary_prompt("x" * 5000)
    assert p.startswith("Summarize this news article in 2 or 3 sentences.\n\n")
    assert len(p.split("\n\n", 1)[1]) == 3000


def test_cells_are_full_model_first_budget_and_recovery_budget():
    assert final.cells() == [
        ("gemma", 0),
        ("gemma", 24),
        ("gemma", 48),
        ("nemotron", 0),
        ("nemotron", 16),
        ("nemotron", 48),
        ("qwen36", 0),
        ("qwen36", 24),
        ("qwen36", 128),
    ]


def test_summary_file_names_carry_dataset_start_and_budget():
    assert final.sum_path("xsum", "gemma", 0, 48).name == "sum_xsum_gemma_d0_k48.jsonl"


def test_flags_accept_only_when_both_jurors_say_yes_and_skip_missing_verdicts():
    items = [
        {"id": 1, "cell": "gemma_k0", "i": 600},
        {"id": 2, "cell": "gemma_k0", "i": 601},
        {"id": 3, "cell": "gemma_k0", "i": 602},
    ]
    claude = {1: YES, 2: YES, 3: YES}
    local = {1: YES, 2: NO}
    assert final.cell_flags(items, claude, local) == {"gemma_k0": {600: True, 601: False}}
