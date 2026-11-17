"""tools/collect.py: flatten scored runs into one long CSV for analysis and plots."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import tools.collect
from tools.collect import FIELDS, gsm8k_rows, parse_run, short_rows, wilson, write_csv


def ci(mean, lo, hi):
    return {"mean": mean, "lo": lo, "hi": hi}


def scores():
    return {
        "k0_s32": {"qa_f1": ci(0.8, 0.7, 0.9), "loads_per_layer": None, "n": 1200, "kept": None},
        "k16_s32": {
            "qa_f1": ci(0.4, 0.3, 0.5),
            "loads_per_layer": 27.5,
            "n": 1200,
            "kept": {"qa_f1": ci(0.5, 0.4, 0.6)},
        },
    }


def test_parse_run_reads_k_and_s():
    assert parse_run("k16_s32") == (16, 32)
    assert parse_run("k0_s128") == (0, 128)


def test_parse_run_rejects_other_names():
    with pytest.raises(ValueError):
        parse_run("gemma_k16")


def test_short_rows_one_row_per_run_and_metric(tmp_path):
    (tmp_path / "gemma_short_n300_k16_s32.jsonl.meta.json").write_text(
        json.dumps({"model": "m", "git_commit": "abc", "git_dirty": True, "start": "2026-10-27T12:00:00+01:00"})
    )
    rows = short_rows("gemma", 300, scores(), tmp_path)
    assert len(rows) == 2
    base, masked = sorted(rows, key=lambda r: r["k"])
    assert base["kept"] == "" and base["frac"] == 0.0 and base["git_commit"] == ""
    assert masked == {
        "model_tag": "gemma",
        "n_experts": 128,
        "suite": "short",
        "metric": "qa_f1",
        "k": 16,
        "frac": 0.125,
        "s": 32,
        "n_items": 300,
        "n_records": 1200,
        "value": 0.4,
        "lo": 0.3,
        "hi": 0.5,
        "kept": 0.5,
        "kept_lo": 0.4,
        "kept_hi": 0.6,
        "loads_per_layer": 27.5,
        "raw_file": "gemma_short_n300_k16_s32.jsonl",
        "model": "m",
        "git_commit": "abc",
        "git_dirty": True,
        "start": "2026-10-27T12:00:00+01:00",
    }


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson(94, 100)
    assert 0.87 < lo < 0.94 < hi < 0.98
    assert wilson(0, 10)[0] == 0.0


def test_gsm8k_rows_accuracy_from_ok_flags(tmp_path):
    path = tmp_path / "qwen36_gsm8k_k32_s32.jsonl"
    path.write_text("\n".join(json.dumps({"i": i, "ok": i < 3, "loads_per_layer": 10.0}) for i in range(4)) + "\n")
    (row,) = gsm8k_rows(tmp_path)
    assert (row["model_tag"], row["k"], row["s"], row["frac"], row["n_items"]) == ("qwen36", 32, 32, 0.125, 4)
    assert row["value"] == 0.75 and row["metric"] == "gsm8k_acc" and row["loads_per_layer"] == 10.0
    assert row["lo"] < 0.75 < row["hi"]


def test_gsm8k_rows_skip_unknown_tags_for_n_experts(tmp_path):
    (tmp_path / "refactor3_gsm8k_k16_s32.jsonl").write_text(json.dumps({"i": 0, "ok": True}) + "\n")
    (row,) = gsm8k_rows(tmp_path)
    assert row["n_experts"] == "" and row["frac"] == ""


def test_write_csv_round_trips_header_and_rows(tmp_path):
    rows = short_rows("gemma", 300, scores(), tmp_path)
    out = tmp_path / "summary.csv"
    write_csv(rows, out)
    with open(out) as f:
        back = list(csv.DictReader(f))
    assert list(back[0]) == FIELDS
    assert len(back) == 2


def test_module_entry_point_runs_from_the_repo_root():
    root = Path(tools.collect.__file__).resolve().parent.parent
    done = subprocess.run(
        [sys.executable, "-m", "tools.collect", "--help"], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
