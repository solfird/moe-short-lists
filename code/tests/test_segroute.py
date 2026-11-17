"""segroute.py gsm8k: the record layout, checked with a stub model and dataset. No model is loaded."""

import argparse
import json

import datasets
import pytest

import records
import segroute

QUESTIONS = [{"question": f"q{i}", "answer": f"work #### {i}"} for i in range(3)]


class _Split(list):
    def select(self, idx: range) -> "_Split":
        return _Split(self[j] for j in idx)


class _Tok:
    def decode(self, ids: list[int]) -> str:
        return " ".join(map(str, ids))


def _wire(monkeypatch, tmp_path) -> list[list[int]]:
    prompts: list[list[int]] = []
    monkeypatch.setattr(segroute, "RESULTS", tmp_path)
    monkeypatch.setattr(segroute, "load", lambda repo, return_config: (object(), _Tok(), {}))
    monkeypatch.setattr(segroute, "install", lambda model: segroute.Controller(2, 128))
    monkeypatch.setattr(segroute, "chat", lambda tok, text: [len(text)])
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: _Split(QUESTIONS))

    def generate(model, tok, ids, max_tokens, config=None):
        prompts.append(ids)
        return [7, 1]

    monkeypatch.setattr(segroute, "generate", generate)
    return prompts


def test_gsm_prompt_text():
    assert segroute.gsm_prompt("Q?") == "Q?\n\nSolve step by step. End with a line 'The answer is N.'"


def test_gsm8k_record_stores_gen_ids(monkeypatch, tmp_path):
    prompts = _wire(monkeypatch, tmp_path)
    a = argparse.Namespace(model="m", tag="t", k=0, s=32, n=3, max_tokens=8, router=None)
    segroute.cmd_gsm8k(a)
    recs = segroute.read_records(tmp_path / "t_gsm8k_k0_s32.jsonl")
    assert [r["gen_ids"] for r in recs] == [[7, 1]] * 3
    assert recs[1]["ok"] is True
    assert prompts[0] == [len(segroute.gsm_prompt("q0"))]
    assert [r["prompt_ids_len"] for r in recs] == [1, 1, 1]


def test_gsm8k_refuses_a_second_writer(monkeypatch, tmp_path):
    prompts = _wire(monkeypatch, tmp_path)
    out = tmp_path / "t_gsm8k_k0_s32.jsonl"
    a = argparse.Namespace(model="m", tag="t", k=0, s=32, n=3, max_tokens=8, router=None)
    with records.exclusive(records.lock_path(out)), pytest.raises(SystemExit, match=f"busy.*{out.name}"):
        segroute.cmd_gsm8k(a)
    assert prompts == []
    assert not out.exists() or out.read_text() == ""


def test_gsm8k_writes_provenance_beside_the_output(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    segroute.cmd_gsm8k(argparse.Namespace(model="m", tag="t", k=16, s=32, n=3, max_tokens=8, router=None))
    meta = json.loads((tmp_path / "t_gsm8k_k16_s32.jsonl.meta.json").read_text())
    assert meta["model"] == "m"
    assert (meta["k"], meta["s"], meta["n"]) == (16, 32, 3)
    assert {"snapshot", "mlx", "mlx_lm", "python", "git_commit", "start"} <= meta.keys()
