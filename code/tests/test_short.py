"""Short suite: budgets, bootstrap CIs, resumable runs and the legacy scorer. No model is loaded."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import records
import segroute
import short

RESULTS = Path(short.__file__).resolve().parents[1] / "results"


def _rec(i: int) -> dict:
    return {"task": "qa", "i": i, "out": "x", "ref": ["x"]}


def test_budget_from_fraction():
    assert short.budget(128, 0.25) == 32
    assert short.budget(256, 0.25) == 64
    assert short.budget(128, 1 / 16) == 8
    assert short.budget(256, 1.0) == 256
    for e in (128, 256):
        for frac in (0.001, 1 / 16, 0.3, 0.5, 1.0):
            assert 1 <= short.budget(e, frac) <= e


def test_bootstrap_ci_brackets_mean():
    lo, hi = short.bootstrap_ci([0.0, 1.0] * 150)
    assert lo < 0.5 < hi
    assert hi - lo < 0.15
    assert short.bootstrap_ci([0.0, 1.0] * 150) == (lo, hi)


def test_run_path_names():
    assert short.run_path("gemma", 300, 32, 32) == RESULTS / "gemma_short_n300_k32_s32.jsonl"


def test_unmasked_run_ignores_s():
    assert short.run_path("gemma", 300, 0, 8) == RESULTS / "gemma_short_n300_k0_s32.jsonl"


def test_read_records_skips_truncated_last_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps(_rec(i)) + "\n" for i in range(3)) + '{"task": "qa", "i": 3, "ou')
    assert [r["i"] for r in short.read_records(p)] == [0, 1, 2]


def test_read_records_rejects_corrupt_middle_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_rec(0)) + "\nnot json\n" + json.dumps(_rec(2)) + "\n")
    with pytest.raises(json.JSONDecodeError):
        short.read_records(p)


def test_resume_ignores_partial_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps(_rec(i)) + "\n" for i in range(3)) + '{"task": "qa", "i": 3, "ou')
    todo = [("qa", i, f"p{i}", ["x"]) for i in range(5)]
    calls: list[int] = []

    def step(task: str, i: int, prompt: str, ref) -> dict:
        calls.append(i)
        return _rec(i)

    short.run_suite(p, todo, step)
    assert calls == [3, 4]
    assert [r["i"] for r in short.read_records(p)] == [0, 1, 2, 3, 4]


def test_item_record_stores_token_ids():
    rec = short.item_record("qa", 7, "ref", [1, 2, 3], [9, 8], lambda ids: "nine eight")
    assert rec["gen_ids"] == [9, 8]
    assert rec["prompt_ids_len"] == 3
    assert rec["out"] == "nine eight"
    assert rec["n_gen"] == 2


@pytest.mark.data
def test_items_count_per_task():
    its = short.items(5)
    for task in ("summarize", "qa", "classify", "rewrite"):
        assert sum(1 for t, *_ in its if t == task) == 5


def test_article_slices_disjoint():
    sl = short.article_slices(300)
    assert set(sl["summarize"]).isdisjoint(sl["rewrite"])
    assert sl["rewrite"].start == 1000


def test_legacy_rescore_matches_stored_scores():
    """The refactored scorer reproduces results/gemma_short_scores.json (n = 100 files) to 1e-9, read-only."""
    stored = json.loads((RESULTS / "gemma_short_scores.json").read_text())
    teacher = short.teacher_outputs(short.read_records(RESULTS / "gemma_short_k0_s32.jsonl"))
    for row in stored:
        if not (RESULTS / f"{row['run']}.jsonl").exists():  # a row whose raw file is not stored here is skipped
            continue
        new = short.score_run(row["run"], short.read_records(RESULTS / f"{row['run']}.jsonl"), teacher)
        for key, val in row.items():
            if isinstance(val, float):
                assert new[key] == pytest.approx(val, abs=1e-9), (row["run"], key)
            else:
                assert new[key] == val, (row["run"], key)


def test_ci_scores_have_mean_lo_hi_and_kept():
    def recs(correct: int) -> list[dict]:
        return [
            {"task": "classify", "i": i, "out": "World" if i < correct else "Sports", "ref": "World"} for i in range(10)
        ]

    teacher = short.teacher_outputs(recs(10))
    out = short.ci_scores({"k0": recs(10), "k8": recs(5)}, teacher)
    assert out["k8"]["cls_acc"]["mean"] == 0.5
    assert out["k8"]["cls_acc"]["lo"] <= 0.5 <= out["k8"]["cls_acc"]["hi"]
    assert out["k0"]["cls_acc"]["mean"] == 1.0
    assert out["k8"]["kept"]["cls_acc"]["mean"] == 0.5
    assert "kept" not in out["k0"]


def _cls(i: int, ok: bool) -> dict:
    return {"task": "classify", "i": i, "out": "World" if ok else "Sports", "ref": "World"}


def test_kept_paired_ci():
    teacher: dict = {}
    # masked = 0.5 x k0 on every item: degenerate interval at 0.5
    k0 = [_cls(i, True) for i in range(20)]
    masked = [_cls(i, i % 2 == 0) for i in range(20)]
    kept = short.ci_scores({"k0_s32": k0, "k8_s32": masked}, teacher)["k8_s32"]["kept"]["cls_acc"]
    assert kept["mean"] == 0.5
    assert kept["lo"] <= 0.5 <= kept["hi"]
    exact = short.paired_kept_ci({("c", i): 0.5 for i in range(20)}, {("c", i): 1.0 for i in range(20)})
    assert exact == {"mean": 0.5, "lo": 0.5, "hi": 0.5}
    noisy = short.paired_kept_ci({("c", i): float(i % 2) for i in range(20)}, {("c", i): 1.0 for i in range(20)})
    assert noisy["lo"] < noisy["mean"] < noisy["hi"]


def test_kept_skips_metric_missing_from_baseline():
    teacher: dict = {}
    k0 = [{"task": "qa", "i": i, "out": "a", "ref": ["a"]} for i in range(5)]
    masked = [_cls(i, True) for i in range(5)]
    entry = short.ci_scores({"k0_s32": k0, "k8_s32": masked}, teacher)["k8_s32"]
    assert "cls_acc" in entry
    assert "cls_acc" not in entry["kept"]


class _Tok:
    def decode(self, ids: list[int]) -> str:
        return " ".join(map(str, ids))


def _wire_run(monkeypatch, tmp_path):
    calls: dict = {"configs": [], "ctl": None}
    config = {"eos_token_id": 1}
    monkeypatch.setattr(short, "RESULTS", tmp_path)
    monkeypatch.setattr(short, "load", lambda repo, return_config: (object(), _Tok(), config))

    def install(model):
        calls["ctl"] = segroute.Controller(2, 128)
        return calls["ctl"]

    def generate(model, tok, ids, max_tokens, cfg=None):
        calls["configs"].append(cfg)
        calls["ctl"].loads = [3, 5]
        calls["ctl"].reselects = [1, 1]
        calls["ctl"].touched = [{1, 2, 3}, {4, 5, 6, 7, 8}]
        return [7, 8]

    monkeypatch.setattr(short.sr, "install", install)
    monkeypatch.setattr(short.sr, "chat", lambda tok, text: [1, 2, 3])
    monkeypatch.setattr(short.sr, "generate", generate)
    its = [("classify", i, f"p{i}", "World") for i in range(3)]
    monkeypatch.setattr(short, "items", lambda n: its)
    return calls, config


def _args(**kw) -> argparse.Namespace:
    base = {"model_tag": "gemma", "tag": None, "n": 3, "frac": 0.25, "s": 32, "router": None}
    return argparse.Namespace(**(base | kw))


def test_run_unmasked_writes_k0_s32(monkeypatch, tmp_path):
    calls, _ = _wire_run(monkeypatch, tmp_path)
    short.cmd_run(_args(frac=0, s=16))
    assert calls["ctl"].mode == "off"
    recs = short.read_records(tmp_path / "gemma_short_n3_k0_s32.jsonl")
    assert len(recs) == 3
    assert "loads_per_layer" not in recs[0]
    assert "touched_per_layer" not in recs[0]


def test_run_masked_sets_controller_and_records(monkeypatch, tmp_path):
    calls, config = _wire_run(monkeypatch, tmp_path)
    short.cmd_run(_args(frac=0.25, s=16))
    ctl = calls["ctl"]
    assert (ctl.mode, ctl.k, ctl.s) == ("mask", 32, 16)
    rec = short.read_records(tmp_path / "gemma_short_n3_k32_s16.jsonl")[0]
    assert rec["gen_ids"] == [7, 8]
    assert rec["prompt_ids_len"] == 3
    assert rec["loads_per_layer"] == 4.0
    assert rec["touched_per_layer"] == 4.0
    assert rec["reselects"] == 1
    assert all(c is config for c in calls["configs"])


def test_run_resume_skips_done_items(monkeypatch, tmp_path):
    calls, _ = _wire_run(monkeypatch, tmp_path)
    short.cmd_run(_args())
    n_first = len(calls["configs"])
    short.cmd_run(_args())
    assert len(calls["configs"]) == n_first
    assert len(short.read_records(tmp_path / "gemma_short_n3_k32_s32.jsonl")) == 3


def _write_run(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_score_n_writes_named_file(monkeypatch, tmp_path):
    monkeypatch.setattr(short, "RESULTS", tmp_path)
    _write_run(tmp_path / "gemma_short_n4_k0_s32.jsonl", [_cls(i, True) for i in range(4)])
    _write_run(tmp_path / "gemma_short_n4_k8_s32.jsonl", [_cls(i, i < 2) for i in range(4)])
    short.cmd_score(argparse.Namespace(tag="gemma", n=4))
    out = json.loads((tmp_path / "gemma_short_n4_scores.json").read_text())
    assert out["k8_s32"]["cls_acc"]["mean"] == 0.5
    assert out["k8_s32"]["kept"]["cls_acc"]["mean"] == 0.5


def test_score_n_without_k0_exits_naming_file(monkeypatch, tmp_path):
    monkeypatch.setattr(short, "RESULTS", tmp_path)
    _write_run(tmp_path / "gemma_short_n4_k8_s32.jsonl", [_cls(i, True) for i in range(4)])
    with pytest.raises(SystemExit, match="gemma_short_n4_k0_s32.jsonl"):
        short.cmd_score(argparse.Namespace(tag="gemma", n=4))


def test_resume_torn_line_file_is_valid_after_append(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps(_rec(i)) + "\n" for i in range(3)) + '{"task": "qa", "i": 3, "ou')
    short.run_suite(p, [("qa", i, f"p{i}", ["x"]) for i in range(4)], lambda task, i, prompt, ref: _rec(i))
    assert [json.loads(x)["i"] for x in p.read_text().splitlines()] == [0, 1, 2, 3]


def test_run_suite_refuses_a_second_writer(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_rec(0)) + "\n" + '{"task": "qa", "i": 1, "ou')
    before = p.read_bytes()
    calls: list[int] = []

    def step(task: str, i: int, prompt: str, ref) -> dict:
        calls.append(i)
        return _rec(i)

    with records.exclusive(records.lock_path(p)), pytest.raises(SystemExit, match=f"busy.*{p.name}"):  # 1st writer
        short.run_suite(p, [("qa", i, f"p{i}", ["x"]) for i in range(3)], step)
    assert calls == []
    assert p.read_bytes() == before


def test_run_writes_provenance_beside_the_output(monkeypatch, tmp_path):
    _wire_run(monkeypatch, tmp_path)
    short.cmd_run(_args(frac=0.25, s=16))
    meta = json.loads((tmp_path / "gemma_short_n3_k32_s16.jsonl.meta.json").read_text())
    assert meta["model"] == short.MODELS["gemma"]
    assert (meta["frac"], meta["k"], meta["s"], meta["n"]) == (0.25, 32, 16, 3)
    assert {"snapshot", "mlx", "mlx_lm", "python", "git_commit", "start"} <= meta.keys()


def test_kept_ci_finite_when_resamples_hit_a_zero_baseline():
    base = {("c", i): float(i == 0) for i in range(20)}  # one nonzero item: many resamples have a zero mean
    masked = {("c", i): float(i == 0) for i in range(20)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = short.paired_kept_ci(masked, base)
    assert kept["mean"] == 1.0
    assert np.isfinite(kept["lo"]) and np.isfinite(kept["hi"])
    assert kept["lo"] == kept["hi"] == 1.0


def test_kept_ci_without_matched_items_raises_clearly():
    with pytest.raises(ValueError, match="no matched items"):
        short.paired_kept_ci({("a", 0): 1.0}, {("b", 0): 1.0})


def test_article_slices_from_a_start_are_disjoint_from_the_first_set():
    first, second = short.article_slices(300), short.article_slices(300, start=300)
    assert second["summarize"] == range(300, 600)
    assert second["rewrite"] == range(1300, 1600)
    for task in first:
        assert set(first[task]).isdisjoint(second[task])
    assert set(second["summarize"]).isdisjoint(second["rewrite"])


def test_items_from_a_start_share_no_prompt_with_the_first_set():
    first, second = short.items(5), short.items(5, start=5)
    assert not {x[2] for x in first} & {x[2] for x in second}
    assert sorted({x[1] for x in second}) == [5, 6, 7, 8, 9]
