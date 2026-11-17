"""C2 disagreement test: first divergence, teacher-forced passes, the run CLI and the report. Slow tests load Gemma."""

import argparse
import gc
import json
from pathlib import Path

import datasets
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm import load

import c2
import records
import segroute
import short
import sweep
from models import MODELS


def test_first_divergence():
    assert c2.first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert c2.first_divergence([1, 2], [1, 2]) is None
    assert c2.first_divergence([1, 2], [1, 2, 3]) == 2
    assert c2.first_divergence([1, 2, 3], [1]) == 1
    assert c2.first_divergence([], []) is None


class Toy:
    """One-layer MoE language model whose router logits pass through a real Controller; no KV state needed."""

    def __init__(self, ctl: segroute.Controller, vocab: int = 6, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.ctl = ctl
        self.router = mx.array(rng.normal(size=(vocab, ctl.n_experts)).astype(np.float32) * 3)
        self.out = mx.array(rng.normal(size=(ctl.n_experts, vocab)).astype(np.float32) * 3)

    def make_cache(self) -> list:
        return []

    def __call__(self, x: mx.array, cache: list | None = None) -> mx.array:
        r = self.ctl(0, self.router[x])
        return mx.softmax(r, axis=-1) @ self.out


PROMPT = [0, 3, 1]


def _toy(k_free: int | None = None, s: int = 2) -> tuple[Toy, list[int], list[int]]:
    """Toy model, its unmasked greedy answer, and its masked free-run answer at k_free."""
    ctl = segroute.Controller(1, 8)
    toy = Toy(ctl)
    ctl.mode = "off"
    base = segroute.generate(toy, object(), PROMPT, 12)
    masked: list[int] = []
    if k_free:
        ctl.reset()
        ctl.mode, ctl.k, ctl.s = "mask", k_free, s
        masked = segroute.generate(toy, object(), PROMPT, 12)
        ctl.mode = "off"
    return toy, base, masked


def test_teacher_forced_full_budget_is_identity():
    toy, base, _ = _toy()
    out = c2.teacher_forced(toy, toy.ctl, PROMPT, base, 8, 2)
    assert out["disagree"].dtype == bool
    assert not out["disagree"].any()
    assert (out["kl"] < 1e-6).all()
    assert out["seg"].tolist() == [t // 2 for t in range(len(base))]
    assert len(out["kl"]) == len(base) == 12


def test_teacher_forced_first_disagreement_is_free_run_divergence():
    toy, base, masked = _toy(k_free=2)
    out = c2.teacher_forced(toy, toy.ctl, PROMPT, base, 2, 2)
    assert out["disagree"].any()
    assert int(np.argmax(out["disagree"])) == c2.first_divergence(base, masked)
    assert (out["kl"] >= -1e-6).all()
    assert out["kl"][out["disagree"]].min() > 0
    assert toy.ctl.mode == "off"
    assert toy.ctl.loads[0] > 2


def test_teacher_forced_resets_controller_and_matches_direct_kl():
    toy, base, _ = _toy()
    clean = c2.teacher_forced(toy, toy.ctl, PROMPT, base, 2, 2)
    toy.ctl.mode, toy.ctl.k, toy.ctl.s = "mask", 2, 2
    segroute.generate(toy, object(), [5, 5, 5, 5], 7)  # leaves stale sets and windows
    dirty = c2.teacher_forced(toy, toy.ctl, PROMPT, base, 2, 2)
    assert dirty["disagree"].tolist() == clean["disagree"].tolist()
    np.testing.assert_allclose(dirty["kl"], clean["kl"], atol=1e-6)
    toy.ctl.reset()
    toy.ctl.mode = "off"
    lu = nn.log_softmax(toy(mx.array(PROMPT)[None])[0, -1], axis=-1)
    toy.ctl.mode, toy.ctl.k, toy.ctl.s = "mask", 2, 2
    lm = nn.log_softmax(toy(mx.array(PROMPT)[None])[0, -1], axis=-1)
    assert clean["kl"][0] == pytest.approx(float((mx.exp(lu) * (lu - lm)).sum()), abs=1e-5)


def test_item_line_rounds_kl_and_lists():
    out = {"disagree": np.array([False, True]), "kl": np.array([0.1234567891, 2.0]), "seg": np.array([0, 0])}
    line = c2.item_line("gsm8k", 4, 17, out)
    assert line == {
        "i": 4,
        "task": "gsm8k",
        "n_prompt": 17,
        "n_gen": 2,
        "disagree": [False, True],
        "kl": [0.123457, 2.0],
        "seg": [0, 0],
    }
    json.dumps(line)


def test_source_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(short, "RESULTS", tmp_path)
    assert c2.source_path("gemma", "gsm8k") == tmp_path / "gemma_ids_gsm8k_k0_s32.jsonl"
    assert c2.source_path("qwen", "gsm8k") == tmp_path / "qwen_ids_gsm8k_k0_s32.jsonl"
    assert c2.source_path("nemotron", "gsm8k") == tmp_path / "nemotron_gsm8k_k0_s32.jsonl"
    assert c2.source_path("qwen36", "summarize") == tmp_path / "qwen36_short_n300_k0_s32.jsonl"


@pytest.mark.data
def test_prompt_texts_match_the_free_runs():
    gsm = c2.prompt_texts("gsm8k")
    assert sorted(gsm) == list(range(200))
    assert gsm[0] == segroute.gsm_prompt(datasets.load_dataset("openai/gsm8k", "main", split="test")[0]["question"])
    summ = c2.prompt_texts("summarize")
    assert sorted(summ) == list(range(300))
    assert summ[5] == next(p for t, i, p, _ in short.items(300) if t == "summarize" and i == 5)


class _Tok:
    def decode(self, ids: list[int]) -> str:
        return " ".join(map(str, ids))


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _wire_run(monkeypatch, tmp_path) -> dict:
    calls: dict = {"tf": [], "loaded": 0}
    monkeypatch.setattr(short, "RESULTS", tmp_path)

    def load(repo, return_config):
        calls["loaded"] += 1
        return object(), _Tok(), {}

    def teacher_forced(model, ctl, prompt_ids, gen_ids, k, s):
        calls["tf"].append((prompt_ids, gen_ids, k, s))
        n = len(gen_ids)
        return {"disagree": np.zeros(n, bool), "kl": np.full(n, 0.5), "seg": np.arange(n) // s}

    monkeypatch.setattr(c2, "load", load)
    monkeypatch.setattr(c2.sr, "install", lambda model: segroute.Controller(2, 128))
    monkeypatch.setattr(c2.sr, "chat", lambda tok, text: [9] * len(text))
    monkeypatch.setattr(c2, "teacher_forced", teacher_forced)
    monkeypatch.setattr(c2, "prompt_texts", lambda task: {i: "p" * (i + 1) for i in range(3)})
    return calls


def _run_args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**({"model_tag": "nemotron", "task": "gsm8k", "frac": 0.125, "s": 32} | kw))


def test_run_missing_source_exits_naming_file(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="nemotron_gsm8k_k0_s32.jsonl"):
        c2.cmd_run(_run_args())
    assert calls["loaded"] == 0


def test_run_source_without_gen_ids_exits(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    _write(tmp_path / "nemotron_gsm8k_k0_s32.jsonl", [{"i": 0, "ok": True}])
    with pytest.raises(SystemExit, match="gen_ids"):
        c2.cmd_run(_run_args())
    assert calls["loaded"] == 0


def test_run_source_without_prompt_ids_len_exits_naming_file(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    _write(tmp_path / "nemotron_gsm8k_k0_s32.jsonl", [{"i": 0, "gen_ids": [1, 2]}])
    with pytest.raises(SystemExit, match="nemotron_gsm8k_k0_s32.jsonl.*prompt_ids_len"):
        c2.cmd_run(_run_args())
    assert calls["loaded"] == 0


def test_run_gsm8k_prompt_length_mismatch_exits(monkeypatch, tmp_path):
    _wire_run(monkeypatch, tmp_path)
    _write(tmp_path / "nemotron_gsm8k_k0_s32.jsonl", [{"i": 0, "gen_ids": [1], "prompt_ids_len": 5}])
    with pytest.raises(SystemExit, match="prompt"):
        c2.cmd_run(_run_args())


def test_run_writes_one_line_per_item_with_budget(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    _write(
        tmp_path / "qwen36_gsm8k_k0_s32.jsonl",
        [{"i": i, "gen_ids": [1] * (i + 2), "prompt_ids_len": i + 1} for i in range(3)],
    )
    c2.cmd_run(_run_args(model_tag="qwen36", frac=0.25))
    recs = short.read_records(sweep.c2_path("qwen36", "gsm8k", 0.25, 32))
    assert [r["i"] for r in recs] == [0, 1, 2]
    assert recs[2] == {
        "i": 2,
        "task": "gsm8k",
        "n_prompt": 3,
        "n_gen": 4,
        "disagree": [False] * 4,
        "kl": [0.5] * 4,
        "seg": [0] * 4,
    }
    assert {k for _, _, k, _ in calls["tf"]} == {64}
    assert calls["tf"][1][:2] == ([9, 9], [1, 1, 1])


def test_run_summarize_uses_short_k0_summaries_only(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    src = [
        {"task": t, "i": i, "gen_ids": [i, 5], "prompt_ids_len": i + 1} for t in ("summarize", "qa") for i in range(3)
    ]
    _write(tmp_path / "gemma_short_n300_k0_s32.jsonl", src)
    c2.cmd_run(_run_args(model_tag="gemma", task="summarize", frac=0.125))
    recs = short.read_records(tmp_path / "gemma_c2_summarize_f0.125_s32.jsonl")
    assert [(r["task"], r["i"]) for r in recs] == [("summarize", i) for i in range(3)]
    assert {k for _, _, k, _ in calls["tf"]} == {16}


def test_run_prompt_length_mismatch_exits(monkeypatch, tmp_path):
    _wire_run(monkeypatch, tmp_path)
    _write(
        tmp_path / "gemma_short_n300_k0_s32.jsonl", [{"task": "summarize", "i": 0, "gen_ids": [1], "prompt_ids_len": 5}]
    )
    with pytest.raises(SystemExit, match="prompt"):
        c2.cmd_run(_run_args(model_tag="gemma", task="summarize"))


def test_run_resume_repairs_tail_and_skips_done(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    _write(
        tmp_path / "nemotron_gsm8k_k0_s32.jsonl",
        [{"i": i, "gen_ids": [1, 2], "prompt_ids_len": i + 1} for i in range(3)],
    )
    out = sweep.c2_path("nemotron", "gsm8k", 0.125, 32)
    out.write_text(json.dumps({"i": 0, "task": "gsm8k"}) + "\n" + '{"i": 1, "ta')
    c2.cmd_run(_run_args())
    assert [p for p, _, _, _ in calls["tf"]] == [[9, 9], [9, 9, 9]]
    assert [json.loads(x)["i"] for x in out.read_text().splitlines()] == [0, 1, 2]


def test_run_refuses_a_second_writer(monkeypatch, tmp_path):
    calls = _wire_run(monkeypatch, tmp_path)
    _write(
        tmp_path / "nemotron_gsm8k_k0_s32.jsonl",
        [{"i": i, "gen_ids": [1, 2], "prompt_ids_len": i + 1} for i in range(3)],
    )
    out = sweep.c2_path("nemotron", "gsm8k", 0.125, 32)
    out.write_text('{"i": 0, "ta')
    with records.exclusive(records.lock_path(out)), pytest.raises(SystemExit, match=f"busy.*{out.name}"):
        c2.cmd_run(_run_args())
    assert calls["tf"] == []
    assert out.read_text() == '{"i": 0, "ta'


def test_run_writes_provenance_beside_the_output(monkeypatch, tmp_path):
    _wire_run(monkeypatch, tmp_path)
    _write(
        tmp_path / "qwen36_gsm8k_k0_s32.jsonl",
        [{"i": i, "gen_ids": [1, 2], "prompt_ids_len": i + 1} for i in range(3)],
    )
    c2.cmd_run(_run_args(model_tag="qwen36", frac=0.25))
    meta = json.loads(records.meta_path(sweep.c2_path("qwen36", "gsm8k", 0.25, 32)).read_text())
    assert meta["model"] == MODELS["qwen36"]
    assert (meta["task"], meta["frac"], meta["k"], meta["s"], meta["n"]) == ("gsm8k", 0.25, 64, 32, 3)
    assert {"snapshot", "mlx", "mlx_lm", "python", "git_commit", "start"} <= meta.keys()


def test_main_parses_the_queue_command(monkeypatch):
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(c2, "cmd_run", seen.append)
    cmd = sweep.command(sweep.Run("qwen36", "c2-summarize", 1 / 8, 32))
    c2.main(cmd[cmd.index("run") :])
    a = seen[0]
    assert (a.model_tag, a.task, a.frac, a.s) == ("qwen36", "summarize", 0.125, 32)


C2_RECS = [
    {"i": 0, "disagree": [False, True, False, False], "seg": [0, 0, 1, 1], "kl": [0.0, 1.0, 0.0, 1.0]},
    {"i": 1, "disagree": [True, True], "seg": [0, 0], "kl": [2.0, 2.0]},
]


def test_cell_stats_teacher_forced_means():
    st = c2.cell_stats("gsm8k", C2_RECS, None, None)
    assert st["items"] == 2
    assert st["d"] == pytest.approx(0.5)
    assert st["kl"] == pytest.approx(1.0)
    assert st["by_seg"][:2] == [pytest.approx(0.75), pytest.approx(0.0)]
    assert st["by_seg"][2:] == [None] * 8
    assert len(st["by_seg"]) == 10
    assert st["fd_median"] is None and st["recovery"] is None


def test_cell_stats_denominators_matched_and_item_weighted():
    recs = [*C2_RECS, {"i": 2, "disagree": [False] * 4 + [True] * 2, "seg": [3, 3, 3, 3, 4, 5], "kl": [0.0] * 6}]
    st = c2.cell_stats("gsm8k", recs, None, None)
    assert (st["items"], st["tokens"]) == (3, 12)
    assert st["d"] == pytest.approx(5 / 12)
    assert st["tokens_matched"] == 10  # segments 0-3 only
    assert st["d_matched"] == pytest.approx(3 / 10)
    assert st["d_item"] == pytest.approx((0.25 + 1.0 + 2 / 6) / 3)


def test_cell_stats_free_run_without_gen_ids_is_missing_not_keyerror():
    base = [{"i": 0, "gen_ids": [1, 2], "ok": True}]
    free = [{"i": 0, "ok": True}]  # an old record from before gen_ids were stored
    for f, b in ((free, base), (base, free)):
        st = c2.cell_stats("gsm8k", C2_RECS, f, b)
        assert st["free_note"] == "missing (no gen_ids)"
        assert st["n_div"] is None and st["fd_median"] is None and st["recovery"] is None
        assert st["d"] == pytest.approx(0.5)
    cell = _cell("gemma", "gsm8k", 1) | {"free": free, "base": base}
    assert "missing (no gen_ids)" in c2.report([cell])


def test_cell_stats_gsm8k_divergence_and_recovery():
    base = [{"i": i, "gen_ids": [1, 2, 3, 4, 5], "ok": True} for i in range(3)]
    free = [
        {"i": 0, "gen_ids": [1, 2, 3, 4, 5], "ok": True},
        {"i": 1, "gen_ids": [1, 2, 3, 9], "ok": True},
        {"i": 2, "gen_ids": [1, 7], "ok": False},
    ]
    st = c2.cell_stats("gsm8k", None, free, base)
    assert st["d"] is None
    assert st["n_div"] == 2
    assert st["n_free"] == 3
    assert st["fd_median"] == 2.0
    assert st["recovery"] == 0.5


def test_cell_stats_summarize_recovery_is_rouge_against_k0():
    base = [{"task": "summarize", "i": i, "gen_ids": [1, 2], "out": "a b c d"} for i in range(2)]
    free = [
        {"task": "summarize", "i": 0, "gen_ids": [1, 2], "out": "a b c d"},
        {"task": "summarize", "i": 1, "gen_ids": [1, 3], "out": "a b x y"},
    ]
    st = c2.cell_stats("summarize", C2_RECS, free, base)
    assert st["n_div"] == 1
    assert st["fd_median"] == 1.0
    assert st["recovery"] == pytest.approx(0.5)


def _cell(model: str, task: str, d_true: int, missing: list[str] | None = None) -> dict:
    recs = [{"i": 0, "disagree": [True] * d_true + [False] * (4 - d_true), "seg": [0] * 4, "kl": [0.1] * 4}]
    return {
        "model": model,
        "task": task,
        "frac": 0.125,
        "k": 16,
        "c2": None if missing else recs,
        "free": None,
        "base": None,
        "missing": missing or [],
    }


def test_report_tables_and_c2_bar():
    cells = [
        _cell("gemma", "gsm8k", 2),
        _cell("gemma", "summarize", 1),
        _cell("qwen", "gsm8k", 1),
        _cell("qwen", "summarize", 1),
        _cell("nemotron", "gsm8k", 0, ["nemotron_c2_gsm8k_f0.125_s32.jsonl"]),
        _cell("nemotron", "summarize", 1),
    ]
    md = c2.report(cells)
    assert md.startswith("# C2 report")
    assert "EOS step is not scored" in md
    assert "| gemma | gsm8k | 0.125 | 16 | 1 | 4 | 0.500 | 0.500 (4) | 0.500 | 0.100 |" in md
    assert "missing: nemotron_c2_gsm8k_f0.125_s32.jsonl" in md
    assert "| gemma | 0.125 | 2.00 | fails | 2.00 | 2.00 |" in md
    assert "| qwen | 0.125 | 1.00 | holds | 1.00 | 1.00 |" in md
    assert "| nemotron | 0.125 | n/a | missing | n/a | n/a |" in md
    assert "| gemma | gsm8k | 0.125 | 0.500 | n/a |" in md


def test_report_matched_ratio_differs_from_pooled():
    long_math = {
        "i": 0,
        "disagree": [False] * 128 + [True] * 128,
        "seg": [t // 32 for t in range(256)],
        "kl": [0.0] * 256,
    }
    summ = {"i": 0, "disagree": [True] + [False] * 3, "seg": [0] * 4, "kl": [0.0] * 4}
    cells = [
        _cell("gemma", "gsm8k", 0) | {"c2": [long_math]},
        _cell("gemma", "summarize", 0) | {"c2": [summ]},
    ]
    md = c2.report(cells)
    assert "| gemma | 0.125 | 2.00 | fails | 0.00 | 2.00 |" in md


def test_cmd_report_reads_files_and_skips_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(short, "RESULTS", tmp_path)
    _write(sweep.c2_path("gemma", "gsm8k", 0.125, 32), C2_RECS)
    _write(tmp_path / "gemma_ids_gsm8k_k0_s32.jsonl", [{"i": 0, "gen_ids": [1, 2], "ok": True}])
    _write(tmp_path / "gemma_ids_gsm8k_k16_s32.jsonl", [{"i": 0, "gen_ids": [1, 3], "ok": False}])
    _write(short.run_path("gemma", 300, 0, 32), [{"task": "qa", "i": 0, "gen_ids": [1], "out": "x"}])
    c2.cmd_report(argparse.Namespace())
    md = (tmp_path / "c2_report.md").read_text()
    assert "| gemma | gsm8k | 0.125 | 16 | 2 | 6 | 0.500 | 0.500 (6) | 0.625 | 1.000 | 1.0 | 1/1 | 0.000 |" in md
    assert "missing: gemma_c2_summarize_f0.125_s32.jsonl, gemma_short_n300_k16_s32.jsonl" in md
    assert "qwen36_c2_gsm8k_f0.25_s32.jsonl" in md
    assert "| qwen36 | gsm8k | 0.25 | 64 |" in md


def test_report_c2_bar_zero_summarize_disagreement():
    md = c2.report([_cell("qwen", "gsm8k", 1), _cell("qwen", "summarize", 0)])
    assert "| qwen | 0.125 | inf | fails | inf | inf |" in md
    md = c2.report([_cell("qwen", "gsm8k", 0), _cell("qwen", "summarize", 0)])
    assert "| qwen | 0.125 | n/a | no disagreement | n/a | n/a |" in md


@pytest.fixture(scope="module")
def gemma():
    """Gemma with the controller, and 2 GSM8K test problems: prompt ids, unmasked and masked answers (128 tokens)."""
    model, tok, config = load(MODELS["gemma"], return_config=True)
    ctl = segroute.install(model)
    probs = []
    for ex in datasets.load_dataset("openai/gsm8k", "main", split="test").select(range(2)):
        ids = segroute.chat(tok, segroute.gsm_prompt(ex["question"]))
        ctl.reset()
        ctl.mode = "off"
        k0 = segroute.generate(model, tok, ids, 128, config)
        masked = {}
        for k in (16, 32):
            ctl.reset()
            ctl.mode, ctl.k, ctl.s = "mask", k, 32
            masked[k] = segroute.generate(model, tok, ids, 128, config)
        ctl.mode = "off"
        probs.append((ids, k0, masked))
    yield model, ctl, probs
    del model, tok, ctl
    mx.clear_cache()
    gc.collect()


@pytest.mark.slow
def test_teacher_forced_identity(gemma):
    model, ctl, probs = gemma
    for ids, k0, _ in probs:
        out = c2.teacher_forced(model, ctl, ids, k0, 128, 32)
        print("K'=128 n_gen", len(k0), "max kl", float(out["kl"].max()))
        assert len(out["disagree"]) == len(k0) > 0
        assert not out["disagree"].any()
        assert (out["kl"] < 1e-6).all()


@pytest.mark.slow
@pytest.mark.parametrize("k", [16, 32])  # Gemma diverges at token 0 at 16 on both problems; 32 diverges later
def test_teacher_forced_matches_free_run(gemma, k):
    model, ctl, probs = gemma
    for ids, k0, masked in probs:
        out = c2.teacher_forced(model, ctl, ids, k0, k, 32)
        first = int(np.argmax(out["disagree"])) if out["disagree"].any() else len(k0)
        fd = c2.first_divergence(k0, masked[k])
        print(f"K'={k} first disagreement", first, "first divergence", fd, "n_gen", len(k0), len(masked[k]))
        assert first == (len(k0) if fd is None else fd)


@pytest.mark.parametrize(
    ("ratio", "verdict"), [(0.0, "holds"), (0.3, "holds"), (1.5, "holds"), (1.7, "open"), (2.0, "fails")]
)
def test_verdict_holds_at_or_below_one_and_a_half(ratio, verdict):
    # amended 2026-10-30: math disagreeing less per token than summaries also supports C2
    assert c2._verdict(ratio) == verdict
