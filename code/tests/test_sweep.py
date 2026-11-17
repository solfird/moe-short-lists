"""Queue runner: plans, commands, output paths, skip logic and logging. No subprocess or model is started."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import records
import short
import sweep
from models import MODELS
from sweep import Run

FRACS = (1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2)


def test_plan_c1_size():
    plan = sweep.plan_runs("c1")
    assert len(plan) == 108
    assert len(set(plan)) == 108
    assert sum(r.suite == "short" for r in plan) == 100
    assert sum(r.suite == "gsm8k" for r in plan) == 8


def test_plan_c1_order():
    plan = sweep.plan_runs("c1")
    assert plan[0] == Run("gemma", "short", 0.0, 32)
    assert plan[1:7] == [Run("gemma", "short", f, 32) for f in FRACS]
    assert plan[7].model_tag == "qwen36"
    assert [r.s for r in plan[28:34]] == [16] * 6  # after 4 x 7 runs at S = 32, S = 16 starts
    assert [r.s for r in plan[28 + 24 :][:1]] == [8]
    assert plan[-4:] == [Run("nemotron", "gsm8k", f, 32) for f in (0.0, 1 / 8, 1 / 4, 1 / 2)]


def test_plan_c1_k0_once_per_model():
    k0 = [r for r in sweep.plan_runs("c1") if r.suite == "short" and r.frac == 0]
    assert sorted(r.model_tag for r in k0) == sorted(MODELS)
    assert {r.s for r in k0} == {32}


def test_plan_c2_size():
    plan = sweep.plan_runs("c2")
    assert len(plan) == 22
    assert len(set(plan)) == 22
    reruns = plan[:6]
    assert {r.out_tag for r in reruns} == {"gemma_ids", "qwen_ids"}
    assert {r.model_tag for r in reruns} == {"gemma", "qwen"}
    assert {(r.suite, r.s) for r in reruns} == {("gsm8k", 32)}
    assert {r.suite for r in plan[6:]} == {"c2-gsm8k", "c2-summarize"}


def test_plan_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        sweep.plan_runs("c9")


def test_command_short():
    cmd = sweep.command(Run("qwen36", "short", 0.25, 16))
    assert cmd == [
        sys.executable,
        "short.py",
        "run",
        "--model-tag",
        "qwen36",
        "--n",
        "300",
        "--frac",
        "0.25",
        "--s",
        "16",
    ]


def test_command_gsm8k_uses_budget_and_out_tag():
    cmd = sweep.command(Run("qwen36", "gsm8k", 0.125, 32))
    assert cmd[:2] == [sys.executable, "segroute.py"]
    assert cmd[cmd.index("--k") + 1] == "32"
    assert cmd[cmd.index("--model") + 1] == MODELS["qwen36"]
    assert cmd[cmd.index("--tag") + 1] == "qwen36"
    assert cmd[cmd.index("--n") + 1] == "200"
    cmd = sweep.command(Run("gemma", "gsm8k", 0.0, 32, "gemma_ids"))
    assert cmd[cmd.index("--k") + 1] == "0"
    assert cmd[cmd.index("--tag") + 1] == "gemma_ids"


def test_command_c2():
    cmd = sweep.command(Run("nemotron", "c2-summarize", 0.125, 32))
    assert cmd == [
        sys.executable,
        "c2.py",
        "run",
        "--model-tag",
        "nemotron",
        "--task",
        "summarize",
        "--frac",
        "0.125",
        "--s",
        "32",
    ]


def test_output_path():
    assert sweep.output_path(Run("gemma", "short", 0.0, 16)) == short.run_path("gemma", 300, 0, 32)
    assert sweep.output_path(Run("qwen36", "short", 1 / 8, 16)) == short.run_path("qwen36", 300, 32, 16)
    assert sweep.output_path(Run("qwen36", "gsm8k", 1 / 8, 32)).name == "qwen36_gsm8k_k32_s32.jsonl"
    assert sweep.output_path(Run("gemma", "gsm8k", 0.0, 32, "gemma_ids")).name == "gemma_ids_gsm8k_k0_s32.jsonl"
    assert sweep.output_path(Run("qwen", "c2-gsm8k", 0.125, 32)) == sweep.c2_path("qwen", "gsm8k", 0.125, 32)


def test_c2_path_format():
    assert sweep.c2_path("gemma", "gsm8k", 0.125, 32).name == "gemma_c2_gsm8k_f0.125_s32.jsonl"
    assert sweep.c2_path("gemma", "summarize", 0.25, 32).name == "gemma_c2_summarize_f0.25_s32.jsonl"


def test_expected_items():
    assert sweep.expected_items(Run("gemma", "short", 0.25, 32)) == 1200
    assert sweep.expected_items(Run("gemma", "gsm8k", 0.25, 32)) == 200
    assert sweep.expected_items(Run("gemma", "c2-gsm8k", 0.25, 32)) == 200
    assert sweep.expected_items(Run("gemma", "c2-summarize", 0.25, 32)) == 300


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(short, "RESULTS", tmp_path)
    monkeypatch.setattr(sweep, "SWEEP_LOCK", tmp_path / "sweep.lock")
    return tmp_path


def _write(path: Path, n: int, tail: str = "") -> None:
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(n)) + tail)


def _write_short(path: Path, n: int) -> None:
    """n short-suite records: 300 per task in suite order."""
    tasks = ("summarize", "qa", "classify", "rewrite")
    path.write_text("".join(json.dumps({"task": tasks[j // 300], "i": j % 300}) + "\n" for j in range(n)))


def test_done_run_skipped(results):
    run = Run("gemma", "short", 0.25, 32)
    assert not sweep.is_done(run)
    _write_short(sweep.output_path(run), 1199)
    assert not sweep.is_done(run)
    _write_short(sweep.output_path(run), 1200)
    assert sweep.is_done(run)


def test_partial_final_line_not_counted(results):
    run = Run("gemma", "gsm8k", 0.0, 32)
    _write(sweep.output_path(run), 199, tail='{"i": 19')
    assert not sweep.is_done(run)


def _fake(rcs: dict[int, int] | None = None):
    calls: list[list[str]] = []

    def runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, (rcs or {}).get(len(calls), 0))

    return calls, runner


def test_main_skips_done_runs_and_orders(results):
    plan = sweep.plan_runs("c2")
    _write(sweep.output_path(plan[0]), 200)
    calls, runner = _fake()
    sweep.main(["c2"], runner=runner, log_path=results / "sweep.log")
    assert len(calls) == 21
    assert calls[0][:2] == ["caffeinate", "-i"]
    assert calls[0][2:] == sweep.command(plan[1])
    assert calls[-1][2:] == sweep.command(plan[-1])


def test_main_logs_and_continues_on_failure(results):
    calls, runner = _fake({2: 3})
    log = results / "sweep.log"
    sweep.main(["c2"], runner=runner, log_path=log)
    assert len(calls) == 22
    lines = log.read_text().splitlines()
    assert len(lines) == 44
    assert lines[0].endswith(f"start {sweep.plan_runs('c2')[0]}")
    assert lines[1].endswith(f"end {sweep.plan_runs('c2')[0]} rc=0")
    assert lines[3].endswith("rc=3")
    datetime.fromisoformat(lines[0].split(" ")[0])


def test_corrupt_file_logged_and_queue_continues(results):
    plan = sweep.plan_runs("c2")
    sweep.output_path(plan[0]).write_text(json.dumps({"i": 0}) + "\nnot json\n" + json.dumps({"i": 2}) + "\n")
    _write(sweep.output_path(plan[1]), 200)
    calls, runner = _fake()
    log = results / "sweep.log"
    sweep.main(["c2"], runner=runner, log_path=log)
    assert len(calls) == 21  # plan[0] attempted despite corruption, plan[1] skipped as done
    assert calls[0][2:] == sweep.command(plan[0])
    text = log.read_text().splitlines()
    assert text[0].endswith(f"corrupt {sweep.output_path(plan[0])}")
    datetime.fromisoformat(text[0].split(" ")[0])
    assert text[1].endswith(f"start {plan[0]}")


def test_duplicates_do_not_make_a_run_done(results):
    run = Run("gemma", "short", 0.25, 32)
    recs = [{"task": t, "i": i} for t in ("summarize", "qa", "classify", "rewrite") for i in range(299)]
    recs += recs[:4]  # 1,200 lines, 1,196 distinct items
    sweep.output_path(run).write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert not sweep.is_done(run)
    gsm = Run("gemma", "gsm8k", 0.0, 32)
    sweep.output_path(gsm).write_text("".join(json.dumps({"i": i % 199}) + "\n" for i in range(200)))
    assert not sweep.is_done(gsm)
    c2run = Run("gemma", "c2-gsm8k", 0.125, 32)
    sweep.output_path(c2run).write_text("".join(json.dumps({"i": i % 199}) + "\n" for i in range(200)))
    assert not sweep.is_done(c2run)


def test_short_run_done_on_distinct_task_items(results):
    run = Run("gemma", "short", 0.25, 32)
    recs = [{"task": t, "i": i} for t in ("summarize", "qa", "classify", "rewrite") for i in range(300)]
    sweep.output_path(run).write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert sweep.is_done(run)


def test_main_refuses_while_another_sweep_runs(results):
    calls, runner = _fake()
    with records.exclusive(sweep.SWEEP_LOCK), pytest.raises(SystemExit, match="another sweep is running"):
        sweep.main(["c2"], runner=runner, log_path=results / "sweep.log")
    assert calls == []


def test_commands_use_the_running_interpreter():
    for run in sweep.plan_runs("c1")[:1] + sweep.plan_runs("c2"):
        assert sweep.command(run)[0] == sys.executable


@pytest.mark.parametrize("argv", [[], ["c9"]])
def test_main_rejects_missing_or_unknown_phase(results, argv, capsys):
    calls, runner = _fake()
    with pytest.raises(SystemExit) as exc:
        sweep.main(argv, runner=runner, log_path=results / "sweep.log")
    assert exc.value.code == 2
    assert "c1" in capsys.readouterr().err
    assert calls == []
