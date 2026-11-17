"""Resumable queue for the phase 1 measurement runs: one subprocess per run, complete runs skipped, every run logged."""

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import short
from models import MODELS, N_EXPERTS
from records import exclusive

CODE = Path(__file__).resolve().parent
HERE = CODE.parent
LOG = HERE / "results" / "logs" / "sweep.log"
SWEEP_LOCK = HERE / "results" / "logs" / "sweep.lock"
PYTHON = sys.executable  # the interpreter running the sweep, so every run uses the same venv
FRACS = (1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2)
S_ORDER = (16, 8, 64)
SHORT_N = 300
GSM_N = 200
C2_FRACS = (1 / 8, 1 / 4)
C2_ITEMS = {"gsm8k": GSM_N, "summarize": 300}
IDS_TAGS = {"gemma": "gemma_ids", "qwen": "qwen_ids"}


class Run(NamedTuple):
    """One measurement; frac 0 is unmasked and out_tag renames a gsm8k output file."""

    model_tag: str
    suite: str
    frac: float
    s: int
    out_tag: str = ""


def _k(run: Run) -> int:
    """Experts kept per layer for the run; 0 means unmasked."""
    return short.budget(N_EXPERTS[run.model_tag], run.frac) if run.frac else 0


def c2_path(tag: str, task: str, frac: float, s: int) -> Path:
    """Raw output file of a c2 run; the fraction is written with the general format, so 0.125 gives f0.125."""
    return short.RESULTS / f"{tag}_c2_{task}_f{frac:g}_s{s}.jsonl"


def plan_runs(phase: str) -> list[Run]:
    """Ordered runs of a phase: c1 (short suite then GSM8K) or c2 (id-storing GSM8K reruns then c2 runs)."""
    if phase == "c1":
        plan = [Run(m, "short", f, 32) for m in MODELS for f in (0.0, *FRACS)]
        plan += [Run(m, "short", f, s) for s in S_ORDER for m in MODELS for f in FRACS]
        plan += [Run(m, "gsm8k", f, 32) for m in ("qwen36", "nemotron") for f in (0.0, *C2_FRACS, 1 / 2)]
        return plan
    if phase == "c2":
        plan = [Run(m, "gsm8k", f, 32, tag) for m, tag in IDS_TAGS.items() for f in (0.0, *C2_FRACS)]
        plan += [Run(m, f"c2-{t}", f, 32) for m in MODELS for t in C2_ITEMS for f in C2_FRACS]
        return plan
    raise ValueError(f"unknown phase {phase!r}, expected c1 or c2")


def command(run: Run) -> list[str]:
    """Command line that performs the run."""
    frac = f"{run.frac:g}"
    if run.suite == "short":
        return [
            PYTHON,
            "short.py",
            "run",
            "--model-tag",
            run.model_tag,
            "--n",
            str(SHORT_N),
            "--frac",
            frac,
            "--s",
            str(run.s),
        ]
    if run.suite == "gsm8k":
        tag = run.out_tag or run.model_tag
        return [
            PYTHON,
            "segroute.py",
            "gsm8k",
            "--model",
            MODELS[run.model_tag],
            "--tag",
            tag,
            "--k",
            str(_k(run)),
            "--s",
            str(run.s),
            "--n",
            str(GSM_N),
        ]
    task = run.suite.removeprefix("c2-")
    return [PYTHON, "c2.py", "run", "--model-tag", run.model_tag, "--task", task, "--frac", frac, "--s", str(run.s)]


def output_path(run: Run) -> Path:
    """Raw result file of the run."""
    if run.suite == "short":
        return short.run_path(run.model_tag, SHORT_N, _k(run), run.s)
    if run.suite == "gsm8k":
        tag = run.out_tag or run.model_tag
        return short.RESULTS / f"{tag}_gsm8k_k{_k(run)}_s{run.s}.jsonl"
    return c2_path(run.model_tag, run.suite.removeprefix("c2-"), run.frac, run.s)


def expected_items(run: Run) -> int:
    """Distinct items a complete output file holds."""
    if run.suite == "short":
        return 4 * SHORT_N
    if run.suite == "gsm8k":
        return GSM_N
    return C2_ITEMS[run.suite.removeprefix("c2-")]


def is_done(run: Run) -> bool:
    """True when the output file holds every item of the run, counted as distinct keys so duplicates mask nothing."""
    path = output_path(run)
    if not path.exists():
        return False
    recs = short.read_records(path)
    keys = {(r["task"], r["i"]) for r in recs} if run.suite == "short" else {r["i"] for r in recs}
    return len(keys) >= expected_items(run)


def main(
    argv: list[str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    log_path: Path = LOG,
) -> None:
    """Run a phase in order, skipping complete runs; a failed run is logged and the queue goes on."""
    p = argparse.ArgumentParser(description="Run the phase 1 queue, skipping complete runs.")
    p.add_argument("phase", choices=["c1", "c2"])
    phase = p.parse_args(argv).phase

    def log(text: str) -> None:
        with log_path.open("a") as f:
            f.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {text}\n")

    with exclusive(SWEEP_LOCK, "another sweep is running: " + str(SWEEP_LOCK) + " is held"):
        for run in plan_runs(phase):
            try:
                if is_done(run):
                    continue
            except json.JSONDecodeError:
                log(f"corrupt {output_path(run)}")
            log(f"start {run}")
            code = runner(["caffeinate", "-i", *command(run)], cwd=CODE).returncode
            log(f"end {run} rc={code}")


if __name__ == "__main__":
    main()
