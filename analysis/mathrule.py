"""GSM8K under the first paper's selection rule (H41): no training, prompt set from both prompt ends.

mathrule.py run --model-tag gemma --k 24      # 200 items, resumable; --k 0 --n 30 reruns the unmasked baseline
mathrule.py report                            # results/mathrule_h41.json
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import failures
import segroute as sr
import selrule as sl

RESULTS = Path(__file__).resolve().parents[1] / "results"
N, S = 200, 32
GRID = {"gemma": (24, 32, 48), "nemotron": (16, 24, 32), "qwen36": (24, 32, 48)}


def cell_tag(model_tag: str) -> str:
    return f"{model_tag}_ends64"


def run_path(model_tag: str, k: int) -> Path:
    return RESULTS / f"{cell_tag(model_tag)}_gsm8k_k{k}_s{S}.jsonl"


def base_path(model_tag: str) -> Path:
    return RESULTS / f"{model_tag}_gsm8k_k0_s{S}.jsonl"


def kept(arm: list[dict], base: list[dict]) -> dict[str, float]:
    return sl.kept_ci({r["i"]: float(r["ok"]) for r in arm}, {r["i"]: float(r["ok"]) for r in base})


def h41_verdict(first: dict[str, dict[str, float]]) -> str:
    """Holds if GSM8K is at the bar at the first paper's budget on at least 2 of the 3 models."""
    return "holds" if sum(sl.at_bar([d]) for d in first.values()) >= 2 else "fails"


def smallest_at_bar(cells: dict[int, dict[str, float]]) -> int | None:
    return next((k for k in sorted(cells) if sl.at_bar([cells[k]])), None)


def identical_share(rerun: list[dict], stored: list[dict]) -> float:
    """Share of rerun items with the stored prediction and answer length."""
    by_i = {r["i"]: (r["pred"], r["n_gen"]) for r in stored}
    return float(np.mean([by_i[r["i"]] == (r["pred"], r["n_gen"]) for r in rerun]))


def _read(path: Path) -> list[dict]:
    return sr.read_records(path)


def cmd_run(a: argparse.Namespace) -> None:
    from models import MODELS

    orig = sr.install
    sr.install = lambda model: sl.install(model, lambda: sl.TAIL, "ends")
    try:
        sr.cmd_gsm8k(
            argparse.Namespace(
                model=MODELS[a.model_tag],
                tag=cell_tag(a.model_tag),
                k=a.k,
                s=S,
                n=a.n,
                max_tokens=512,
                router=None,
            )
        )
    finally:
        sr.install = orig


def report() -> dict:
    """Write results/mathrule_h41.json."""
    models: dict[str, Any] = {}
    for m, grid in GRID.items():
        base = _read(base_path(m))
        cells: dict[int, dict[str, Any]] = {}
        for k in grid:
            recs = _read(run_path(m, k))
            cells[k] = {
                **kept(recs, base),
                "loops": float(np.mean([failures.is_loop(r["gen_ids"]) for r in recs])),
                "loads_per_layer": float(np.mean([r["loads_per_layer"] for r in recs])),
            }
        models[m] = {
            "unmasked_accuracy": float(np.mean([r["ok"] for r in base])),
            "unmasked_identical": identical_share(_read(run_path(m, 0)), base),
            "cells": cells,
            "smallest_at_bar": smallest_at_bar(cells),
        }
    out = {"verdict": h41_verdict({m: r["cells"][GRID[m][0]] for m, r in models.items()}), "models": models}
    (RESULTS / "mathrule_h41.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True, choices=sorted(GRID))
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--n", type=int, default=N)
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        print(json.dumps(report(), indent=1))


if __name__ == "__main__":
    main()
