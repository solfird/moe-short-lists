"""H65: the both-ends rule against the whole-prompt rule on the held-out items, at the final budgets.

heldrule.py report   # results/heldrule_h65.json, from stored runs (selrule.py run --arm whole --start 600)
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import json
from pathlib import Path
from typing import Any

import numpy as np

import promptlen as pl
import selrule as sl
import short

OUT = Path(__file__).resolve().parents[1] / "results" / "heldrule_h65.json"
START = 600
BUDGETS = {"gemma": 24, "nemotron": 16, "qwen36": 24}
CLAUSE_METRICS = ("qa_f1", "cls_acc")


def model_holds(diffs: dict[str, dict[str, float]]) -> bool:
    """Both ends minus whole prompt: above 0 with its CI on one task at least, and on no task below 0 with its CI."""
    return any(d["lo"] > 0 for d in diffs.values()) and not any(d["hi"] < 0 for d in diffs.values())


def verdict(models: dict[str, bool]) -> str:
    n = sum(models.values())
    return "holds" if n == len(models) else "holds on two of three" if n == 2 else "fails"


def report(n: int = 300) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, k in BUDGETS.items():
        whole, ends = sl.tag(model, sl.WHOLE_ARM, START), sl.tag(model, sl.ENDS_ARM, START)
        base_recs = short.read_records(short.run_path(whole, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        recs = {
            arm: short.read_records(short.run_path(name, n, k, 32))
            for arm, name in (("whole", whole), ("ends64", ends))
        }
        scores = {arm: short.item_metrics(r, teacher) for arm, r in recs.items()}
        diffs = {m: pl.paired_diff(scores["ends64"][m], scores["whole"][m]) for m in sl.KEPT_METRICS}
        out[model] = {
            "k": k,
            "ends64": {m: sl.kept_ci(scores["ends64"][m], base[m]) for m in sl.KEPT_METRICS},
            "whole": {m: sl.kept_ci(scores["whole"][m], base[m]) for m in sl.KEPT_METRICS},
            "unmasked": {m: float(np.mean(list(base[m].values()))) for m in sl.KEPT_METRICS},
            "diffs": diffs,
            "loops_summarize": {arm: sl.loop_rate(r, "summarize") for arm, r in recs.items()},
            "loads_per_layer": {
                arm: float(np.mean([float(x["loads_per_layer"]) for x in r])) for arm, r in recs.items()
            },
            "holds": model_holds({m: diffs[m] for m in CLAUSE_METRICS}),
        }
    result = {"verdict": verdict({m: r["holds"] for m, r in out.items()}), "models": out}
    OUT.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
