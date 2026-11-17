"""The three-vendor jury on the held-out items 600 to 899, from the stored verdicts: no juror is called.

    python analysis/three_vendor_heldout.py   # results/jury_heldout_h57.json

Three jurors (GPT-5.6 Luna, Gemini 3.5 Flash-Lite, Claude Haiku 5.5 in blind batches) answer yes or no to three
questions for each of the 2,700 answers (three models, the full model and two budgets, 300 items). The majority decides
each question, and an answer is accepted with three yeses. Accepted kept is the masked acceptance rate over the full
model's.
"""

import json
import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import numpy as np

import jury
import selrule as sl

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "jury_heldout_h57.json"
BUDGETS = {"gemma": (24, 48), "nemotron": (16, 48), "qwen36": (24, 128)}


def _votes(path: pathlib.Path) -> dict[int, dict[str, bool]]:
    return {r["id"]: r["verdict"] for r in jury._read(path) if r.get("verdict")}


def report() -> dict:
    items = jury._read(RESULTS / "jury_heldout" / "items.jsonl")
    votes = [
        _votes(RESULTS / "panel_heldout" / "claude_blind.jsonl"),
        _votes(RESULTS / "jury_heldout" / "luna.jsonl"),
        _votes(RESULTS / "jury_heldout" / "flash.jsonl"),
    ]
    accepted: dict[str, dict[int, float]] = {}
    for it in items:
        cast = [v[it["id"]] for v in votes if it["id"] in v]
        if len(cast) == 3:
            accepted.setdefault(it["cell"], {})[it["i"]] = float(jury.majority(cast)["accepted"])
    models: dict[str, dict] = {}
    for m, ks in BUDGETS.items():
        base = accepted[f"{m}_k0"]
        row: dict = {"full_accepted": float(np.mean(list(base.values()))), "n": len(base)}
        for k in ks:
            arm = accepted[f"{m}_k{k}"]
            kept = sl.kept_ci(arm, base)
            row[str(k)] = {"accepted": float(np.mean(list(arm.values()))), "kept": kept, "at_bar": sl.at_bar([kept])}
        models[m] = row
    cost = sum(
        r["usage"]["cost"]
        for name in ("luna", "flash")
        for r in jury._read(RESULTS / "jury_heldout" / f"{name}.jsonl")
        if r.get("usage")
    )
    out = {
        "items": "600 to 899",
        "jurors": ["claude_blind", "luna", "flash"],
        "rule": "majority on each question, accepted = all three",
        "cost_usd": round(cost, 2),
        "models": models,
    }
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
