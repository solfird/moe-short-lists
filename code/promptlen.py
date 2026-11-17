"""Prompt length as a cause of first-token damage (H25): the same tasks with cut articles or a padded prompt.

promptlen.py run --model-tag gemma --arm c8 --frac 0.125 --n 200   # also --frac 0 for its unmasked run
promptlen.py report --n 200                                        # results/promptlen_h25.json and the verdict
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import failures as fl
import short

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
CUTS = {"c8": 0.125, "c4": 0.25, "c2": 0.5, "c1": 1.0}
ARMS = (*CUTS, "pad")
FULL = 3000
Item = tuple[str, int, str, Any]
MODELS = {"gemma": 16, "qwen36": 32}


def tag(model_tag: str, arm: str) -> str:
    """Output tag of one arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    return f"{model_tag}_pl_{arm}"


def summarize_items(articles: Sequence[str], highlights: Sequence[str], frac: float) -> list[Item]:
    """Summarize items of short.items, with the article cut to frac of its 3,000 characters."""
    size = int(FULL * frac)
    return [
        ("summarize", i, "Summarize this news article in 2 or 3 sentences.\n\n" + a[:size], h)
        for i, (a, h) in enumerate(zip(articles, highlights, strict=True))
    ]


def pad_item(item: Item, article: str) -> Item:
    """One item with an unrelated article of up to 3,000 characters before its unchanged prompt."""
    task, i, prompt, ref = item
    return task, i, article[:FULL] + "\n\n" + prompt, ref


def arm_items(arm: str, n: int) -> list[Item]:
    """The n items of one arm; indices match the short suite, so arms pair with its runs."""
    from datasets import load_dataset

    cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    if arm in CUTS:
        rows = cnn.select(short.article_slices(n)["summarize"])
        return summarize_items(rows["article"], rows["highlights"], CUTS[arm])
    fillers = cnn.select(short.article_slices(n)["rewrite"])["article"]
    plain = [x for x in short.items(n) if x[0] == "classify"]
    return [pad_item(x, a) for x, a in zip(plain, fillers, strict=True)]


def paired_diff(a: dict[int, float], b: dict[int, float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of a minus b on matched items, with a 95% paired bootstrap interval."""
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError("paired_diff: no matched items")
    d = np.array([a[k] - b[k] for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(keys)}


def verdict(models: dict[str, dict[str, dict[str, float]]]) -> str:
    """H25: fails if the cut misses on a model; partial if the cut passes but padding misses; else holds."""
    if any(r["cut"]["mean"] < 0.20 or r["cut"]["lo"] <= 0 for r in models.values()):
        return "fails"
    if any(r["pad"]["mean"] > -0.20 or r["pad"]["hi"] >= 0 for r in models.values()):
        return "partial"
    return "holds"


def cmd_run(a: argparse.Namespace) -> None:
    """One arm through short.cmd_run, the path of every other run, with the suite replaced by the arm."""
    original = short.items
    built = arm_items(a.arm, a.n)  # the padded arm reads short.items, so build it before the replacement
    short.items = lambda n: built
    try:
        short.cmd_run(argparse.Namespace(model_tag=a.model_tag, tag=tag(a.model_tag, a.arm), n=a.n, frac=a.frac, s=32))
    finally:
        short.items = original


def _scores(masked: Path, unmasked: Path, task: str) -> dict[str, Any]:
    """Per-item first-token agreement and LCS F1 of one run against its unmasked run, with the mean prompt length."""
    base = {r["i"]: r for r in fl._read(unmasked) if r["task"] == task}
    rs = [r for r in fl._read(masked) if r["task"] == task]
    return {
        "first": {r["i"]: float(fl.first_token_agrees(r["gen_ids"], base[r["i"]]["gen_ids"])) for r in rs},
        "lcs": {r["i"]: fl.lcs_f1(r["gen_ids"], base[r["i"]]["gen_ids"]) for r in rs},
        "prompt_tokens": float(np.mean([r["prompt_ids_len"] for r in rs])),
    }


def report(n: int) -> dict:
    """Write results/promptlen_h25.json: arms against the stored full runs on the same items."""
    out: dict[str, Any] = {}
    for model, k in MODELS.items():
        stored = RESULTS / f"{model}_short_n300_k{{}}_s32.jsonl"
        full = _scores(Path(str(stored).format(k)), Path(str(stored).format(0)), "summarize")
        plain = _scores(Path(str(stored).format(k)), Path(str(stored).format(0)), "classify")

        def arm(a: str, task: str, model: str = model, k: int = k) -> dict[str, Any]:
            return _scores(short.run_path(tag(model, a), n, k, 32), short.run_path(tag(model, a), n, 0, 32), task)

        row: dict[str, Any] = {"full_tokens": full["prompt_tokens"], "plain_tokens": plain["prompt_tokens"]}
        for a in ("c8", "c4", "c2"):
            s = arm(a, "summarize")
            row[a] = {
                "prompt_tokens": s["prompt_tokens"],
                "first": float(np.mean(list(s["first"].values()))),
                "lcs": float(np.mean(list(s["lcs"].values()))),
                "first_minus_full": paired_diff(s["first"], full["first"]),
            }
        pad = arm("pad", "classify")
        row["cut"] = row["c8"]["first_minus_full"]
        row["pad"] = paired_diff(pad["first"], plain["first"])
        row["pad_prompt_tokens"] = pad["prompt_tokens"]
        row["pad_lcs"] = float(np.mean(list(pad["lcs"].values())))
        out[model] = row
    result = {"verdict": verdict(out), "models": out}
    (RESULTS / "promptlen_h25.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True, choices=list(MODELS))
    r.add_argument("--arm", required=True, choices=[x for x in ARMS if x != "c1"])
    r.add_argument("--frac", type=float, required=True)
    r.add_argument("--n", type=int, default=200)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("--n", type=int, default=200)
    s.set_defaults(fn=lambda a: print(json.dumps(report(a.n), indent=1)))
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
