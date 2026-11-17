"""Copying audit of summaries from stored answers: word runs lifted from the article, no generation.

copyaudit.py report   # results/copyaudit.json
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import failures
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUN = 8
CAP = 120
CELLS = {
    "gemma_d300_ends64": (12, 16, 24),
    "nemotron_d300_ends64": (8, 12, 16),
    "qwen36_d300_ends64": (12, 16, 24, 32, 48),
    "gemma_ends64": (8, 12, 16, 24, 32),
    "nemotron_ends64": (8, 12, 16, 24, 32),
    "qwen36_ends64": (16, 24, 32, 48, 64),
    "gemma_d300": (16,),
    "nemotron_d300": (16,),
    "qwen36_d300": (32,),
}


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _grams(words: list[str]) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    for j in range(len(words) - RUN + 1):
        out.setdefault(tuple(words[j : j + RUN]), j)
    return out


def copied_share(answer: str, article: str) -> float:
    """Share of the answer's words inside a run of RUN or more words that also appears in the article."""
    a = _words(answer)
    if len(a) < RUN:
        return 0.0
    grams = _grams(_words(article))
    covered = np.zeros(len(a), dtype=bool)
    for j in range(len(a) - RUN + 1):
        if tuple(a[j : j + RUN]) in grams:
            covered[j : j + RUN] = True
    return float(covered.mean())


def first_copy_position(answer: str, article: str) -> float | None:
    """Where in the article the answer's first copied run sits, as a share of the article; None if nothing is copied."""
    a, b = _words(answer), _words(article)
    grams = _grams(b)
    for j in range(len(a) - RUN + 1):
        hit = grams.get(tuple(a[j : j + RUN]))
        if hit is not None:
            return hit / len(b)
    return None


def _row(recs: list[dict], prompts: dict[int, str]) -> dict[str, Any]:
    sm = [r for r in recs if r["task"] == "summarize"]
    share = [copied_share(r["out"], prompts[r["i"]]) for r in sm]
    pos = [p for p in (first_copy_position(r["out"], prompts[r["i"]]) for r in sm) if p is not None]
    return {
        "n": len(sm),
        "copied_mean": float(np.mean(share)),
        "over_half": int(sum(s > 0.5 for s in share)),
        "loops": int(sum(failures.is_loop(r["gen_ids"]) for r in sm)),
        "at_cap": int(sum(len(r["gen_ids"]) >= CAP for r in sm)),
        "first_copy_median": float(np.median(pos)) if pos else None,
        "starts_in_first_tenth": int(sum(p < 0.1 for p in pos)),
    }


def report() -> dict:
    """Write results/copyaudit.json: every stored masked cell, with the unmasked run on the same items."""
    prompts = {start: {i: p for t, i, p, _ in short.items(300, start) if t == "summarize"} for start in (0, 300)}
    out: dict[str, Any] = {}
    for tag, ks in CELLS.items():
        p = prompts[300 if "d300" in tag else 0]
        for k in ks:
            path = short.run_path(tag, 300, k, 32)
            out[f"{tag}_k{k}"] = _row(short.read_records(path), p)
    for model in ("gemma", "nemotron", "qwen36"):
        for tag, start in ((f"{model}_d300", 300), (model, 0)):
            out[f"{tag}_k0"] = _row(short.read_records(short.run_path(tag, 300, 0, 32)), prompts[start])
    (RESULTS / "copyaudit.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
