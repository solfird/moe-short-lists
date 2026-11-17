"""H66: is the damage to masked summaries present in the first 24 words? Stored answers only, no generation.

earlydamage.py report   # results/earlydamage.json
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import baseline
import copyaudit
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "earlydamage.json"
HEAD = 24
START = 600
CELLS = {"gemma": 24, "nemotron": 16, "qwen36": 24}
NUMBER_MODELS = ("gemma", "qwen36")
COPY_MODELS = ("nemotron", "qwen36")
Counts = tuple[int, int]


def head(text: str, n: int = HEAD) -> str:
    return " ".join(text.split()[:n])


def rest(text: str, n: int = HEAD) -> str:
    return " ".join(text.split()[n:])


def number_counts(answer: str, article: str) -> Counts:
    """(numbers of the answer absent from the article, numbers the answer writes)."""
    foreign, written = baseline.foreign_numbers(answer, article)
    return len(foreign), written


def pooled_rate(counts: list[Counts]) -> float | None:
    """Foreign numbers per number written, over all items. None when no number is written."""
    written = sum(w for _, w in counts)
    return sum(f for f, _ in counts) / written if written else None


def share(flags: list[bool]) -> float:
    return float(np.mean(flags))


def paired_ci(
    masked: list, base: list, stat: Callable[[list], float | None], n: int = 2000, seed: int = 0
) -> list[float]:
    """95% bootstrap CI over items of stat(masked) minus stat(base), the same items drawn for both."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(masked), len(masked))
        a, b = stat([masked[i] for i in idx]), stat([base[i] for i in idx])
        if a is not None and b is not None:
            vals.append(a - b)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def head_copied(answer: str, article: str, n: int = HEAD) -> bool:
    """True when more than half of the head's words sit in runs copied from the article."""
    return copyaudit.copied_share(head(answer, n), article) > 0.5


def clause_a(masked: float, base: float, ci: tuple[float, float] | list[float]) -> bool:
    return masked >= 2 * base and ci[0] > 0


def clause_b(diff: float, ci: tuple[float, float] | list[float]) -> bool:
    return diff >= 0.20 and ci[0] > 0


def verdict(a: dict[str, bool], b: dict[str, bool]) -> str:
    if all(a.values()) and all(b.values()):
        return "early"
    return "not early" if not any(a.values()) and not any(b.values()) else "split"


def _summaries(tag: str, k: int) -> dict[int, str]:
    recs = short.read_records(short.run_path(tag, 300, k, 32))
    return {int(r["i"]): r["out"] for r in recs if r["task"] == "summarize"}


def _part(masked: dict[int, str], base: dict[int, str], articles: dict[int, str], cut: Callable[[str], str]) -> dict:
    ids = sorted(set(masked) & set(base))
    num_m = [number_counts(cut(masked[i]), articles[i]) for i in ids]
    num_b = [number_counts(cut(base[i]), articles[i]) for i in ids]
    cop_m = [copyaudit.copied_share(cut(masked[i]), articles[i]) > 0.5 for i in ids]
    cop_b = [copyaudit.copied_share(cut(base[i]), articles[i]) > 0.5 for i in ids]
    return {
        "n": len(ids),
        "numbers_written": [sum(w for _, w in num_m), sum(w for _, w in num_b)],
        "foreign_rate": [pooled_rate(num_m), pooled_rate(num_b)],
        "foreign_rate_diff_ci": paired_ci(num_m, num_b, pooled_rate),
        "copied_share": [share(cop_m), share(cop_b)],
        "copied_diff_ci": paired_ci(cop_m, cop_b, share),
    }


def report() -> dict[str, Any]:
    articles = {int(i): p for t, i, p, _ in short.items(300, START) if t == "summarize"}
    out: dict[str, Any] = {"head_words": HEAD, "items": f"{START} to {START + 299}", "models": {}}
    a: dict[str, bool] = {}
    b: dict[str, bool] = {}
    for model, k in CELLS.items():
        masked, base = _summaries(f"{model}_d{START}_ends64", k), _summaries(f"{model}_d{START}", 0)
        row = {
            "k": k,
            "words_mean": [float(np.mean([len(x.split()) for x in d.values()])) for d in (masked, base)],
            "head": _part(masked, base, articles, head),
            "rest": _part(masked, base, articles, rest),
        }
        h = row["head"]
        if model in NUMBER_MODELS:
            a[model] = clause_a(h["foreign_rate"][0], h["foreign_rate"][1], h["foreign_rate_diff_ci"])
        if model in COPY_MODELS:
            b[model] = clause_b(h["copied_share"][0] - h["copied_share"][1], h["copied_diff_ci"])
        out["models"][model] = row
    out["clause_a"], out["clause_b"], out["verdict"] = a, b, verdict(a, b)
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
