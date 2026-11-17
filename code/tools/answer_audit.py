"""Damage the kept score does not see: loops, distance from the unmasked answer, names and numbers not in the prompt.

python -m tools.answer_audit   # results/answer_audit.json
python -m tools.answer_audit --start 300   # results/answer_audit_d300.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import failures
import short

RESULTS = Path(__file__).resolve().parents[2] / "results"
CELLS = {"gemma": ("gemma", 16), "nemotron": ("nemotron", 16), "qwen36": ("qwen36", 32)}
RULES = ("tail64", "head64", "ends64")
TASKS = ("summarize", "qa", "classify", "rewrite")
SPLIT = re.compile(r"(?<=[.!?:\n])\s+|\n+")
WORD = re.compile(r"\d[\d,.:]*\d|\d|[A-Z][\w'-]*")


def names_and_numbers(text: str) -> set[str]:
    """Lower-cased numbers and capitalized words, without the first word of each sentence."""
    found: set[str] = set()
    for sentence in SPLIT.split(text):
        words = sentence.split()
        for word in words[1:]:
            found.update(m.lower() for m in WORD.findall(word))
        if words:
            found.update(m.lower() for m in WORD.findall(words[0]) if m[0].isdigit())
    return found


def supported_share(out: str, prompt: str) -> float | None:
    """Share of the answer's names and numbers that occur in the prompt. None: the answer names nothing."""
    found = names_and_numbers(out)
    if not found:
        return None
    low = prompt.lower()
    return sum(x in low for x in found) / len(found)


def summarize_task(recs: list[dict], unmasked: dict[Any, dict], prompts: dict[Any, str]) -> dict[str, Any]:
    """Audit of one task of one run."""
    key = [(r["task"], r["i"]) for r in recs]
    shares = [supported_share(r["out"], prompts[k]) for r, k in zip(recs, key, strict=True)]
    named = [x for x in shares if x is not None]
    pairs = [(r["gen_ids"], unmasked[k]["gen_ids"]) for r, k in zip(recs, key, strict=True)]
    return {
        "n": len(recs),
        "loop": float(np.mean([failures.is_loop(r["gen_ids"]) for r in recs])),
        "same": float(np.mean([a == b for a, b in pairs])),
        "lcs": float(np.mean([failures.lcs_f1(a, b) for a, b in pairs])),
        "supported": float(np.mean(named)) if named else None,
    }


def _read(name: str) -> list[dict]:
    return [json.loads(x) for x in (RESULTS / f"{name}.jsonl").open()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=0, help="position of the first item: 300 for the H30 runs")
    start = p.parse_args().start
    prompts = {(task, i): prompt for task, i, prompt, _ in short.items(300, start)}
    prefix = f"_d{start}" if start else ""
    out: dict[str, Any] = {}
    for model, (base, k) in CELLS.items():
        base = f"{model}{prefix}" if start else base
        unmasked_recs = _read(f"{base}_short_n300_k0_s32")
        unmasked = {(r["task"], r["i"]): r for r in unmasked_recs}
        runs = {"unmasked": unmasked_recs, "whole": _read(f"{base}_short_n300_k{k}_s32")}
        runs |= {
            rule: _read(f"{model}{prefix}_{rule}_short_n300_k{k}_s32") for rule in (("ends64",) if start else RULES)
        }
        out[model] = {
            arm: {t: summarize_task([r for r in recs if r["task"] == t], unmasked, prompts) for t in TASKS}
            for arm, recs in runs.items()
        }
    (RESULTS / f"answer_audit{prefix}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
