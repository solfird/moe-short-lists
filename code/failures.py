"""Failure analysis from stored token ids (H20, H24): a common metric across tasks and failure types. No generation.

failures.py report   # results/failures_h20.json, failures_h24.json and the verdicts
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from short import MAX, bootstrap_ci

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
TAIL = 24
MAX_BLOCK = 12
GSM_MAX = 512
# (tag of the GSM8K files, tag of the short-suite files, routed experts kept at 1/8)
CELLS = {"gemma": ("gemma_ids", "gemma", 16), "qwen36": ("qwen36", "qwen36", 32)}
CONTRAST = {"nemotron": ("nemotron", "nemotron", 16), "qwen3": ("qwen_ids", "qwen", 16)}
DEGENERATE = ("empty", "loop")


def lcs_f1(a: Sequence[int], b: Sequence[int]) -> float:
    """F1 of the longest common subsequence of two token-id lists; two empty answers agree."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    row = [0] * (len(b) + 1)
    for x in a:
        diag = 0
        for j, y in enumerate(b, 1):
            diag, row[j] = row[j], diag + 1 if x == y else max(row[j], row[j - 1])
    n = row[-1]
    return 2 * n / (len(a) + len(b))


def is_loop(ids: Sequence[int]) -> bool:
    """True when the last TAIL tokens are one block of at most MAX_BLOCK tokens, repeated."""
    if len(ids) < TAIL:
        return False
    tail = list(ids[-TAIL:])
    return any(all(tail[i] == tail[i - p] for i in range(p, TAIL)) for p in range(1, MAX_BLOCK + 1))


def failure_type(ids: Sequence[int], text: str, limit: int) -> str:
    """One type per answer, first match wins: empty, loop, limit, other."""
    if not ids or not text.strip():
        return "empty"
    if is_loop(ids):
        return "loop"
    return "limit" if len(ids) >= limit else "other"


def first_token_agrees(masked: Sequence[int], unmasked: Sequence[int]) -> bool:
    """True when both answers start with the same token."""
    return bool(masked) and bool(unmasked) and masked[0] == unmasked[0]


def echoes_instruction(out: str, prompt: str) -> bool:
    """True when the answer repeats the first line of the prompt, the instruction."""
    return prompt.split("\n", 1)[0].strip().lower() in out.lower()


def gap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of a minus mean of b, with a 95% bootstrap interval; the two item sets are independent."""
    if not len(a) or not len(b):
        raise ValueError("gap: an empty side")
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    dx = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    dy = y[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(axis=1)
    lo, hi = np.percentile(dx - dy, [2.5, 97.5])
    return {"mean": float(x.mean() - y.mean()), "lo": float(lo), "hi": float(hi)}


def length_quartiles(lengths: dict[Any, int]) -> dict[Any, int]:
    """Quartile 0 to 3 of each item by length, ties broken by key, equal-sized groups."""
    order = sorted(lengths, key=lambda k: (lengths[k], k))
    return {k: i * 4 // len(order) for i, k in enumerate(order)}


def verdict(models: dict[str, dict]) -> str:
    """H20: holds, fails, or a metric effect named by model."""
    no_gap = [m for m, r in models.items() if r["a_gap"]["lo"] <= 0]
    if no_gap:
        return "metric effect: " + ", ".join(no_gap)
    for r in models.values():
        if r["b_excess"]["mean"] < 0.10 or r["b_excess"]["lo"] <= 0:
            return "fails"
        if r["gap_clean"]["mean"] > r["a_gap"]["mean"] / 2:
            return "fails"
    return "holds"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _rows(masked: list[dict], unmasked: list[dict], limit_of: Any) -> list[dict]:
    """Per-item common metric and failure type, matched on (task, index)."""
    base = {(r.get("task", "gsm8k"), r["i"]): r for r in unmasked}
    rows = []
    for r in masked:
        task = r.get("task", "gsm8k")
        u = base[(task, r["i"])]
        text = r.get("out", r.get("text", ""))
        rows.append(
            {
                "task": task,
                "i": r["i"],
                "lcs": lcs_f1(r["gen_ids"], u["gen_ids"]),
                "type": failure_type(r["gen_ids"], text, limit_of(task)),
                "first": first_token_agrees(r["gen_ids"], u["gen_ids"]),
                "len_unmasked": len(u["gen_ids"]),
            }
        )
    return rows


def _limit(task: str) -> int:
    return GSM_MAX if task == "gsm8k" else MAX[task]


def analyze(gsm_tag: str, short_tag: str, k: int) -> dict:
    """All H20 numbers of one model at one budget."""
    rows = _rows(
        _read(RESULTS / f"{gsm_tag}_gsm8k_k{k}_s32.jsonl"), _read(RESULTS / f"{gsm_tag}_gsm8k_k0_s32.jsonl"), _limit
    )
    rows += _rows(
        _read(RESULTS / f"{short_tag}_short_n300_k{k}_s32.jsonl"),
        _read(RESULTS / f"{short_tag}_short_n300_k0_s32.jsonl"),
        _limit,
    )
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    tasks = {}
    for task, rs in by_task.items():
        lcs = [r["lcs"] for r in rs]
        lo, hi = bootstrap_ci(lcs)
        quart = length_quartiles({r["i"]: r["len_unmasked"] for r in rs})
        tasks[task] = {
            "n": len(rs),
            "lcs": {"mean": float(np.mean(lcs)), "lo": lo, "hi": hi},
            "types": {t: sum(r["type"] == t for r in rs) / len(rs) for t in ("empty", "loop", "limit", "other")},
            "first_token": float(np.mean([r["first"] for r in rs])),
            "lcs_by_length_quartile": [float(np.mean([r["lcs"] for r in rs if quart[r["i"]] == q])) for q in range(4)],
        }
    gsm, summ = by_task["gsm8k"], by_task["summarize"]
    degen = lambda rs: [float(r["type"] in DEGENERATE) for r in rs]  # noqa: E731
    clean = lambda rs: [r["lcs"] for r in rs if r["type"] not in DEGENERATE]  # noqa: E731
    out = {
        "tasks": tasks,
        "a_gap": gap([r["lcs"] for r in gsm], [r["lcs"] for r in summ]),
        "b_excess": gap(degen(summ), degen(gsm)),
    }
    out["gap_clean"] = gap(clean(gsm), clean(summ))
    return out


def _ranks(x: Sequence[float]) -> np.ndarray:
    """Average ranks, ties shared."""
    _, inv, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2)[inv]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation; 0 when either side is constant."""
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Spearman correlation with a 95% bootstrap interval over items."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    lo, hi = np.percentile([spearman(a[i], b[i]) for i in idx], [2.5, 97.5])
    return {"mean": spearman(a, b), "lo": float(lo), "hi": float(hi)}


def h24_verdict(models: dict[str, dict[str, dict[str, float]]]) -> str:
    """H24: the first-token correlation has a CI entirely below 0 on at least 2 of 3 tasks, on every model."""
    return "holds" if all(sum(c["hi"] < 0 for c in m.values()) >= 2 for m in models.values()) else "fails"


def prompt_length_effect(gsm_tag: str, short_tag: str, k: int) -> dict[str, dict[str, dict[str, float]]]:
    """Within-task correlation of prompt length with first-token agreement and LCS F1 (H24)."""
    pairs = [("gsm8k", f"{gsm_tag}_gsm8k"), ("qa", f"{short_tag}_short_n300"), ("classify", f"{short_tag}_short_n300")]
    first, lcs = {}, {}
    for task, stem in pairs:
        base = {r["i"]: r for r in _read(RESULTS / f"{stem}_k0_s32.jsonl") if r.get("task", "gsm8k") == task}
        rs = [r for r in _read(RESULTS / f"{stem}_k{k}_s32.jsonl") if r.get("task", "gsm8k") == task]
        n = [r["prompt_ids_len"] for r in rs]
        first[task] = spearman_ci(n, [first_token_agrees(r["gen_ids"], base[r["i"]]["gen_ids"]) for r in rs])
        lcs[task] = spearman_ci(n, [lcs_f1(r["gen_ids"], base[r["i"]]["gen_ids"]) for r in rs])
    return {"first": first, "lcs": lcs}


def report() -> dict:
    """Write results/failures_h20.json and print the verdict."""
    primary = {m: analyze(*c) for m, c in CELLS.items()}
    contrast = {m: analyze(*c) for m, c in CONTRAST.items()}
    result = {"verdict": verdict(primary), "primary": primary, "contrast": contrast}
    (RESULTS / "failures_h20.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    res = report()
    print("H20", res["verdict"])
    effects = {m: prompt_length_effect(*c) for m, c in CELLS.items()}
    h24 = h24_verdict({m: e["first"] for m, e in effects.items()})
    (RESULTS / "failures_h24.json").write_text(json.dumps({"verdict": h24, "effects": effects}, indent=1))
    print("H24", h24, json.dumps(effects))
    for name, r in {**res["primary"], **res["contrast"]}.items():
        print(name, {k: round(r[k]["mean"], 3) for k in ("a_gap", "b_excess", "gap_clean")})
