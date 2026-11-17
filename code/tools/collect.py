"""Flatten every scored run into results/summary.csv: one row per model, suite, metric, K' and S.

Short-suite rows come from results/{tag}_short_n{n}_scores.json (run `short.py score --tag T --n N` first);
GSM8K rows are computed from the ok flags of results/{tag}_gsm8k_k*_s*.jsonl. Provenance comes from the
.meta.json beside each raw file. Run from the repo root:
python -m tools.collect
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from models import N_EXPERTS
from records import read_records

RESULTS = Path(__file__).resolve().parents[2] / "results"
RUN = re.compile(r"k(\d+)_s(\d+)")
SHORT_SCORES = re.compile(r"(.+)_short_n(\d+)_scores\.json")
GSM8K_FILE = re.compile(r"(.+)_gsm8k_k(\d+)_s(\d+)\.jsonl")
META_KEYS = ("model", "git_commit", "git_dirty", "start")
FIELDS = [
    "model_tag",
    "n_experts",
    "suite",
    "metric",
    "k",
    "frac",
    "s",
    "n_items",
    "n_records",
    "value",
    "lo",
    "hi",
    "kept",
    "kept_lo",
    "kept_hi",
    "loads_per_layer",
    "raw_file",
    *META_KEYS,
]


def parse_run(name: str) -> tuple[int, int]:
    """(K', S) of a run key such as k16_s32."""
    m = RUN.fullmatch(name)
    if not m:
        raise ValueError(f"not a run key: {name!r}")
    return int(m[1]), int(m[2])


def meta(path: Path) -> dict[str, Any]:
    """Provenance fields of a raw file's .meta.json; empty strings when there is none."""
    side = path.with_name(path.name + ".meta.json")
    found = json.loads(side.read_text()) if side.exists() else {}
    return {key: found.get(key, "") for key in META_KEYS}


def frac(tag: str, k: int) -> float | str:
    """K' as a share of the routed experts; empty for a tag outside the registry."""
    return k / N_EXPERTS[tag] if tag in N_EXPERTS else ""


def short_rows(tag: str, n: int, scores: dict[str, dict], results: Path) -> list[dict[str, Any]]:
    """Rows of one {tag}_short_n{n}_scores.json, one per run and metric."""
    rows = []
    for name, entry in scores.items():
        k, s = parse_run(name)
        raw = f"{tag}_short_n{n}_{name}.jsonl"
        kept = entry.get("kept") or {}
        for metric, est in entry.items():
            if not isinstance(est, dict) or "mean" not in est:
                continue
            ratio = kept.get(metric, {})
            rows.append(
                {
                    "model_tag": tag,
                    "n_experts": N_EXPERTS.get(tag, ""),
                    "suite": "short",
                    "metric": metric,
                    "k": k,
                    "frac": frac(tag, k),
                    "s": s,
                    "n_items": n,
                    "n_records": entry.get("n", ""),
                    "value": est["mean"],
                    "lo": est["lo"],
                    "hi": est["hi"],
                    "kept": ratio.get("mean", ""),
                    "kept_lo": ratio.get("lo", ""),
                    "kept_hi": ratio.get("hi", ""),
                    "loads_per_layer": entry.get("loads_per_layer") if entry.get("loads_per_layer") is not None else "",
                    "raw_file": raw,
                    **meta(results / raw),
                }
            )
    return rows


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a binomial rate."""
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def gsm8k_rows(results: Path) -> list[dict[str, Any]]:
    """Accuracy rows of every {tag}_gsm8k_k*_s*.jsonl, with Wilson intervals and mean loads per layer."""
    rows = []
    for path in sorted(results.glob("*_gsm8k_k*_s*.jsonl")):
        m = GSM8K_FILE.fullmatch(path.name)
        if not m:
            continue
        tag, k, s = m[1], int(m[2]), int(m[3])
        recs = read_records(path)
        if not recs:
            continue
        hits = sum(bool(r.get("ok")) for r in recs)
        lo, hi = wilson(hits, len(recs))
        loads = [r["loads_per_layer"] for r in recs if r.get("loads_per_layer") is not None]
        rows.append(
            {
                "model_tag": tag,
                "n_experts": N_EXPERTS.get(tag, ""),
                "suite": "gsm8k",
                "metric": "gsm8k_acc",
                "k": k,
                "frac": frac(tag, k),
                "s": s,
                "n_items": len(recs),
                "n_records": len(recs),
                "value": hits / len(recs),
                "lo": lo,
                "hi": hi,
                "kept": "",
                "kept_lo": "",
                "kept_hi": "",
                "loads_per_layer": sum(loads) / len(loads) if loads else "",
                "raw_file": path.name,
                **meta(path),
            }
        )
    return rows


def collect(results: Path) -> list[dict[str, Any]]:
    """Every short-suite score file and every GSM8K run file under results/."""
    rows = []
    for path in sorted(results.glob("*_short_n*_scores.json")):
        m = SHORT_SCORES.fullmatch(path.name)
        if m:
            rows += short_rows(m[1], int(m[2]), json.loads(path.read_text()), results)
    return rows + gsm8k_rows(results)


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    """Write rows under the fixed FIELDS header."""
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Rebuild results/summary.csv from the score and run files."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=RESULTS / "summary.csv")
    out = p.parse_args().out
    rows = collect(RESULTS)
    write_csv(rows, out)
    print(f"{out.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
