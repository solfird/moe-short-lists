"""Router flatness and top-k weight kept by the C1 segment sets, replayed on unmasked router logs.

Reads runs/{tag}/*.npz (segroute.py log: probs of shape (layers, tokens, experts), prompt tokens first) and
writes results/router_flatness.csv and results/router_kept.csv; the second carries the quality kept from
results/summary.csv at the same K' and S (run tools.collect first). Run from the repo root:
python -m tools.router_stats
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from models import N_EXPERTS
from short import budget
from sweep import FRACS

HERE = Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"
RESULTS = HERE / "results"
TOP_K = {"gemma": 8, "qwen36": 8, "nemotron": 6, "qwen": 8}
# Logs hold the full softmax for these; Nemotron's GateTap logs only its normalized top-k weights.
FULL_DISTRIBUTION = {"gemma", "qwen36", "qwen"}
QUALITY = ("sum_rougeL_ref", "qa_f1", "cls_acc")
MIN_GEN = 8


def _top(gen: np.ndarray, top_k: int) -> np.ndarray:
    """Indices of the top_k experts per token, as the unmasked router picks them."""
    return np.argpartition(-gen, top_k, axis=-1)[..., :top_k]


def segment_kept(probs: np.ndarray, n_prompt: int, k: int, s: int, top_k: int) -> tuple[float, float]:
    """Mean share of the unmasked top-k weight inside the segment set, and mean share of top-k experts inside it.

    The set is the top K' of the prompt's summed scores, then of each finished S-token window; it applies from
    the next generated token, as in segroute.Controller. Means run over every (layer, generated token).
    """
    masses, hits = [], []
    for p in probs:
        gen = p[n_prompt:]
        top = _top(gen, top_k)
        top_p = np.take_along_axis(gen, top, axis=-1)
        n_seg = -(-len(gen) // s)
        windows = [p[:n_prompt].sum(0)] + [gen[j * s : (j + 1) * s].sum(0) for j in range(n_seg - 1)]
        allowed = np.zeros((n_seg, p.shape[1]), dtype=bool)
        for j, w in enumerate(windows):
            allowed[j, np.argsort(-w)[:k]] = True
        inside = np.take_along_axis(allowed[np.arange(len(gen)) // s], top, axis=-1)
        masses.append((top_p * inside).sum(-1) / top_p.sum(-1))
        hits.append(inside.mean(-1))
    return float(np.concatenate(masses).mean()), float(np.concatenate(hits).mean())


def flatness(probs: np.ndarray, n_prompt: int, top_k: int) -> tuple[float, float, float]:
    """Mean top-1 weight, top-k weight and entropy over log(experts), over generated tokens of every layer."""
    gen = probs[:, n_prompt:].astype(np.float64)
    top1 = gen.max(-1).mean()
    topk = np.take_along_axis(gen, _top(gen, top_k), axis=-1).sum(-1).mean()
    entropy = -(gen * np.log(np.where(gen > 0, gen, 1.0))).sum(-1) / np.log(gen.shape[-1])
    return float(top1), float(topk), float(entropy.mean())


def join_quality(rows: list[dict[str, Any]], summary: Path) -> list[dict[str, Any]]:
    """Add the short-suite quality kept (share of unmasked) at the same model, K' and S to each row."""
    kept: dict[tuple[str, int, int], dict[str, float]] = {}
    with open(summary) as f:
        for r in csv.DictReader(f):
            if r["suite"] == "short" and r["metric"] in QUALITY and r["kept"]:
                kept.setdefault((r["model_tag"], int(r["k"]), int(r["s"])), {})[r["metric"]] = float(r["kept"])
    return [{**row, **kept.get((row["model_tag"], row["k"], row["s"]), {})} for row in rows]


def _logs(tag: str) -> list[tuple[str, np.ndarray, int]]:
    """(kind, probs, n_prompt) of every logged answer with at least MIN_GEN generated tokens."""
    meta = {m["i"]: m for m in map(json.loads, (RUNS / tag / "meta.jsonl").read_text().splitlines())}
    out = []
    for path in sorted((RUNS / tag).glob("*.npz")):
        d = np.load(path)
        probs, n_prompt = d["probs"].astype(np.float32), int(d["n_prompt"])
        if probs.shape[1] - n_prompt >= MIN_GEN:
            out.append((meta.get(int(path.stem), {}).get("kind", "?"), probs, n_prompt))
    return out


def analyse(tag: str, ss: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatness rows and kept rows of one model; answers are weighted equally."""
    logs, top_k, n_exp = _logs(tag), TOP_K[tag], N_EXPERTS[tag]
    flat = []
    if tag in FULL_DISTRIBUTION:
        per = np.array([flatness(p, n, top_k) for _, p, n in logs])
        flat.append(
            {
                "model_tag": tag,
                "n_experts": n_exp,
                "top_k": top_k,
                "n_answers": len(logs),
                "top1_weight": per[:, 0].mean(),
                "topk_weight": per[:, 1].mean(),
                "entropy_norm": per[:, 2].mean(),
            }
        )
    kept = []
    for s in ss:
        for frac in FRACS:
            k = budget(n_exp, frac)
            per = np.array([segment_kept(p, n, k, s, top_k) for _, p, n in logs])
            kept.append(
                {
                    "model_tag": tag,
                    "k": k,
                    "frac": k / n_exp,
                    "s": s,
                    "n_answers": len(logs),
                    "mass": per[:, 0].mean(),
                    "mass_p10": np.percentile(per[:, 0], 10),
                    "hit": per[:, 1].mean(),
                }
            )
    return flat, kept


def _write(rows: list[dict[str, Any]], out: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Analyse every model with router logs and write the two CSVs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=[t for t in TOP_K if (RUNS / t / "meta.jsonl").exists()])
    p.add_argument("--ss", type=int, nargs="+", default=[32])
    a = p.parse_args()
    flat, kept = [], []
    for tag in a.tags:
        f, k = analyse(tag, a.ss)
        flat += f
        kept += k
        print(f"{tag}: {k[0]['n_answers']} answers", flush=True)
    kept = join_quality(kept, RESULTS / "summary.csv")
    _write(flat, RESULTS / "router_flatness.csv")
    _write(kept, RESULTS / "router_kept.csv")
    for row in flat:
        print(
            f"{row['model_tag']:>9} top1 {row['top1_weight']:.3f} top{row['top_k']} {row['topk_weight']:.3f}"
            f" entropy {row['entropy_norm']:.3f}"
        )
    for row in kept:
        quality = " ".join(f"{m} {row[m]:.2f}" for m in QUALITY if m in row)
        head = f"{row['model_tag']:>9} K'={row['k']:<3} S={row['s']}"
        print(f"{head} mass {row['mass']:.3f} hit {row['hit']:.3f} {quality}")


if __name__ == "__main__":
    main()
