"""llama.cpp MoE expert cache (LRU and SLRU) replayed on unmasked router logs, against the segment mask at equal slots.

The cache keeps exact routing and pays misses; the mask caps the set and pays quality. Both are counted in expert reads
per generated token per layer, on the same logs (runs/{tag}/*.npz). The cache code is llama.cpp's own, copied
into tools/cache_sim.cpp and compiled to build/cache_sim. Slots per layer follow the C1 budgets; the cache pools them
over all MoE layers, as llama.cpp does. Writes results/cache_replay.csv. Run from the repo root:
python -m tools.cache_replay
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from models import N_EXPERTS
from short import budget
from sweep import FRACS
from tools.router_stats import RESULTS, RUNS, TOP_K, _logs, join_quality

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cache_sim.cpp"
BINARY = HERE.parent / "build" / "cache_sim"
S = 32


def build() -> Path:
    """Compile cache_sim.cpp when the binary is missing or older than the source."""
    if not BINARY.exists() or BINARY.stat().st_mtime < SOURCE.stat().st_mtime:
        BINARY.parent.mkdir(exist_ok=True)
        subprocess.run(["c++", "-O2", "-std=c++17", "-o", str(BINARY), str(SOURCE)], check=True)
    return BINARY


def decode_ids(probs: np.ndarray, n_prompt: int, top_k: int) -> np.ndarray:
    """Top-k expert ids of each generated token, shape (tokens, layers, k), in decode order."""
    gen = probs[:, n_prompt:]
    return np.ascontiguousarray(np.argpartition(-gen, top_k, axis=-1)[..., :top_k].transpose(1, 0, 2), dtype=np.int32)


def write_trace(answers: list[np.ndarray], n_expert: int, path: Path) -> None:
    """Binary trace for cache_sim: header, tokens per answer, then the ids."""
    n_layer, k = answers[0].shape[1:]
    head = np.array([n_layer, n_expert, k, len(answers)] + [len(a) for a in answers], dtype="<i4")
    with open(path, "wb") as f:
        f.write(head.tobytes())
        for a in answers:
            f.write(a.astype("<i4").tobytes())


def run_trace(path: Path, n_slots: int, policy: str, warm: bool) -> dict[str, int]:
    """Hits, misses, accesses and tokens of one cache replay."""
    out = subprocess.run(
        [str(build()), str(path), str(n_slots), policy, "warm" if warm else "cold"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return dict(zip(("hits", "misses", "accesses", "tokens"), map(int, out), strict=True))


def simulate(answers: list[np.ndarray], n_expert: int, n_slots: int, policy: str, warm: bool) -> dict[str, int]:
    """Replay the llama.cpp cache over answers (each (tokens, layers, k)); a cold cache restarts every answer."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.bin"
        write_trace(answers, n_expert, path)
        return run_trace(path, n_slots, policy, warm)


def mask_loads(probs: np.ndarray, n_prompt: int, k: int, s: int) -> float:
    """Experts read per layer for one answer under the mask: the prompt set, then the new experts per reselection."""
    loads = 0
    for p in probs:
        gen = p[n_prompt:]
        sets = [set(np.argsort(-p[:n_prompt].sum(0))[:k].tolist())]
        for j in range(len(gen) // s):
            sets.append(set(np.argsort(-gen[j * s : (j + 1) * s].sum(0))[:k].tolist()))
        loads += k + sum(len(new - old) for old, new in zip(sets, sets[1:], strict=False))
    return loads / len(probs)


def analyse(tag: str, tmp: Path) -> list[dict[str, Any]]:
    """Cache rows (two policies, cold and warm) and mask rows for one model, at every C1 budget."""
    logs, top_k, n_exp = _logs(tag), TOP_K[tag], N_EXPERTS[tag]
    answers = [decode_ids(p, n, top_k) for _, p, n in logs]
    n_layer, n_gen = answers[0].shape[1], sum(len(a) for a in answers)
    path = tmp / f"{tag}.bin"
    write_trace(answers, n_exp, path)
    rows = []
    for frac in FRACS:
        k = budget(n_exp, frac)
        base = {"model_tag": tag, "k": k, "frac": k / n_exp, "n_layers": n_layer, "n_answers": len(logs)}
        for policy in ("lru", "slru"):
            for warm in (False, True):
                r = run_trace(path, k * n_layer, policy, warm)
                rows.append(
                    {
                        **base,
                        "method": f"cache_{policy}",
                        "start": "warm" if warm else "cold",
                        "hit_rate": r["hits"] / r["accesses"],
                        "reads_per_token_layer": r["misses"] / (r["tokens"] * n_layer),
                        "reads_per_layer_answer": r["misses"] / (n_layer * len(logs)),
                    }
                )
        loads = sum(mask_loads(p, n, k, S) for _, p, n in logs)
        rows.append(
            {
                **base,
                "method": "segment_mask",
                "s": S,
                "start": "cold",
                "reads_per_token_layer": loads / n_gen,
                "reads_per_layer_answer": loads / len(logs),
            }
        )
    return rows


def main() -> None:
    """Replay every model with router logs and write results/cache_replay.csv."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=[t for t in TOP_K if (RUNS / t / "meta.jsonl").exists()])
    a = p.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag in a.tags:
            rows += analyse(tag, Path(tmp))
            print(f"{tag}: done", flush=True)
    masked = join_quality([r for r in rows if r["method"] == "segment_mask"], RESULTS / "summary.csv")
    rows = [r for r in rows if r["method"] != "segment_mask"] + masked
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(RESULTS / "cache_replay.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for r in sorted(rows, key=lambda r: (r["model_tag"], r["k"], r["method"], r["start"])):
        hit = f"hit {r['hit_rate']:.3f}" if "hit_rate" in r else "         "
        head = f"{r['model_tag']:>9} K'={r['k']:<3} {r['method']:<13} {r['start']:<4}"
        print(f"{head} {hit} reads/tok/layer {r['reads_per_token_layer']:.3f}")


if __name__ == "__main__":
    main()
