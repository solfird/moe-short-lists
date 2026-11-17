"""Unmasked router trace of the short suite, prompt tokens included: one forward pass on the stored answers (H21).

python -m tools.short_trace --model-tag gemma --n 300   # runs/short_trace/gemma.npz, about 10 to 30 minutes
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "runs" / "short_trace"
Trace = dict[tuple[str, int], tuple[np.ndarray, int, float]]


def forward_ids(prompt: list[int], gen: list[int]) -> list[int]:
    """Tokens the model reads while it generates: the prompt and every answer token but the last."""
    return prompt + gen[:-1]


def answer_part(ids: np.ndarray, n_prompt: int) -> np.ndarray:
    """The answer tokens of a (tokens, layers, k) trace."""
    return ids[n_prompt:]


def agreement(logits: np.ndarray, n_prompt: int, gen_ids: list[int]) -> float:
    """Share of stored answer tokens that the forward pass predicts; 1 for an empty answer."""
    if not gen_ids:
        return 1.0
    pred = logits[n_prompt - 1 : n_prompt - 1 + len(gen_ids)].argmax(-1)
    return float(np.mean(pred == np.array(gen_ids)))


def save(path: Path, items: Trace) -> None:
    """One npz: ids, prompt length and agreement of each item."""
    arrays = {}
    for (task, i), (ids, n_prompt, agree) in items.items():
        arrays[f"{task}:{i}:ids"] = ids
        arrays[f"{task}:{i}:meta"] = np.array([n_prompt, agree])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_trace(path: Path) -> Trace:
    """Inverse of save."""
    d = np.load(path)
    out: Trace = {}
    for key in d.files:
        task, i, kind = key.split(":")
        if kind == "ids":
            meta = d[f"{task}:{i}:meta"]
            out[(task, int(i))] = (d[key], int(meta[0]), float(meta[1]))
    return out


def memory_note(active: int, cache: int, peak: int) -> str:
    """MLX memory in GiB, from bytes, for the progress line."""
    return f"active {active / 2**30:.1f} GiB, cache {cache / 2**30:.1f} GiB, peak {peak / 2**30:.1f} GiB"


def main() -> None:
    import mlx.core as mx
    from mlx_lm import load

    import segroute as sr
    import short
    from models import MODELS
    from tools.cache_replay import decode_ids
    from tools.router_stats import TOP_K

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-tag", required=True)
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()
    stored = {
        (r["task"], r["i"]): r["gen_ids"] for r in map(json.loads, short.run_path(a.model_tag, a.n, 0, 32).open())
    }
    model, tok = load(MODELS[a.model_tag])
    ctl = sr.install(model)
    ctl.mode = "log"
    items: Trace = {}
    t0 = time.time()
    for task, i, prompt, _ in short.items(a.n):
        ctl.reset()
        ids, gen = sr.chat(tok, prompt), stored[(task, i)]
        logits = model(mx.array(forward_ids(ids, gen))[None]).astype(mx.float32)
        mx.eval(logits)
        probs = np.stack([np.array(mx.concatenate(ctl.log[lyr], axis=0)) for lyr in range(ctl.n_layers)])
        top = decode_ids(probs, 0, TOP_K[a.model_tag]).astype(np.int16)
        items[(task, i)] = (top, len(ids), agreement(np.array(logits[0]), len(ids), gen))
        cache = mx.get_cache_memory()
        mx.clear_cache()  # full passes of varying length leave buffers that later items cannot reuse
        if len(items) % 100 == 0:
            note = memory_note(mx.get_active_memory(), cache, mx.get_peak_memory())
            print(f"{len(items)} items, {time.time() - t0:.0f}s, {note}", flush=True)
    save(OUT / f"{a.model_tag}.npz", items)
    print("mean agreement with the stored answers:", float(np.mean([v[2] for v in items.values()])))


if __name__ == "__main__":
    main()
