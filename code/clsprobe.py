"""First-token label scores on classify across label orders (H40), and a second label set (emotion).

clsprobe.py run --model-tag gemma --k 24 --label untrained --rules ends --orders 0,1,2,3
clsprobe.py run --model-tag gemma --k 24 --label untrained --rules ends --orders 0,1 --suite emotion
clsprobe.py report-h40                                                 # results/clsprobe_h40.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import selrule as sl
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
START, N = 300, 300
LIST = "World, Sports, Business, Sci/Tech"
ROTATED = "Sports, Business, Sci/Tech, World"
LABELS = ("World", "Sports", "Business", "Sci/Tech")
EMOTIONS = ("sadness", "joy", "love", "anger", "fear", "surprise")
FRONTIER = {"gemma": 24, "nemotron": 16, "qwen36": 24}


def rotate_list(prompt: str, labels: tuple[str, ...], n: int = 1) -> str:
    """The prompt with its label list rotated by n: the first n labels move to the end."""
    written = ", ".join(labels)
    if written not in prompt:
        raise ValueError("no label list in this prompt")
    return prompt.replace(written, ", ".join((*labels[n:], *labels[:n])), 1)


def find_spans(ids: list[int], spellings: dict[str, list[list[int]]]) -> dict[str, tuple[int, int]]:
    """First token range of each label, under any of its spellings. Refuses a label it cannot find."""
    out = {}
    for label, forms in spellings.items():
        hits = [
            (i, i + len(form))
            for form in forms
            for i in range(len(ids) - len(form) + 1)
            if ids[i : i + len(form)] == form
        ]
        if not hits:
            raise ValueError(f"label {label} not found")
        out[label] = min(hits)
    return out


def rule_parts(rule: str) -> tuple[str, int]:
    """(selection rule, experts reserved per label): labelsN is ends64 with N experts kept for each label's tokens."""
    return ("ends", int(rule[6:])) if rule.startswith("labels") else (rule, 0)


def order_summary(kept: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(kept)), "worst": min(kept), "gap": max(kept) - min(kept)}


def h40_verdict(models: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Holds if classify is at the bar in every label order on every model."""
    bars = {m: all(sl.at_bar([d]) for d in cells) for m, cells in models.items()}
    return {"verdict": "holds" if all(bars.values()) else "fails", "at_bar": bars}


def emotion_prompt(text: str) -> str:
    """Classify prompt for SetFit/emotion, in the suite's format."""
    return f"Classify this text into one of: {', '.join(EMOTIONS)}. Reply with the label only.\n\n{text}"


def emotion_window(start: int) -> tuple[int, int]:
    """Rows of the shuffled emotion test for a run at this item start: 300 gives the first N, 600 the next N."""
    lo = start - START
    return lo, lo + N


def emotion_items(start: int = START) -> list[tuple[str, int, str, str]]:
    from datasets import load_dataset

    lo, hi = emotion_window(start)
    d = load_dataset("SetFit/emotion", split="test").shuffle(seed=0).select(range(lo, hi))
    return [("classify", j, emotion_prompt(r["text"]), r["label_text"]) for j, r in enumerate(d)]


def article_span(ids: list[int], empty: list[int]) -> tuple[int, int]:
    """Token range of the article: what differs from the same prompt with an empty article."""
    suffix = 0
    while suffix < len(empty) and ids[-1 - suffix] == empty[-1 - suffix]:
        suffix += 1
    prefix = 0
    while prefix < len(empty) - suffix and ids[prefix] == empty[prefix]:
        prefix += 1
    return prefix, len(ids) - suffix


def offset_share(x: np.ndarray) -> float:
    """Share of the per-item label score changes (n, labels) explained by one constant offset per label."""
    c = x - x.mean(axis=1, keepdims=True)
    total = (c**2).sum()
    return float(len(c) * (c.mean(axis=0) ** 2).sum() / total) if total else 0.0


def cell_path(model_tag: str, k: int, label: str, rule: str, rotated: int, suite: str = "", start: int = START) -> Path:
    rot = {0: "", 1: "_rot"}.get(int(rotated), f"_rot{int(rotated)}")
    pre = f"clsprobe_{suite}_" if suite else "clsprobe_"
    at = "" if start == START else f"_d{start}"
    if k == 0:
        return RESULTS / f"{pre}{model_tag}_k0{rot}{at}.jsonl"
    return RESULTS / f"{pre}{model_tag}_k{k}_{label}_{rule}{rot}{at}.jsonl"


def cmd_run(a: argparse.Namespace) -> None:
    from mlx_lm import load

    import segroute as sr
    from models import MODELS

    model, tok = load(MODELS[a.model_tag])
    ctl = sl.install(model, lambda: sl.TAIL, "ends")
    emotion = a.suite == "emotion"
    labels = EMOTIONS if emotion else LABELS
    first = {lab: tok.encode(lab, add_special_tokens=False)[0] for lab in labels}
    items = emotion_items(a.start) if emotion else [x for x in short.items(N, a.start) if x[0] == "classify"]
    orders = [int(x) for x in a.orders.split(",")]
    spellings = {
        lab: [tok.encode(lab, add_special_tokens=False), tok.encode(" " + lab, add_special_tokens=False)]
        for lab in labels
    }
    cells = [(0, "", "", rot) for rot in orders]
    cells += [(a.k, a.label, rule, rot) for rule in a.rules.split(",") for rot in orders]
    for k, label, rule, rot in cells:
        path = cell_path(a.model_tag, k, label, rule, rot, a.suite, a.start)
        if path.exists() and sum(1 for _ in path.open()) == N:
            continue
        ctl.mode, ctl.k, ctl.s = ("mask", k, 32) if k else ("off", None, None)
        ctl.rule, ctl.reserve_m = rule_parts(rule or "ends")
        with path.open("w") as f:
            for _, i, prompt, ref in items:
                text = rotate_list(prompt, labels, rot)
                ids = sr.chat(tok, text)
                ctl.reset()
                ctl.span = article_span(ids, sr.chat(tok, text.split("\n\n", 1)[0] + "\n\n"))
                ctl.reserved = list(find_spans(ids, spellings).values()) if ctl.reserve_m else []
                logits = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
                lp = logits - mx.logsumexp(logits)
                mx.eval(lp)
                scores = {lab: lp[t].item() for lab, t in first.items()}
                f.write(json.dumps({"i": i, "ref": ref, "lp": scores, "top": mx.argmax(lp).item()}) + "\n")
        print(path.name, flush=True)


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open()]


def _pred(r: dict) -> str:
    return max(LABELS, key=lambda lab: r["lp"][lab])


def _right(r: dict, labels: tuple[str, ...] = LABELS) -> bool:
    """Whether the best-scoring label is the dataset label: stored as `hit`, or compared with `ref` in a new run."""
    if "hit" in r:
        return bool(r["hit"])
    return max(labels, key=lambda lab: r["lp"][lab]) == r["ref"]


def _stats(recs: list[dict], base: list[dict]) -> dict[str, Any]:
    acc = float(np.mean([_right(r) for r in recs]))
    base_acc = float(np.mean([_right(r) for r in base]))
    x = np.array([[r["lp"][lab] - b["lp"][lab] for lab in LABELS] for r, b in zip(recs, base, strict=True)])
    return {
        "accuracy": acc,
        "kept": acc / base_acc,
        "share": {lab: float(np.mean([_pred(r) == lab for r in recs])) for lab in LABELS},
        "offset_share": offset_share(x),
        "offset": dict(zip(LABELS, (x - x.mean(axis=1, keepdims=True)).mean(axis=0).tolist(), strict=True)),
    }


def _kept(recs: list[dict], base: list[dict], labels: tuple[str, ...]) -> dict[str, float]:
    def right(rows: list[dict]) -> dict[int, float]:
        return {r["i"]: float(_right(r, labels)) for r in rows}

    return sl.kept_ci(right(recs), right(base))


def report_h40() -> dict:
    """Write results/clsprobe_h40.json: untrained frontier budgets in four label orders, emotion in two."""
    orders = (0, 1, 2, 3)
    models: dict[str, Any] = {}
    for m, k in FRONTIER.items():
        base = {o: _read(cell_path(m, 0, "", "", o)) for o in orders}
        recs = {o: _read(cell_path(m, k, "untrained", "ends", o)) for o in orders}
        cells = [_kept(recs[o], base[o], LABELS) for o in orders]
        emotion = [
            _kept(
                _read(cell_path(m, k, "untrained", "ends", o, "emotion")),
                _read(cell_path(m, 0, "", "", o, "emotion")),
                EMOTIONS,
            )
            for o in (0, 1)
        ]
        models[m] = {
            "k": k,
            "orders": cells,
            **order_summary([d["mean"] for d in cells]),
            "shares": [_stats(recs[o], base[o])["share"] for o in orders],
            "emotion": emotion,
        }
    out = {**h40_verdict({m: r["orders"] for m, r in models.items()}), "models": models}
    (RESULTS / "clsprobe_h40.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True)
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--rules", default="ends,article")
    r.add_argument("--orders", default="0,1")
    r.add_argument("--suite", default="", choices=["", "emotion"])
    r.add_argument("--start", type=int, default=START, help="first item of each source; 300 or 600")
    sub.add_parser("report-h40")
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        reports = {"report-h40": report_h40}
        print(json.dumps(reports[a.cmd](), indent=1))


if __name__ == "__main__":
    main()
