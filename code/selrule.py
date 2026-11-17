"""Prompt set from the end of the prompt (H26, H27), and two padded arms that separate length from content.

selrule.py run --model-tag gemma --arm sel --frac 0.125 --n 200     # sel, tail64: masked only; rep, aft: also frac 0
selrule.py report-h26 --n 200                                       # results/selrule_h26.json
selrule.py report-h27 --n 300                                       # results/selrule_h27.json
selrule.py report-h28 --n 300                                       # results/selrule_h28.json
selrule.py report-h29 --n 300                                       # results/selrule_h29.json
selrule.py run --model-tag gemma --arm ends64 --frac 0.125 --n 300 --start 300   # items 300 to 599; whole: also frac 0
selrule.py report-h30 --n 300                                       # results/selrule_h30.json
selrule.py report-h31 --n 300                                       # results/selrule_h31.json
selrule.py report-h31-repeat --n 300                                # results/selrule_h31_repeat.json
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import ablate
import failures
import promptlen as pl
import segroute as sr
import short
from adapters import SOFTMAX_LINEAR, GateTap, find_routers, n_experts

RESULTS = Path(__file__).resolve().parents[1] / "results"
PADDED_ARMS = ("sel", "rep", "aft")
TAIL = 64
TAIL_ARM = f"tail{TAIL}"
HEAD_ARM = f"head{TAIL}"
ENDS_ARM = f"ends{TAIL}"
RULES = {TAIL_ARM: "tail", HEAD_ARM: "head", ENDS_ARM: "ends"}
BUDGETS = {"gemma": 16, "nemotron": 16, "qwen36": 32}  # 1/8 of the experts
KEPT_METRICS = ("sum_rougeL_ref", "qa_f1", "cls_acc")
WHOLE_ARM = "whole"  # whole-prompt rule on disjoint items; the first set has its stored runs
DISJOINT_START = 300
FRACTIONS = (1 / 16, 3 / 32, 1 / 8, 3 / 16, 1 / 4)
MAX_LOOP = 0.05  # clean bar, H31
REPEAT_CELLS = {"gemma": (24,), "qwen36": (24, 48)}  # H31 frontiers not yet run on the disjoint items


def reserve_bonus(probs: Any, spans: list[tuple[int, int]], m: int) -> Any:
    """A bonus large enough to put the top m experts of each token span in the set, ahead of any summed weight."""
    bonus = mx.zeros(probs.shape[1], dtype=mx.float32)
    for a, b in spans:
        top = mx.argsort(-probs[a:b].sum(axis=0))[:m]
        bonus = bonus + (mx.arange(probs.shape[1])[None, :] == top[:, None]).any(axis=0).astype(mx.float32) * 1e6
    return bonus


class TailController(sr.Controller):
    """Prompt set from tail() prompt tokens: the last, the first, or half from each end. None: all of them."""

    def __init__(self, n_layers: int, n_experts: int, tail: Callable[[], int | None], rule: str = "tail") -> None:
        if rule not in (*RULES.values(), "article"):
            raise ValueError(f"unknown rule {rule}")
        self.tail, self.rule = tail, rule
        self.span: tuple[int, int] = (0, 0)  # the article's token range, for the "article" rule
        self.reserved: list[tuple[int, int]] = []  # token spans that each keep reserve_m experts in the set
        self.reserve_m = 0
        super().__init__(n_layers, n_experts)

    def observe(self, layer: int, probs: Any) -> Any:
        if self.mode == "mask" and self.bias[layer] is None:
            t = self.tail()
            weight = self._chosen(probs, t).sum(axis=0)
            if self.reserve_m:
                weight = weight + reserve_bonus(probs, self.reserved, self.reserve_m)
            self._select(layer, weight)
            return self.bias[layer]
        return super().observe(layer, probs)

    def _chosen(self, probs: Any, t: int | None) -> Any:
        if self.rule == "article":
            probs = probs[self.span[0] : self.span[1]]
        if t is None or probs.shape[0] <= t:
            return probs
        if self.rule == "head":
            return probs[:t]
        if self.rule == "tail":
            return probs[-t:]
        return mx.concatenate([probs[: t // 2], probs[-(t - t // 2) :]], axis=0)


def install(model: nn.Module, tail: Callable[[], int | None], rule: str = "tail") -> TailController:
    """segroute.install with a TailController."""
    targets = find_routers(model)
    ctl = TailController(len(targets), n_experts(*targets[0]), tail, rule)
    for idx, (owner, name, kind) in enumerate(targets):
        router = getattr(owner, name)
        setattr(owner, name, sr.Tap(router, idx, ctl) if kind == SOFTMAX_LINEAR else GateTap(router, idx, ctl))
    return ctl


def repeat_item(item: pl.Item, size: int = pl.FULL) -> pl.Item:
    """Same length as the H25 filler, made by repeating the news item itself, before the unchanged prompt."""
    task, i, prompt, ref = item
    text = prompt.split("\n\n", 1)[1]
    return task, i, (text * (size // len(text) + 1))[:size] + "\n\n" + prompt, ref


def after_item(item: pl.Item, article: str, size: int = pl.FULL) -> pl.Item:
    """The H25 filler after the prompt instead of before it."""
    task, i, prompt, ref = item
    return task, i, prompt + "\n\n" + article[:size], ref


def tag(model_tag: str, arm: str, start: int = 0) -> str:
    """Output tag of one arm."""
    prefix = f"{model_tag}_d{start}" if start else model_tag
    if arm in RULES:
        return f"{prefix}_{arm}"
    if arm == WHOLE_ARM and start:
        return prefix
    if arm not in PADDED_ARMS:
        raise ValueError(f"unknown arm {arm}")
    return f"{model_tag}_pl_{arm}"


def contributes(d: dict[str, float]) -> bool:
    """H26: first-token agreement rises by at least 0.20, with a paired CI above 0."""
    return d["mean"] >= 0.20 and d["lo"] > 0


def h26_clauses(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """H26 clauses, each on both models."""
    selection = all(contributes(r["sel"]) for r in models.values())
    content = all(contributes(r["rep"]) for r in models.values())
    close = all(r["sel_first"] >= r["plain_first"] - 0.10 for r in models.values())
    verdict = {(True, True): "both", (True, False): "selection", (False, True): "content", (False, False): "fails"}
    return {
        "selection": selection,
        "content": content,
        "suffices": selection and close,
        "verdict": verdict[selection, content],
    }


def h27_verdict(models: dict[str, list[dict[str, float]]]) -> str:
    """H27: kept up 5 points or more, paired CI above 0, on 2 of 3 tasks, on both models (intact minus arm)."""

    def up(d: dict[str, float]) -> bool:
        return d["mean"] <= -0.05 and d["hi"] < 0

    return "holds" if all(sum(up(d) for d in row) >= 2 for row in models.values()) else "fails"


def _arm_items(arm: str, n: int) -> list[pl.Item]:
    from datasets import load_dataset

    if arm == "sel":
        return pl.arm_items("pad", n)
    plain = [x for x in short.items(n) if x[0] == "classify"]
    if arm == "rep":
        return [repeat_item(x) for x in plain]
    cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    fillers = cnn.select(short.article_slices(n)["rewrite"])["article"]
    return [after_item(x, a) for x, a in zip(plain, fillers, strict=True)]


def cmd_run(a: argparse.Namespace) -> None:
    """One arm through short.cmd_run, the path of every other run."""
    if (a.arm == "sel" or a.arm in RULES) and a.frac == 0:
        raise SystemExit(f"{a.arm} changes the mask only: its unmasked run already exists")
    start = getattr(a, "start", 0)
    orig_items, orig_chat, orig_install = short.items, sr.chat, sr.install
    if start:
        short.items = lambda n: orig_items(n, start)
    state: dict[str, int | None] = {"tail": TAIL if a.arm in RULES else None}
    plain: dict[str, str] = {}
    if a.arm == "sel":
        built = _arm_items("sel", a.n)
        originals = [x for x in orig_items(a.n) if x[0] == "classify"]
        plain = {x[2]: y[2] for x, y in zip(built, originals, strict=True)}
    if a.arm in PADDED_ARMS:
        built = built if a.arm == "sel" else _arm_items(a.arm, a.n)
        short.items = lambda n: built

    def chat(tok: Any, text: str) -> list[int]:
        if text in plain:
            state["tail"] = len(orig_chat(tok, plain[text]))
        return orig_chat(tok, text)

    sr.chat = chat
    sr.install = lambda model: install(model, lambda: state["tail"], RULES.get(a.arm, "tail"))
    try:
        short.cmd_run(
            argparse.Namespace(
                model_tag=a.model_tag,
                tag=tag(a.model_tag, a.arm, start),
                n=a.n,
                frac=a.frac,
                s=32,
            )
        )
    finally:
        short.items, sr.chat, sr.install = orig_items, orig_chat, orig_install


def _accuracy(path: Path) -> dict[int, float]:
    return {r["i"]: short.reference_scores(r)["cls_acc"] for r in short.read_records(path) if r["task"] == "classify"}


def kept_ci(arm: dict[Any, float], base: dict[Any, float]) -> dict[str, float]:
    """Mean of arm over mean of base on matched items, with a 95% bootstrap CI."""
    keys = sorted(arm.keys() & base.keys())
    a, b = np.array([arm[k] for k in keys]), np.array([base[k] for k in keys])
    idx = np.random.default_rng(0).integers(0, len(keys), size=(2000, len(keys)))
    lo, hi = np.percentile(a[idx].mean(axis=1) / b[idx].mean(axis=1), [2.5, 97.5])
    return {"mean": float(a.mean() / b.mean()), "lo": float(lo), "hi": float(hi)}


def at_bar(row: list[dict[str, float]]) -> bool:
    """Every task at 90% kept, or with a CI that includes 90%."""
    return all(d["mean"] >= 0.90 or d["lo"] <= 0.90 <= d["hi"] for d in row)


def h29_verdict(bars: dict[str, bool]) -> str:
    """H29: the rule is at the bar on at least 2 of the 3 models."""
    return "holds" if sum(bars.values()) >= 2 else "fails"


def h30_verdict(models: dict[str, dict[str, Any]]) -> str:
    """H30: at the bar on at least 2 of 3 models, and each of them above the whole-prompt rule on at least one task."""
    at = [r for r in models.values() if r["at_bar"]]
    return "holds" if len(at) >= 2 and all(any(d["lo"] > 0 for d in r["diffs"]) for r in at) else "fails"


def loop_rate(recs: list[dict], task: str) -> float:
    """Share of one task's answers that end in a repeated block."""
    return float(np.mean([failures.is_loop(r["gen_ids"]) for r in recs if r["task"] == task]))


def frontier(rows: dict[int, dict[str, bool]], key: str) -> int | None:
    """Smallest budget where key is true, or None."""
    return min((k for k, r in rows.items() if r[key]), default=None)


def h31_verdict(front: dict[str, int | None], experts: dict[str, int]) -> str:
    """H31: Gemma 4 at the bar at 3/16 or below, and at least one model at 3/32 or below."""
    gemma = front["gemma"]
    low = any(k is not None and k <= experts[m] * 3 / 32 for m, k in front.items())
    return "holds" if gemma is not None and gemma <= experts["gemma"] * 3 / 16 and low else "fails"


def report_h26(n: int) -> dict:
    """Write results/selrule_h26.json: each arm against its own unmasked run, on the padded items."""
    out: dict[str, Any] = {}
    for model, k in pl.MODELS.items():
        stored = RESULTS / f"{model}_short_n300_k{{}}_s32.jsonl"
        plain_paths = Path(str(stored).format(k)), Path(str(stored).format(0))
        pad_paths = short.run_path(pl.tag(model, "pad"), n, k, 32), short.run_path(pl.tag(model, "pad"), n, 0, 32)
        paths = {
            "pad": pad_paths,
            "sel": (short.run_path(tag(model, "sel"), n, k, 32), pad_paths[1]),
            "rep": (short.run_path(tag(model, "rep"), n, k, 32), short.run_path(tag(model, "rep"), n, 0, 32)),
            "aft": (short.run_path(tag(model, "aft"), n, k, 32), short.run_path(tag(model, "aft"), n, 0, 32)),
        }
        scores = {a: pl._scores(*p, "classify") for a, p in paths.items()}
        plain = pl._scores(*plain_paths, "classify")
        ids = set(scores["pad"]["first"])
        plain_first = float(np.mean([v for i, v in plain["first"].items() if i in ids]))
        plain_unmasked = float(np.mean([v for i, v in _accuracy(plain_paths[1]).items() if i in ids]))
        row: dict[str, Any] = {
            "plain_first": plain_first,
            "plain_unmasked_accuracy": plain_unmasked,
            "sel": pl.paired_diff(scores["sel"]["first"], scores["pad"]["first"]),
            "rep": pl.paired_diff(scores["rep"]["first"], scores["pad"]["first"]),
            "aft": pl.paired_diff(scores["aft"]["first"], scores["pad"]["first"]),
            "sel_first": float(np.mean(list(scores["sel"]["first"].values()))),
            "arms": {},
        }
        for a, (masked, unmasked) in paths.items():
            m, u = float(np.mean(list(_accuracy(masked).values()))), float(np.mean(list(_accuracy(unmasked).values())))
            row["arms"][a] = {
                "prompt_tokens": scores[a]["prompt_tokens"],
                "first": float(np.mean(list(scores[a]["first"].values()))),
                "lcs": float(np.mean(list(scores[a]["lcs"].values()))),
                "masked_accuracy": m,
                "unmasked_accuracy": u,
                "kept": m / u if u else float("nan"),
            }
        row["aft_interpretable"] = row["arms"]["aft"]["unmasked_accuracy"] >= 0.5 * plain_unmasked
        out[model] = row
    result = {**h26_clauses(out), "models": out}
    (RESULTS / "selrule_h26.json").write_text(json.dumps(result, indent=1))
    return result


def report_h27(n: int) -> dict:
    """Write results/selrule_h27.json: the tail rule against the stored whole-prompt run on the same items."""
    out: dict[str, Any] = {}
    rows: dict[str, list[dict[str, float]]] = {}
    for model, k in pl.MODELS.items():
        base_recs = short.read_records(short.run_path(model, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        whole = short.item_metrics(short.read_records(short.run_path(model, n, k, 32)), teacher)
        arm = short.item_metrics(short.read_records(short.run_path(tag(model, TAIL_ARM), n, k, 32)), teacher)
        rows[model] = [ablate.kept_difference(whole[m], base[m], arm[m], base[m]) for m in KEPT_METRICS]
        out[model] = dict(zip(KEPT_METRICS, rows[model], strict=True))
    result = {"verdict": h27_verdict(rows), "models": out}
    (RESULTS / "selrule_h27.json").write_text(json.dumps(result, indent=1))
    return result


def report_h28(n: int) -> dict:
    """Write results/selrule_h28.json: the head rule against the tail rule and the whole prompt, on the same items."""
    out: dict[str, Any] = {}
    rows: dict[str, list[dict[str, float]]] = {}
    for model, k in pl.MODELS.items():
        base_recs = short.read_records(short.run_path(model, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        whole = short.item_metrics(short.read_records(short.run_path(model, n, k, 32)), teacher)
        tail = short.item_metrics(short.read_records(short.run_path(tag(model, TAIL_ARM), n, k, 32)), teacher)
        head = short.item_metrics(short.read_records(short.run_path(tag(model, HEAD_ARM), n, k, 32)), teacher)
        rows[model] = [ablate.kept_difference(tail[m], base[m], head[m], base[m]) for m in KEPT_METRICS]
        out[model] = {
            "tail_minus_head": dict(zip(KEPT_METRICS, rows[model], strict=True)),
            "whole_minus_head": {m: ablate.kept_difference(whole[m], base[m], head[m], base[m]) for m in KEPT_METRICS},
        }
    result = {"verdict": h27_verdict(rows), "models": out}
    (RESULTS / "selrule_h28.json").write_text(json.dumps(result, indent=1))
    return result


def report_h29(n: int) -> dict:
    """Write results/selrule_h29.json: kept per rule and model, with the whole-prompt run beside them."""
    out: dict[str, Any] = {}
    bars: dict[str, bool] = {}
    for model, k in BUDGETS.items():
        base_recs = short.read_records(short.run_path(model, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        tags = {"whole": model, **{arm: tag(model, arm) for arm in RULES}}
        out[model] = {}
        for arm, name in tags.items():
            recs = short.read_records(short.run_path(name, n, k, 32))
            scores = short.item_metrics(recs, teacher)
            row = [kept_ci(scores[m], base[m]) for m in KEPT_METRICS]
            out[model][arm] = {
                **dict(zip(KEPT_METRICS, row, strict=True)),
                "at_bar": at_bar(row),
                "loads_per_layer": float(np.mean([r["loads_per_layer"] for r in recs])),
            }
        bars[model] = out[model][ENDS_ARM]["at_bar"]
    result = {"verdict": h29_verdict(bars), "ends_at_bar": bars, "models": out}
    (RESULTS / "selrule_h29.json").write_text(json.dumps(result, indent=1))
    return result


def report_h30(n: int) -> dict:
    """Write results/selrule_h30.json: ends64 and the whole-prompt rule on the disjoint items."""
    out: dict[str, Any] = {}
    for model, k in BUDGETS.items():
        whole, ends = tag(model, WHOLE_ARM, DISJOINT_START), tag(model, ENDS_ARM, DISJOINT_START)
        base_recs = short.read_records(short.run_path(whole, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        recs = {
            arm: short.read_records(short.run_path(name, n, k, 32))
            for arm, name in (("whole", whole), ("ends64", ends))
        }
        scores = {arm: short.item_metrics(r, teacher) for arm, r in recs.items()}
        row = [kept_ci(scores["ends64"][m], base[m]) for m in KEPT_METRICS]
        out[model] = {
            "ends64": dict(zip(KEPT_METRICS, row, strict=True)),
            "whole": {m: kept_ci(scores["whole"][m], base[m]) for m in KEPT_METRICS},
            "unmasked": {m: float(np.mean(list(base[m].values()))) for m in KEPT_METRICS},
            "at_bar": at_bar(row),
            "diffs": [pl.paired_diff(scores["ends64"][m], scores["whole"][m]) for m in KEPT_METRICS],
            "loads_per_layer": {arm: float(np.mean([x["loads_per_layer"] for x in r])) for arm, r in recs.items()},
        }
    result = {"verdict": h30_verdict(out), "models": out}
    (RESULTS / "selrule_h30.json").write_text(json.dumps(result, indent=1))
    return result


def report_h31(n: int) -> dict:
    """Write results/selrule_h31.json: the both-ends rule at each budget, on the development items."""
    out: dict[str, Any] = {}
    experts = {model: short.N_EXPERTS[model] for model in BUDGETS}
    for model in BUDGETS:
        base_recs = short.read_records(short.run_path(model, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        out[model] = {}
        for frac in FRACTIONS:
            k = short.budget(experts[model], frac)
            recs = short.read_records(short.run_path(tag(model, ENDS_ARM), n, k, 32))
            scores = short.item_metrics(recs, teacher)
            row = [kept_ci(scores[m], base[m]) for m in KEPT_METRICS]
            loops = {t: loop_rate(recs, t) for t in ("summarize", "rewrite")}
            out[model][k] = {
                **dict(zip(KEPT_METRICS, row, strict=True)),
                "at_bar": at_bar(row),
                "clean": at_bar(row) and max(loops.values()) <= MAX_LOOP,
                "loops": loops,
                "loads_per_layer": float(np.mean([r["loads_per_layer"] for r in recs])),
            }
    fronts = {m: frontier(r, "at_bar") for m, r in out.items()}
    result = {
        "verdict": h31_verdict(fronts, experts),
        "frontier": fronts,
        "clean_frontier": {m: frontier(r, "clean") for m, r in out.items()},
        "models": out,
    }
    (RESULTS / "selrule_h31.json").write_text(json.dumps(result, indent=1))
    return result


def report_h31_repeat(n: int) -> dict:
    """Write results/selrule_h31_repeat.json: the H31 frontiers on the disjoint items, with the 1/8 runs of H30."""
    out: dict[str, Any] = {}
    cells = {m: sorted({BUDGETS[m], *REPEAT_CELLS.get(m, ())}) for m in BUDGETS}
    for model, budgets in cells.items():
        base_recs = short.read_records(short.run_path(tag(model, WHOLE_ARM, DISJOINT_START), n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        out[model] = {}
        for k in budgets:
            recs = short.read_records(short.run_path(tag(model, ENDS_ARM, DISJOINT_START), n, k, 32))
            scores = short.item_metrics(recs, teacher)
            row = [kept_ci(scores[m], base[m]) for m in KEPT_METRICS]
            loops = {t: loop_rate(recs, t) for t in ("summarize", "rewrite")}
            out[model][k] = {
                **dict(zip(KEPT_METRICS, row, strict=True)),
                "at_bar": at_bar(row),
                "clean": at_bar(row) and max(loops.values()) <= MAX_LOOP,
                "loops": loops,
                "loads_per_layer": float(np.mean([r["loads_per_layer"] for r in recs])),
            }
    (RESULTS / "selrule_h31_repeat.json").write_text(json.dumps(out, indent=1))
    return out


def identical_share(a: list[dict], b: list[dict]) -> float:
    """Share of items of a with the same generated tokens in b."""
    other = {(r["task"], r["i"]): r["gen_ids"] for r in b}
    return float(np.mean([r["gen_ids"] == other[(r["task"], r["i"])] for r in a]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True, choices=list(BUDGETS))
    r.add_argument("--arm", required=True, choices=[*PADDED_ARMS, *RULES, WHOLE_ARM])
    r.add_argument("--frac", type=float, required=True)
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--start", type=int, default=0, help="position of the first item of each source")
    r.set_defaults(fn=cmd_run)
    for name, fn in (
        ("report-h26", report_h26),
        ("report-h27", report_h27),
        ("report-h28", report_h28),
        ("report-h29", report_h29),
        ("report-h30", report_h30),
        ("report-h31", report_h31),
        ("report-h31-repeat", report_h31_repeat),
    ):
        s = sub.add_parser(name)
        s.add_argument("--n", type=int, required=True)
        s.set_defaults(fn=lambda a, f=fn: print(json.dumps(f(a.n), indent=1)))
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
