"""Reports of the first paper's why tests (H43, H45 to H48, T3, H52), from stored runs only.

Gates as written on 2026-11-02 23:40, before any run. Writes results/why_report.json.
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import json
from pathlib import Path
from typing import Any

import numpy as np

import copyaudit
import failures
import mathrule
import promptlen as pl
import selrule as sl
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = ("gemma", "nemotron", "qwen36")
BUDGETS = {"gemma": 24, "nemotron": 16, "qwen36": 24}
FRONTIER = {"gemma": (24, 48, 64), "nemotron": (16, 48, 64), "qwen36": (24, 96, 128)}
H47_MARGIN = 0.2
H43_RISE, H43_FLAT, GATE = 0.10, 0.02, 0.5
ARMS = ("short", "mid", "long")
VALID = 0.8
EQUAL_SHARE_K = 12  # 9.4% of 128 experts, the Qwen3.6 share at K′ = 24


def h47_margin(rows: list[dict[str, float]]) -> float:
    """Mean over layers of same-token cosine minus the other-token floor."""
    return float(np.mean([r["same_token"] - r["other_token_floor"] for r in rows]))


def h47_verdict(models: dict[str, list[dict[str, float]]]) -> str:
    return "holds" if all(h47_margin(r) > H47_MARGIN for r in models.values()) else "fails"


def h48_verdict(models: dict[str, dict[str, float]]) -> str:
    n = sum(r["across_tasks_same_type"] > r["across_types_same_task"] for r in models.values())
    return "holds" if n >= 2 else "fails"


def h45_verdict(models: dict[str, dict[str, dict[str, float]]]) -> str:
    n = sum(sl.at_bar([r["qa_f1"], r["cls_acc"]]) for r in models.values())
    return "holds" if n >= 2 else "fails"


def longest_valid(acc: dict[str, float]) -> str | None:
    """Longest of mid and long whose unmasked accuracy is at least 80%."""
    return next((a for a in ("long", "mid") if acc.get(a, 0.0) >= VALID), None)


def h43_holds(masked_rise: float, unmasked_rise: float, gate: bool) -> bool:
    return gate and masked_rise > H43_RISE and unmasked_rise < H43_FLAT


def t3_verdict(diffs: list[dict[str, float]]) -> str:
    return "holds" if sum(d["lo"] > 0 for d in diffs) >= 2 else "fails"


def h55_verdict(models: dict[str, list[dict[str, float]]]) -> str:
    """H55: ends64 minus the fixed set, paired CI above 0 on 2 of 3 tasks, on 2 of 3 models."""
    return "holds" if sum(sum(d["lo"] > 0 for d in diffs) >= 2 for diffs in models.values()) >= 2 else "fails"


def h60_verdict(gemma_diffs: list[dict[str, float]]) -> str:
    """H60: on Gemma 4 at K′ = 12, ends64 beats the fixed list, paired CI above 0, on 2 of 3 tasks."""
    return "holds" if sum(d["lo"] > 0 for d in gemma_diffs) >= 2 else "fails"


def h56_specific(routed: dict[str, float], always_on: dict[str, float], measures: list[str]) -> bool:
    """H56: the masked rise under the routed cut is under half the always-on rise on every measure that held in H43."""
    return all(routed[q] < 0.5 * always_on[q] for q in measures)


def _ok(path: Path) -> dict[int, float]:
    return {r["i"]: float(r["ok"]) for r in sr_read(path)}


def sr_read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prompts(start: int) -> dict[int, str]:
    return {i: p for t, i, p, _ in short.items(300, start) if t == "summarize"}


def _suite(tag: str, k: int) -> list[dict]:
    return short.read_records(short.run_path(tag, 300, k, 32))


def _kept_row(recs: list[dict], base: dict, teacher: dict) -> dict[str, Any]:
    scores = short.item_metrics(recs, teacher)
    return {m: sl.kept_ci(scores[m], base[m]) for m in sl.KEPT_METRICS}


def report_h45() -> dict:
    models: dict[str, Any] = {}
    for m, k in BUDGETS.items():
        base_recs = _suite(f"{m}_d300", 0)
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        rand, ends = _suite(f"{m}_d300_random", k), _suite(f"{m}_d300_ends64", k)
        gsm_base = _ok(mathrule.base_path(m))
        p = _prompts(300)
        models[m] = {
            "k": k,
            "random": {
                **_kept_row(rand, base, teacher),
                "gsm8k": sl.kept_ci(_ok(RESULTS / f"{m}_random_gsm8k_k{k}_s32.jsonl"), gsm_base),
                **copyaudit._row(rand, p),
            },
            "ends64": {
                **_kept_row(ends, base, teacher),
                "gsm8k": sl.kept_ci(_ok(mathrule.run_path(m, k)), gsm_base),
                **copyaudit._row(ends, p),
            },
        }
    return {"verdict": h45_verdict({m: r["random"] for m, r in models.items()}), "models": models}


def report_h46() -> dict:
    models: dict[str, Any] = {}
    for m, k in BUDGETS.items():
        acc, kept = {}, {}
        for arm in ARMS:
            off = _ok(RESULTS / f"arith_{m}_{arm}_off_k{k}.jsonl")
            acc[arm] = float(np.mean(list(off.values())))
            kept[arm] = {
                r: sl.kept_ci(_ok(RESULTS / f"arith_{m}_{arm}_{r}_k{k}.jsonl"), off) for r in ("ends", "random")
            }
        arm = longest_valid(acc)
        reading = {
            r: mathrule_reading(kept["short"][r], kept[arm][r]) if arm and acc["short"] >= VALID else "invalid"
            for r in ("ends", "random")
        }
        models[m] = {"k": k, "unmasked_accuracy": acc, "kept": kept, "longest_valid": arm, "reading": reading}
    return {"models": models}


def mathrule_reading(short_kept: dict[str, float], long_kept: dict[str, float]) -> str:
    import curriculum

    return curriculum.h46_reading(short_kept, long_kept)


def report_h47_h48() -> dict:
    overlap = {m: json.loads((RESULTS / f"curriculum_overlap_{m}.json").read_text()) for m in MODELS}
    routers = {m: json.loads((RESULTS / f"curriculum_routers_{m}.json").read_text()) for m in MODELS}
    return {
        "h47": {"verdict": h47_verdict(overlap), "margin": {m: h47_margin(r) for m, r in overlap.items()}},
        "h48": {"verdict": h48_verdict(routers), "models": routers},
    }


def _sum_rates(recs: list[dict]) -> dict[str, float]:
    sm = [r for r in recs if r["task"] == "summarize"]
    return {
        "loops": float(np.mean([failures.is_loop(r["gen_ids"]) for r in sm])),
        "at_cap": float(np.mean([len(r["gen_ids"]) >= short.MAX["summarize"] for r in sm])),
    }


def report_h43() -> dict:
    models: dict[str, Any] = {}
    for m, k in BUDGETS.items():
        intact_u, intact_m = _suite(m, 0), _suite(f"{m}_ends64", k)
        abl_u, abl_m = _suite(f"{m}_a0.75_ends64", 0), _suite(f"{m}_a0.75_ends64", k)
        teacher = short.teacher_outputs(intact_u)
        base = short.item_metrics(intact_u, teacher)
        abl_base = short.item_metrics(abl_u, teacher)
        gate = {
            t: float(np.mean(list(abl_base[t].values())) / np.mean(list(base[t].values()))) for t in sl.KEPT_METRICS
        }
        rates = {
            a: _sum_rates(r)
            for a, r in (("intact_u", intact_u), ("intact_m", intact_m), ("abl_u", abl_u), ("abl_m", abl_m))
        }
        rise = {
            q: {
                "masked": rates["abl_m"][q] - rates["intact_m"][q],
                "unmasked": rates["abl_u"][q] - rates["intact_u"][q],
            }
            for q in ("loops", "at_cap")
        }
        ok = gate["sum_rougeL_ref"] >= GATE
        models[m] = {
            "k": k,
            "gate": gate,
            "rates": rates,
            "rise": rise,
            "kept_intact": _kept_row(intact_m, base, teacher),
            "kept_ablated": _kept_row(abl_m, abl_base, teacher),
            "holds": any(h43_holds(r["masked"], r["unmasked"], ok) for r in rise.values()),
        }
    return {"held_on": [m for m, r in models.items() if r["holds"]], "models": models}


def report_t3() -> dict:
    base_recs = _suite("gemma_d300", 0)
    teacher = short.teacher_outputs(base_recs)
    base = short.item_metrics(base_recs, teacher)
    exp, full = _suite("gemma_d300_exposure13", 16), _suite("gemma_d300_ends64", 16)
    se, sf = short.item_metrics(exp, teacher), short.item_metrics(full, teacher)
    diffs = {t: pl.paired_diff(se[t], sf[t]) for t in sl.KEPT_METRICS}
    p = _prompts(300)
    return {
        "verdict": t3_verdict(list(diffs.values())),
        "paired_diff_raw": diffs,
        "exposure13": {**_kept_row(exp, base, teacher), **copyaudit._row(exp, p)},
        "all_layers": {**_kept_row(full, base, teacher), **copyaudit._row(full, p)},
    }


def _fixed_vs_rule(m: str, k: int) -> dict[str, Any]:
    """Fixed calibration list against `ends64` at the same K′, items 300 to 599."""
    base_recs = _suite(f"{m}_d300", 0)
    teacher = short.teacher_outputs(base_recs)
    base = short.item_metrics(base_recs, teacher)
    fixed, rule = _suite(f"{m}_d300_static", k), _suite(f"{m}_d300_ends64", k)
    sx, sr = short.item_metrics(fixed, teacher), short.item_metrics(rule, teacher)
    p = _prompts(300)
    return {
        "k": k,
        "paired_diff_raw": {t: pl.paired_diff(sr[t], sx[t]) for t in sl.KEPT_METRICS},
        "fixed": {**_kept_row(fixed, base, teacher), **_sum_rates(fixed), **copyaudit._row(fixed, p)},
        "ends64": {**_kept_row(rule, base, teacher), **_sum_rates(rule), **copyaudit._row(rule, p)},
    }


def report_h55() -> dict:
    models = {m: _fixed_vs_rule(m, k) for m, k in BUDGETS.items()}
    return {
        "verdict": h55_verdict({m: list(r["paired_diff_raw"].values()) for m, r in models.items()}),
        "models": models,
    }


def report_h60() -> dict:
    models = {m: _fixed_vs_rule(m, EQUAL_SHARE_K) for m in ("gemma", "nemotron")}
    return {"verdict": h60_verdict(list(models["gemma"]["paired_diff_raw"].values())), "models": models}


def report_h56(h43: dict) -> dict:
    models: dict[str, Any] = {}
    for m in ("gemma", "qwen36"):
        k = BUDGETS[m]
        intact_u = _suite(m, 0)
        cut_u, cut_m = _suite(f"{m}_r0.75_ends64", 0), _suite(f"{m}_r0.75_ends64", k)
        teacher = short.teacher_outputs(intact_u)
        base, cut_base = short.item_metrics(intact_u, teacher), short.item_metrics(cut_u, teacher)
        rates = {a: _sum_rates(r) for a, r in (("cut_u", cut_u), ("cut_m", cut_m))}
        h = h43["models"][m]
        routed = {q: rates["cut_m"][q] - h["rates"]["intact_m"][q] for q in ("loops", "at_cap")}
        always = {q: h["rise"][q]["masked"] for q in ("loops", "at_cap")}
        held = [
            q
            for q, r in h["rise"].items()
            if h43_holds(r["masked"], r["unmasked"], h["gate"]["sum_rougeL_ref"] >= GATE)
        ]
        models[m] = {
            "k": k,
            "gate": {
                t: float(np.mean(list(cut_base[t].values())) / np.mean(list(base[t].values()))) for t in sl.KEPT_METRICS
            },
            "rates": rates,
            "routed_rise_masked": routed,
            "routed_rise_unmasked": {q: rates["cut_u"][q] - h["rates"]["intact_u"][q] for q in ("loops", "at_cap")},
            "always_on_rise_masked": always,
            "measures": held,
            "kept_cut": _kept_row(cut_m, cut_base, teacher),
            "specific": h56_specific(routed, always, held),
        }
    return {"specific_on": [m for m, r in models.items() if r["specific"]], "models": models}


def report_h52() -> dict:
    """ROUGE-L kept on summarize items 300 to 399 at larger budgets."""
    out: dict[str, Any] = {}
    for m, ks in FRONTIER.items():
        base_recs = [r for r in _suite(f"{m}_d300", 0) if r["task"] == "summarize" and r["i"] < 400]
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)["sum_rougeL_ref"]
        out[m] = {}
        for k in ks:
            path = RESULTS / f"frontier_{m}_k{k}.jsonl"
            recs = (
                sr_read(path)
                if path.exists()
                else [r for r in _suite(f"{m}_d300_ends64", k) if r["task"] == "summarize" and r["i"] < 400]
            )
            out[m][k] = {
                "rougeL": sl.kept_ci(short.item_metrics(recs, teacher)["sum_rougeL_ref"], base),
                **copyaudit._row(recs, _prompts(300)),
            }
    return out


def main() -> None:
    out = {
        "h45": report_h45(),
        "h46": report_h46(),
        **report_h47_h48(),
        "h43": (h43 := report_h43()),
        "h55": report_h55(),
        "h56": report_h56(h43),
        **({"h60": report_h60()} if (RESULTS / "gemma_d300_static_short_n300_k12_s32.jsonl").exists() else {}),
        "t3": report_t3(),
        "h52": report_h52(),
    }
    (RESULTS / "why_report.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
