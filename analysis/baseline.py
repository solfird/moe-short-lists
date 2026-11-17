"""Copying the article opening against the summarize bar (H50), and how the jury depends on its members (H51).

baseline.py   # results/baseline_h50.json
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import corpus
import jury
import selrule as sl
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
QUESTIONS = ("supported", "condensed", "covers")
INSTRUCTION = "Summarize this news article in 2 or 3 sentences.\n\n"
UNMASKED_D300 = {"gemma": "gemma_d300", "nemotron": "nemotron_d300", "qwen36": "qwen36_d300"}
UNMASKED_D0 = {"gemma": "gemma", "nemotron": "nemotron", "qwen36": "qwen36"}


def article_of(prompt: str) -> str:
    return prompt.removeprefix(INSTRUCTION)


def lead_sentences(article: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", article.strip())
    return " ".join(parts[:n])


def lead_words(article: str, n: int) -> str:
    return " ".join(article.split()[:n])


def fleiss_kappa(ratings: list[list[bool]]) -> float | None:
    """Fleiss kappa for yes or no votes, same number of raters per item. None when all votes match."""
    votes = np.array(ratings, dtype=float)
    raters = votes.shape[1]
    yes = votes.sum(axis=1)
    agree = (yes * (yes - 1) + (raters - yes) * (raters - yes - 1)) / (raters * (raters - 1))
    share = votes.mean()
    chance = share**2 + (1 - share) ** 2
    if chance == 1:
        return None
    return float((agree.mean() - chance) / (1 - chance))


def accepted(votes: list[dict[str, bool]], rule: str) -> bool:
    """Accepted under a rule: per-question majority, or every juror yes on every question."""
    if rule == "unanimous":
        return all(all(v[q] for q in QUESTIONS) for v in votes)
    return all(sum(v[q] for v in votes) * 2 > len(votes) for q in QUESTIONS)


def point_biserial(scores: list[float], flags: list[bool]) -> float | None:
    """Pearson correlation of a score with a yes or no flag. None when either has no spread."""
    x, y = np.array(scores, dtype=float), np.array(flags, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _numbers(text: str) -> list[str]:
    return [x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def foreign_numbers(answer: str, article: str) -> tuple[list[str], int]:
    """Digit strings of the answer absent from the article, and how many the answer writes."""
    known = set(_numbers(article))
    found = _numbers(answer)
    return [x for x in found if x not in known], len(found)


def foreign_names(answer: str, article: str) -> list[str]:
    """Capitalized words of the answer, not first in a sentence, absent from the article."""
    known = set(re.findall(r"\w+", article))
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
        for word in re.findall(r"\w+", sentence)[1:]:
            if word[0].isupper() and word not in known:
                out.append(word)
    return out


def _lcs_table(a: list[str], b: list[str]) -> list[list[int]]:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
    return table


def lcs_length(a: list[str], b: list[str]) -> int:
    return _lcs_table(a, b)[len(a)][len(b)]


def _lcs_positions(a: list[str], b: list[str]) -> set[int]:
    """Positions in a of one longest common subsequence with b."""
    table = _lcs_table(a, b)
    i, j, out = len(a), len(b), set()
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.add(i - 1)
            i, j = i - 1, j - 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def rouge_l_union(ref: str, answer: str) -> float:
    """ROUGE-L with the union LCS per reference sentence, as ROUGE-1.5.5 does it, F-measure."""
    from rouge_score import tokenizers

    tok = tokenizers.DefaultTokenizer(use_stemmer=True).tokenize
    answers = [tok(s) for s in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()) if s.strip()]
    n_ans = sum(map(len, answers))
    hit, n_ref = 0, 0
    for sent in ref.split("\n"):
        words = tok(sent)
        n_ref += len(words)
        matched: set[int] = set()
        for other in answers:
            matched |= _lcs_positions(words, other)
        hit += len(matched)
    if not hit:
        return 0.0
    pr, rc = hit / n_ans, hit / n_ref
    return min(1.0, 2 * pr * rc / (pr + rc))


def _rouge() -> Any:
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def lead_report() -> dict[str, Any]:
    """Lead-3 and Lead-90 kept against each unmasked model, on both item sets."""
    rs = _rouge()
    out: dict[str, Any] = {}
    for start, tags in ((300, UNMASKED_D300), (0, UNMASKED_D0)):
        rows = [x for x in short.items(300, start) if x[0] == "summarize"]
        articles = {i: (article_of(p), ref) for _, i, p, ref in rows}
        arms = {
            "lead3": {i: rs.score(r, lead_sentences(a, 3))["rougeL"].fmeasure for i, (a, r) in articles.items()},
            "lead90": {i: rs.score(r, lead_words(a, 90))["rougeL"].fmeasure for i, (a, r) in articles.items()},
        }
        cell: dict[str, Any] = {name: float(np.mean(list(v.values()))) for name, v in arms.items()}
        for m, tag in tags.items():
            recs = short.read_records(short.run_path(tag, 300, 0, 32))
            base = {r["i"]: short.reference_scores(r, rs)["sum_rougeL_ref"] for r in recs if r["task"] == "summarize"}
            cell[m] = {
                "unmasked": float(np.mean(list(base.values()))),
                **{name: sl.kept_ci(v, base) for name, v in arms.items()},
            }
        out[f"start{start}"] = cell
    bars = [sl.at_bar([out["start300"][m]["lead3"]]) for m in UNMASKED_D300]
    out["verdict"] = "holds" if sum(bars) >= 2 else "fails"
    return out


def jury_report(round_name: str) -> dict[str, Any]:
    """Agreement per question, accepted kept under each rule and juror set, ROUGE-L against accepted per cell."""
    jury.use_round(round_name)
    items = jury._read(jury.ITEMS)
    votes = jury._votes()
    jurors = list(votes)
    complete = [it for it in items if all(it["id"] in votes[j] for j in jurors)]
    out: dict[str, Any] = {"n": len(complete)}
    out["kappa"] = {q: fleiss_kappa([[votes[j][it["id"]][q] for j in jurors] for it in complete]) for q in QUESTIONS}
    variants: dict[str, tuple[list[str], str]] = {"majority": (jurors, "majority"), "unanimous": (jurors, "unanimous")}
    for left_out in jurors:
        variants[f"without_{left_out}"] = ([j for j in jurors if j != left_out], "unanimous")
    rs = _rouge()
    refs = {i: r for t, i, _, r in short.items(300, jury.START) if t == "summarize"}
    cells: dict[str, Any] = {}
    for m, k in jury.CELLS.items():
        arm = [it for it in complete if it["cell"] == f"{m}_k{k}"]
        base = [it for it in complete if it["cell"] == f"{m}_k0"]
        row: dict[str, Any] = {"k": k}
        for name, (who, rule) in variants.items():
            row[name] = sl.kept_ci(
                {it["i"]: float(accepted([votes[j][it["id"]] for j in who], rule)) for it in arm},
                {it["i"]: float(accepted([votes[j][it["id"]] for j in who], rule)) for it in base},
            )
        scores = [rs.score(refs[it["i"]], it["answer"])["rougeL"].fmeasure for it in arm]
        flags = [accepted([votes[j][it["id"]] for j in jurors], "majority") for it in arm]
        row["rouge_accepted_r"] = point_biserial(scores, flags)
        row["rouge_accepted"] = float(np.mean([s for s, f in zip(scores, flags, strict=True) if f]))
        row["rouge_rejected"] = float(np.mean([s for s, f in zip(scores, flags, strict=True) if not f]))
        cells[m] = row
    out["cells"] = cells
    return out


def variants_report() -> dict[str, Any]:
    """H53: other overlap scores on Lead-3 and on the first-budget cells, items 300 to 599."""
    from rouge_score import rouge_scorer

    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3", "rougeL"], use_stemmer=True)

    def scores(ref: str, text: str) -> dict[str, float]:
        got = {k: v.fmeasure for k, v in rs.score(ref, text).items() if k != "rougeL"}
        return {**got, "rougeL_union": rouge_l_union(ref, text)}

    rows = {i: (article_of(p), ref) for t, i, p, ref in short.items(300, 300) if t == "summarize"}
    lead = {i: scores(r, lead_sentences(a, 3)) for i, (a, r) in rows.items()}
    jury.use_round("a")
    items = jury._read(jury.ITEMS)
    votes = jury._votes()
    jurors = list(votes)
    verdict = {
        (it["cell"], it["i"]): accepted([votes[j][it["id"]] for j in jurors], "majority")
        for it in items
        if all(it["id"] in votes[j] for j in jurors)
    }
    out: dict[str, Any] = {}
    names = ("rouge1", "rouge2", "rouge3", "rougeL_union")
    for m, k in jury.CELLS.items():
        answers = {
            key: {
                r["i"]: r["out"]
                for r in short.read_records(short.run_path(tag, 300, kk, 32))
                if r["task"] == "summarize"
            }
            for key, tag, kk in (("base", UNMASKED_D300[m], 0), ("arm", sl.tag(m, sl.ENDS_ARM, 300), k))
        }
        base = {i: scores(rows[i][1], t) for i, t in answers["base"].items()}
        arm = {i: scores(rows[i][1], t) for i, t in answers["arm"].items()}
        teacher = {i: rs.score(answers["base"][i], t)["rougeL"].fmeasure for i, t in answers["arm"].items()}
        row: dict[str, Any] = {"k": k, "rougeL_to_unmasked": float(np.mean(list(teacher.values())))}
        judged = sorted(i for i in arm if (f"{m}_k{k}", i) in verdict)
        for name in names:
            field = {i: v[name] for i, v in base.items()}
            row[name] = {
                "unmasked": float(np.mean(list(field.values()))),
                "lead3_kept": sl.kept_ci({i: v[name] for i, v in lead.items()}, field),
                "masked_kept": sl.kept_ci({i: v[name] for i, v in arm.items()}, field),
                "r_accepted": point_biserial(
                    [arm[i][name] for i in judged], [verdict[(f"{m}_k{k}", i)] for i in judged]
                ),
            }
        row["r_accepted_rougeL_to_unmasked"] = point_biserial(
            [teacher[i] for i in judged], [verdict[(f"{m}_k{k}", i)] for i in judged]
        )
        out[m] = row
    bars = [sum(sl.at_bar([out[m][name]["lead3_kept"]]) for m in jury.CELLS) >= 2 for name in names]
    out["verdict"] = "holds" if all(bars) else "fails"
    return out


def numbers_report() -> dict[str, Any]:
    """H54: foreign numbers and names, first budgets against unmasked, items 300 to 599."""
    jury.use_round("a")
    items = jury._read(jury.ITEMS)
    votes = jury._votes()
    jurors = list(votes)
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {}
    bars = []
    for m, k in jury.CELLS.items():
        rows: dict[str, dict[int, dict[str, Any]]] = {"arm": {}, "base": {}}
        for it in items:
            side = "arm" if it["cell"] == f"{m}_k{k}" else "base" if it["cell"] == f"{m}_k0" else None
            if side is None or not all(it["id"] in votes[j] for j in jurors):
                continue
            foreign, written = foreign_numbers(it["answer"], corpus.article_of(it))
            names = foreign_names(it["answer"], corpus.article_of(it))
            supported = sum(votes[j][it["id"]]["supported"] for j in jurors) >= 2
            rows[side][it["i"]] = {
                "num": bool(foreign),
                "n_foreign": len(foreign),
                "written": written,
                "name": bool(names),
                "supported": supported,
            }
        keys = sorted(rows["arm"].keys() & rows["base"].keys())
        a, b = rows["arm"], rows["base"]
        diff = np.array([float(a[i]["num"]) - float(b[i]["num"]) for i in keys])
        boot = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        share = {s: float(np.mean([rows[s][i]["num"] for i in keys])) for s in rows}
        bars.append(share["arm"] >= 2 * share["base"] and lo > 0)
        unsupported = [i for i in keys if not a[i]["supported"]]
        out[m] = {
            "k": k,
            "n": len(keys),
            "foreign_number_answers": share,
            "difference_ci": [float(lo), float(hi)],
            "foreign_per_number_written": {
                s: sum(rows[s][i]["n_foreign"] for i in keys) / max(1, sum(rows[s][i]["written"] for i in keys))
                for s in rows
            },
            "foreign_name_answers": {s: float(np.mean([rows[s][i]["name"] for i in keys])) for s in rows},
            "r_number_supported": point_biserial([float(a[i]["num"]) for i in keys], [a[i]["supported"] for i in keys]),
            "r_name_supported": point_biserial([float(a[i]["name"]) for i in keys], [a[i]["supported"] for i in keys]),
            "unsupported_share": len(unsupported) / len(keys),
            "unsupported_with_neither": float(np.mean([not a[i]["num"] and not a[i]["name"] for i in unsupported])),
        }
    out["verdict"] = "holds" if all(bars) else "fails"
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--claude", default="claude", choices=("claude", "claude_blind"))
    jury.CLAUDE = p.parse_args().claude
    out = {
        "h54": numbers_report(),
        "h50": lead_report(),
        "h51": {r: jury_report(r) for r in ("a", "b")},
        "h53": variants_report(),
    }
    (RESULTS / jury.out_name("baseline_h50.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
