"""Two programs with no model, on the confirmation items: the first sentence of the context for question answering,
and the most common label for classify. Each is scored like a model and set against the full model's score.

    python analysis/trivial_baselines.py [--datasets-cache DIR]   # results/trivial_baselines.json

Needs the SQuAD contexts and AG News labels of items 300 to 599: the dataset text is fetched from the Hugging Face Hub.
"""

import collections
import json
import pathlib
import re
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import numpy as np

import selrule as sl
import short

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "trivial_baselines.json"
START = 300
TAGS = {"gemma": "gemma_d300", "nemotron": "nemotron_d300", "qwen36": "qwen36_d300"}


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip())[0]


def report() -> dict:
    items = short.items(300, START)
    qa = {i: (p.split("Text: ", 1)[1].split("\n\nQuestion:", 1)[0], ref) for t, i, p, ref in items if t == "qa"}
    labels = {i: ref for t, i, _, ref in items if t == "classify"}
    common, _ = collections.Counter(labels.values()).most_common(1)[0]
    qa_arm = {i: max(short.f1(first_sentence(ctx), g) for g in golds) for i, (ctx, golds) in qa.items()}
    cls_arm = {i: float(common == ref) for i, ref in labels.items()}
    out = {}
    for m, tag in TAGS.items():
        recs = short.read_records(short.run_path(tag, 300, 0, 32))
        scores = short.item_metrics(recs, short.teacher_outputs(recs))
        qa_base = {i: v for (t, i), v in scores["qa_f1"].items()}
        cls_base = {i: v for (t, i), v in scores["cls_acc"].items()}
        out[m] = {
            "qa_unmasked": float(np.mean(list(qa_base.values()))),
            "qa_first_sentence_kept": sl.kept_ci(qa_arm, qa_base),
            "cls_unmasked": float(np.mean(list(cls_base.values()))),
            "cls_majority_kept": sl.kept_ci(cls_arm, cls_base),
        }
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
