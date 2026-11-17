"""Remove dataset text from the per-item records and keep the scores that were computed from it.

    python analysis/strip_dataset_text.py SOURCE_RESULTS_DIR DEST_RESULTS_DIR

This is how the records of `results/` were made from the raw files of the experiments. It is kept to document the
rewrite; the raw files are not in this repository. Every file under SOURCE is copied to DEST, and a record that carries
dataset text is rewritten:

- short-suite and summary records (`ref`): replaced by `ref_scores`, the scores of the answer against the reference,
  computed with the same functions as `short.item_metrics`;
- label probes (`ref`): replaced by `hit`, whether the best-scoring label is the dataset label;
- GSM8K records (`gold`): removed, `ok` and `pred` stay;
- jury and panel items (`article`): replaced by `source` (dataset, config, split, index) and `copied`, the share of the
  answer copied from the article;
- human-label items (`article`, `answer`): removed, `human` (the label) stays;
- batch folders written for the Claude juror (they hold articles): not copied.

The answers of the models under test (`out`, `text`, `answer` of an item that is a model answer) stay.
"""

import json
import pathlib
import shutil
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import clsprobe
import copyaudit
import corpus
import short

SKIP_DIRS = {"claude_batches", "claude_blind_batches", "claude_parts"}
CNN = "abisee/cnn_dailymail"
XSUM = "EdinburghNLP/xsum"


def _dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _replace_key(row: dict, old: str, new: dict) -> dict:
    """Same key order, with `old` replaced in place by the pairs of `new`."""
    out: dict = {}
    for k, v in row.items():
        if k == old:
            out.update(new)
        else:
            out[k] = v
    return out


def rewrite_text(name: str, text: str) -> str:
    """The rewritten content of one jsonl file, chosen by its name and by the keys of its first record."""
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    if not rows:
        return text
    first = rows[0]
    if name.startswith("clsprobe_") and "ref" in first:
        labels = clsprobe.EMOTIONS if name.startswith("clsprobe_emotion_") else clsprobe.LABELS
        return _dump(
            [_replace_key(r, "ref", {"hit": max(labels, key=lambda lab: r["lp"][lab]) == r["ref"]}) for r in rows]
        )
    if "task" in first and "ref" in first:
        return _dump([_replace_key(r, "ref", {"ref_scores": short.reference_scores(r)}) for r in rows])
    if "gsm8k" in name and "gold" in first and not name.startswith("arith_"):
        return _dump([{k: v for k, v in r.items() if k != "gold"} for r in rows])
    if "article" in first and "human" in first:
        return _dump([{k: v for k, v in r.items() if k not in ("article", "answer")} for r in rows])
    if "article" in first and "cell" in first:
        xsum = "xsum" in name or any("xsum" in str(r.get("cell", "")) for r in rows[:1]) or name == "items.jsonl-xsum"
        return _dump([_item(r, xsum) for r in rows])
    return text


def _item(r: dict, xsum: bool) -> dict:
    src = corpus.source(XSUM, r["i"]) if xsum else corpus.source(CNN, r["i"], "3.0.0")
    copied = copyaudit.copied_share(r["answer"], r["article"])
    out = {k: v for k, v in r.items() if k != "article"}
    return {**out, "source": src, "copied": copied}


def main(source: pathlib.Path, dest: pathlib.Path) -> None:
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if path.is_dir() or SKIP_DIRS & set(rel.parts):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            name = path.name
            if name == "items.jsonl" and "xsum" in rel.parts[0]:
                name = "items.jsonl-xsum"
            target.write_text(rewrite_text(name, path.read_text()))
        else:
            shutil.copy2(path, target)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
