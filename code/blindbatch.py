"""Claude juror batches in which no article appears twice, so no batch shows two answers to the same article.

The first Claude batches (rounds a, b and c) held up to four answers to one article; the API jurors saw one answer per
call. These batches restore per-answer blindness across articles. New verdicts go to claude_blind.jsonl; the old
claude.jsonl stays as it was.

blindbatch.py build --dir results/jury          # <dir>/claude_blind_batches/batch_###.json
blindbatch.py merge --dir results/jury          # <dir>/claude_blind.jsonl from batch_###.jsonl verdict files
"""

import argparse
import collections
import json
import math
from pathlib import Path

import corpus
import jury

SIZE = 50


def distinct_batches(rows: list[dict], size: int) -> list[list[dict]]:
    """Fewest batches of at most size rows with no article (key "i") twice in a batch; row order kept inside."""
    by_article: dict[int, list[int]] = collections.defaultdict(list)
    for pos, r in enumerate(rows):
        by_article[r["i"]].append(pos)
    n = max(math.ceil(len(rows) / size), max(len(v) for v in by_article.values()))
    slots: list[list[int]] = [[] for _ in range(n)]
    for _, positions in sorted(by_article.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        free = sorted((b for b in range(n) if len(slots[b]) < size), key=lambda b: (len(slots[b]), b))
        if len(free) < len(positions):
            raise ValueError("distinct_batches: not enough room; raise the size")
        for b, pos in zip(free, positions, strict=False):
            slots[b].append(pos)
    return [[rows[p] for p in sorted(s)] for s in slots if s]


def merge(batch_dir: Path, out_dir: Path) -> tuple[list[dict], list[tuple]]:
    """Verdicts in batch order; problems list batches whose ids differ and answers that do not parse."""
    out: list[dict] = []
    problems: list[tuple] = []
    for b in sorted(batch_dir.glob("batch_*.json")):
        ids = [x["id"] for x in json.loads(b.read_text())]
        got = [json.loads(line) for line in (out_dir / f"{b.stem}.jsonl").read_text().splitlines() if line.strip()]
        if [r["id"] for r in got] != ids:
            problems.append((b.stem, "ids differ"))
            continue
        for r in got:
            v = jury.parse_verdict(json.dumps({q: r.get(q) for q in jury.QUESTIONS}))
            if v is None:
                problems.append((b.stem, r["id"]))
            out.append({"id": r["id"], "verdict": v})
    return out, problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("build", "merge"))
    p.add_argument("--dir", required=True, type=Path)
    a = p.parse_args()
    bdir, odir = a.dir / "claude_blind_batches", a.dir / "claude_blind_out"
    if a.cmd == "build":
        rows = [
            {"id": r["id"], "i": r["i"], "article": corpus.article_of(r), "answer": r["answer"]}
            for r in jury._read(a.dir / "items.jsonl")
        ]
        bdir.mkdir(exist_ok=True)
        odir.mkdir(exist_ok=True)
        batches = distinct_batches(rows, SIZE)
        for n, b in enumerate(batches):
            (bdir / f"batch_{n:03d}.json").write_text(
                json.dumps([{k: r[k] for k in ("id", "article", "answer")} for r in b], indent=0)
            )
        print(len(batches))
    else:
        out, problems = merge(bdir, odir)
        print("verdicts", len(out), "problems", problems)
        if not problems:
            (a.dir / "claude_blind.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))


if __name__ == "__main__":
    main()
