"""H52 and the jury column of H49: Claude and one local model judge all cells on items 300 to 399.

Items hold article text, so the item files stay in a private folder. Only aggregates reach results/.

jury_c.py build            # private/jury_c/items.jsonl and Claude batches
jury_c.py local            # local juror, resumable
jury_c.py report           # results/jury_h52.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import jury
import selrule as sl
import short

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "private" / "jury_c"
RESULTS = ROOT / "results"
START, N = 300, 100
FIRST = {"gemma": 24, "nemotron": 16, "qwen36": 24}
LARGER = {"gemma": (48, 64), "nemotron": (48, 64), "qwen36": (96, 128)}
LOCAL = "mlx-community/Ministral-3-8B-Instruct-2512-4bit"
BATCH = 50


def panel_cells() -> list[tuple[str, int]]:
    out = []
    for m, k in FIRST.items():
        out += [(m, 0), (m, k), *[(m, x) for x in LARGER[m]]]
    return out


def accepted_by_both(a: dict[str, bool], b: dict[str, bool]) -> bool:
    return all(a[q] and b[q] for q in jury.QUESTIONS)


def smallest_at_bar(kept: dict[str, dict[str, float]]) -> int | None:
    good = [int(k) for k, v in kept.items() if sl.at_bar([v])]
    return min(good) if good else None


def agreement(panel: dict[int, bool], jury3: dict[int, bool]) -> float:
    both = panel.keys() & jury3.keys()
    return float(np.mean([panel[i] == jury3[i] for i in both]))


def _answers(m: str, k: int) -> dict[int, str]:
    if k in LARGER[m]:
        rows = [json.loads(x) for x in (RESULTS / f"frontier_{m}_k{k}.jsonl").open()]
    else:
        tag = sl.tag(m, sl.ENDS_ARM if k else sl.WHOLE_ARM, START)
        rows = [r for r in short.read_records(short.run_path(tag, 300, k, 32)) if r["task"] == "summarize"]
    return {r["i"]: r["out"] for r in rows if START <= r["i"] < START + N}


def build() -> None:
    articles = {i: p.split("\n\n", 1)[1] for t, i, p, _ in short.items(N, START) if t == "summarize"}
    rows = []
    for m, k in panel_cells():
        for i, text in _answers(m, k).items():
            rows.append({"cell": f"{m}_k{k}", "i": i, "article": articles[i], "answer": text})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rows = jury.shuffled(rows)
    with (PRIVATE / "items.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = PRIVATE / "claude_batches"
    out.mkdir(exist_ok=True)
    for n, s in enumerate(range(0, len(rows), BATCH)):
        part = [{k: r[k] for k in ("id", "article", "answer")} for r in rows[s : s + BATCH]]
        (out / f"batch_{n:03d}.json").write_text(json.dumps(part, indent=0))
    print(len(rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open()] if path.exists() else []


def local() -> None:
    from mlx_lm import generate, load

    path = PRIVATE / "local.jsonl"
    done = {r["id"] for r in _read(path)}
    model, tok = load(LOCAL, tokenizer_config={"fix_mistral_regex": True})
    with path.open("a") as f:
        for it in _read(PRIVATE / "items.jsonl"):
            if it["id"] in done:
                continue
            msgs = [{"role": "user", "content": f"{jury.SYSTEM}\n\n{jury.juror_prompt(it)}"}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
            text = generate(model, tok, prompt=prompt, max_tokens=40)
            f.write(json.dumps({"id": it["id"], "raw": text, "verdict": jury.parse_verdict(text)}) + "\n")
            f.flush()


def report() -> dict[str, Any]:
    items = _read(PRIVATE / "items.jsonl")
    votes = {
        j: {
            r["id"]: r["verdict"]
            for r in _read(PRIVATE / f"{jury.CLAUDE if j == 'claude' else j}.jsonl")
            if r.get("verdict")
        }
        for j in ("claude", "local")
    }
    flags: dict[str, dict[int, bool]] = {}
    per_juror: dict[str, dict[str, list[bool]]] = {"claude": {}, "local": {}}
    for it in items:
        if it["id"] in votes["claude"] and it["id"] in votes["local"]:
            a, b = votes["claude"][it["id"]], votes["local"][it["id"]]
            flags.setdefault(it["cell"], {})[it["i"]] = accepted_by_both(a, b)
            per_juror["claude"].setdefault(it["cell"], []).append(all(a.values()))
            per_juror["local"].setdefault(it["cell"], []).append(all(b.values()))
    out: dict[str, Any] = {"n_judged": sum(map(len, flags.values())), "models": {}}
    for m in FIRST:
        base = {i: float(v) for i, v in flags[f"{m}_k0"].items()}
        kept = {
            str(k): sl.kept_ci({i: float(v) for i, v in flags[f"{m}_k{k}"].items()}, base)
            for k in (FIRST[m], *LARGER[m])
        }
        out["models"][m] = {
            "unmasked_accepted": float(np.mean(list(base.values()))),
            "accepted": {str(k): float(np.mean(list(flags[f"{m}_k{k}"].values()))) for k in (FIRST[m], *LARGER[m])},
            "kept": kept,
            "smallest_at_bar": smallest_at_bar({k: v for k, v in kept.items() if int(k) in LARGER[m]}),
        }
    out["per_juror_unmasked"] = {j: {m: float(np.mean(v[f"{m}_k0"])) for m in FIRST} for j, v in per_juror.items()}
    jury.use_round("a")
    votes3 = jury._votes()
    h44 = {}
    for it in jury._read(jury.ITEMS):
        if START <= it["i"] < START + N and all(it["id"] in votes3[j] for j in votes3):
            h44.setdefault(it["cell"], {})[it["i"]] = jury.majority([votes3[j][it["id"]] for j in votes3])["accepted"]
    out["agreement_with_h44"] = {c: agreement(flags[c], v) for c, v in h44.items() if c in flags}
    (RESULTS / jury.out_name("jury_h52.json")).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "local":
        local()
    else:
        if sys.argv[2:] == ["blind"]:
            jury.CLAUDE = "claude_blind"
        print(json.dumps(report(), indent=1))
