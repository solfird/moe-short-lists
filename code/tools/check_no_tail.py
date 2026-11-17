"""Does TailController with no tail reproduce the stored whole-prompt answers? 20 items per task. Check, not a test.

python -m tools.check_no_tail   # results/check_no_tail.json
"""

import json
from pathlib import Path

from mlx_lm import load

import segroute as sr
import selrule as sl
import short
from models import MODELS

RESULTS = Path(__file__).resolve().parents[2] / "results"
PER_TASK = 20


def main() -> None:
    out = {}
    for model_tag, k in sl.BUDGETS.items():
        stored = {(r["task"], r["i"]): r["gen_ids"] for r in short.read_records(short.run_path(model_tag, 300, k, 32))}
        model, tok, config = load(MODELS[model_tag], return_config=True)
        ctl = sl.install(model, lambda: None)
        ctl.mode, ctl.k, ctl.s = "mask", k, 32
        same = total = 0
        seen: dict[str, int] = {}
        for task, i, prompt, _ in short.items(300):
            if seen.get(task, 0) == PER_TASK:
                continue
            seen[task] = seen.get(task, 0) + 1
            ctl.reset()
            gen = sr.generate(model, tok, sr.chat(tok, prompt), short.MAX[task], config)
            same += gen == stored[(task, i)]
            total += 1
        out[model_tag] = {"identical": same, "items": total}
        print(model_tag, out[model_tag], flush=True)
        del model
    (RESULTS / "check_no_tail.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
