"""Scale Gemma 4's always-on dense path inside the unchanged MoE model, to test it as a cause of mask tolerance (H16).

In a Gemma 4 MoE layer, h = post_norm(post_norm_1(mlp(h)) + post_norm_2(experts(h))). Multiplying the post_norm_1 output
by alpha removes the dense path at alpha 0 and leaves the routed path and the shared post norm in place.

    ablate.py run --alpha 0.5 --frac 0.25 --s 64     # frac 0 = unmasked; output tag gemma_a0.50
    ablate.py compare --alpha 0.5 --k 32 --s 64
"""

import argparse
import json
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import short
from records import read_records


class ScaledOutput(nn.Module):
    """Multiplies the output of a module by a constant."""

    def __init__(self, inner: nn.Module, alpha: float) -> None:
        super().__init__()
        self.inner = inner
        self.alpha = alpha

    def __call__(self, x: mx.array) -> mx.array:
        return self.inner(x) * self.alpha


def _gemma_layers(model: nn.Module) -> list[nn.Module]:
    root = getattr(model, "language_model", model)
    layers = [x for x in getattr(getattr(root, "model", None), "layers", []) if getattr(x, "enable_moe", False)]
    if not layers:
        raise ValueError("scale_always_on supports Gemma 4 MoE layers only")
    return layers


def scale_always_on(model: nn.Module, alpha: float) -> int:
    """Scale the dense path of every MoE layer by alpha in [0, 1]. Returns the number of layers changed."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], not {alpha}")
    layers = _gemma_layers(model)
    if alpha == 1.0:
        return len(layers)
    for layer in layers:
        if isinstance(layer.post_feedforward_layernorm_1, ScaledOutput):
            raise ValueError("the dense path is already scaled")
        layer.post_feedforward_layernorm_1 = ScaledOutput(layer.post_feedforward_layernorm_1, alpha)
    return len(layers)


def tag(alpha: float) -> str:
    """Output tag of one arm."""
    return f"gemma_a{alpha:.2f}"


def interpretable(unmasked: float, intact_unmasked: float) -> bool:
    """A task stays interpretable at an alpha while the unmasked arm holds half of the intact unmasked score."""
    return unmasked >= 0.5 * intact_unmasked


def kept_difference(
    mask_intact: dict[Any, float], base_intact: dict[Any, float], mask_arm: dict[Any, float], base_arm: dict[Any, float]
) -> dict[str, float]:
    """Kept (intact) minus kept (arm) on matched items, with a paired 95% bootstrap CI. Positive: the arm keeps less."""
    keys = sorted(mask_intact.keys() & base_intact.keys() & mask_arm.keys() & base_arm.keys())
    if not keys:
        raise ValueError("kept_difference: no matched items")
    cols = np.array([[d[k] for k in keys] for d in (mask_intact, base_intact, mask_arm, base_arm)])

    def diff(c: np.ndarray) -> np.ndarray:
        return c[0] / c[1] - c[2] / c[3]

    idx = np.random.default_rng(0).integers(0, len(keys), size=(2000, len(keys)))
    means = cols[:, idx].mean(axis=2)
    ok = (means[1] > 0) & (means[3] > 0)
    if not ok.any():  # the arm scores zero on every item: nothing to keep
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(diff(means[:, ok]), [2.5, 97.5])
    point = diff(cols.mean(axis=1)) if cols[1].mean() > 0 and cols[3].mean() > 0 else float("nan")
    return {"mean": float(point), "lo": float(lo), "hi": float(hi)}


def cmd_run(a: argparse.Namespace) -> None:
    """One arm through short.cmd_run, the path of every other run, with the dense path scaled after loading."""
    original = short.load

    def scaled_load(*args: Any, **kw: Any) -> Any:
        out = original(*args, **kw)
        print("layers scaled:", scale_always_on(out[0], a.alpha))
        return out

    short.load = scaled_load
    try:
        short.cmd_run(argparse.Namespace(model_tag="gemma", tag=tag(a.alpha), n=a.n, frac=a.frac, s=a.s))
    finally:
        short.load = original


def compare(alpha: float, k: int, s: int, n: int = 300) -> dict[str, dict]:
    """Per task: unmasked and masked means of the arm, the interpretability gate, and the kept difference to intact."""

    def metrics(name: str, kk: int) -> dict[str, dict]:
        base = read_records(short.run_path(name, n, 0, 32))
        return short.item_metrics(read_records(short.run_path(name, n, kk, s)), short.teacher_outputs(base))

    def unmasked(name: str) -> dict[str, dict]:
        base = read_records(short.run_path(name, n, 0, 32))
        return short.item_metrics(base, short.teacher_outputs(base))

    arm = tag(alpha)
    mi, bi, ma, ba = metrics("gemma", k), unmasked("gemma"), metrics(arm, k), unmasked(arm)
    out = {}
    for m in ("sum_rougeL_ref", "qa_f1", "cls_acc"):
        intact_mean, arm_mean = float(np.mean(list(bi[m].values()))), float(np.mean(list(ba[m].values())))
        out[m] = {
            "unmasked_intact": intact_mean,
            "unmasked_arm": arm_mean,
            "interpretable": interpretable(arm_mean, intact_mean),
            "kept_difference": kept_difference(mi[m], bi[m], ma[m], ba[m]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--frac", type=float, required=True)
    r.add_argument("--s", type=int, default=64)
    r.add_argument("--n", type=int, default=300)
    c = sub.add_parser("compare")
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--k", type=int, required=True)
    c.add_argument("--s", type=int, default=64)
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        print(json.dumps(compare(a.alpha, a.k, a.s), indent=1))


if __name__ == "__main__":
    main()
