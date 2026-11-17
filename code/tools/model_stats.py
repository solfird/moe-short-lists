"""Routed-parameter share and Nemotron correction-bias ranges for one model tag.

Run from the repo root as a module, which puts the root on sys.path for the models import:
python -m tools.model_stats nemotron
"""

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

from models import MODELS

ROUTED_MARKERS = ("switch_mlp", "experts.switch_glu")  # Qwen, Qwen3.6, Nemotron | Gemma 4
BIAS_SUFFIX = "e_score_correction_bias"


def routed_share(params: list[tuple[str, mx.array]], markers: tuple[str, ...]) -> tuple[int, int]:
    """Return (routed bytes, total bytes) of a flat parameter list; routed means the name holds a marker."""
    routed = sum(v.nbytes for k, v in params if any(m in k for m in markers))
    if routed == 0:
        raise ValueError(f"no parameter matches {markers}")
    return routed, sum(v.nbytes for _, v in params)


def bias_ranges(params: list[tuple[str, mx.array]]) -> list[tuple[str, float, float, float]]:
    """Return (name, min, max, spread) of every e_score_correction_bias."""
    out = []
    for k, v in params:
        if k.endswith(BIAS_SUFFIX):
            lo, hi = v.min().item(), v.max().item()
            out.append((k, lo, hi, hi - lo))
    return out


def main() -> None:
    """Print the routed share and, when present, the per-layer correction-bias range."""
    p = argparse.ArgumentParser()
    p.add_argument("tag", choices=list(MODELS))
    tag = p.parse_args().tag
    params = tree_flatten(load(MODELS[tag])[0].parameters())
    routed, total = routed_share(params, ROUTED_MARKERS)
    print(f"{tag}: total {total / 1e9:.2f} GB, routed {routed / 1e9:.2f} GB, share {routed / total:.3f}")
    biases = bias_ranges(params)
    for name, lo, hi, spread in biases:
        print(f"{name} min {lo:.2f} max {hi:.2f} spread {spread:.2f}")
    if biases:
        print(f"bias layers {len(biases)} min {min(b[1] for b in biases):.2f} max {max(b[2] for b in biases):.2f}")


if __name__ == "__main__":
    main()
