"""tools/router_stats.py: router flatness and top-k weight kept by the C1 segment sets, from unmasked router logs."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tools.router_stats
from segroute import replay
from tools.router_stats import TOP_K, flatness, join_quality, segment_kept


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def random_probs(layers=3, tokens=150, experts=32, seed=0):
    return softmax(np.random.default_rng(seed).normal(size=(layers, tokens, experts)) * 2).astype(np.float32)


@pytest.mark.parametrize("k,s", [(4, 32), (8, 16), (12, 7)])
def test_segment_kept_matches_the_reference_replay(k, s):
    probs, n_prompt = random_probs(), 40
    mass, hit = segment_kept(probs, n_prompt, k, s, top_k=8)
    ref_mass, ref_hit, _, _ = replay(probs, n_prompt, k, s)
    assert mass == pytest.approx(ref_mass, abs=1e-6)
    assert hit == pytest.approx(ref_hit, abs=1e-6)


def test_segment_kept_full_set_keeps_everything():
    probs = random_probs()
    assert segment_kept(probs, 40, 32, 32, top_k=6) == (pytest.approx(1.0), pytest.approx(1.0))


def test_segment_kept_uses_the_model_top_k():
    probs = np.zeros((1, 3, 4), dtype=np.float32)
    probs[0, 0] = [0.7, 0.3, 0.0, 0.0]  # prompt picks expert 0 for K'=1
    probs[0, 1:] = [0.5, 0.0, 0.3, 0.2]  # generated: top-1 is expert 0, top-2 adds expert 2
    assert segment_kept(probs, 1, 1, 32, top_k=1) == (pytest.approx(1.0), pytest.approx(1.0))
    assert segment_kept(probs, 1, 1, 32, top_k=2) == (pytest.approx(0.5 / 0.8), pytest.approx(0.5))


def test_flatness_of_a_uniform_router():
    probs = np.full((2, 10, 16), 1 / 16, dtype=np.float32)
    top1, topk, entropy = flatness(probs, 0, top_k=4)
    assert (top1, topk, entropy) == (pytest.approx(1 / 16), pytest.approx(0.25), pytest.approx(1.0))


def test_flatness_ignores_prompt_tokens():
    probs = np.full((1, 4, 4), 0.25, dtype=np.float32)
    probs[0, 2:] = [1.0, 0.0, 0.0, 0.0]
    top1, topk, entropy = flatness(probs, 2, top_k=2)
    assert (top1, topk, entropy) == (pytest.approx(1.0), pytest.approx(1.0), pytest.approx(0.0, abs=1e-6))


def test_top_k_per_model():
    assert TOP_K == {"gemma": 8, "qwen36": 8, "nemotron": 6, "qwen": 8}


def test_join_quality_adds_kept_quality_at_the_same_k_and_s(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "model_tag,suite,metric,k,s,kept\n"
        "gemma,short,qa_f1,16,32,0.64\n"
        "gemma,short,cls_acc,16,32,0.95\n"
        "gemma,short,qa_f1,16,16,0.1\n"
        "gemma,gsm8k,gsm8k_acc,16,32,\n"
    )
    rows = join_quality([{"model_tag": "gemma", "k": 16, "s": 32, "mass": 0.7}], summary)
    assert rows == [{"model_tag": "gemma", "k": 16, "s": 32, "mass": 0.7, "qa_f1": 0.64, "cls_acc": 0.95}]


def test_module_entry_point_runs_from_the_repo_root():
    root = Path(tools.router_stats.__file__).resolve().parent.parent
    done = subprocess.run(
        [sys.executable, "-m", "tools.router_stats", "--help"], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
