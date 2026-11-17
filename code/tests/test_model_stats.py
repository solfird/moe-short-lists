"""Counting logic of tools/model_stats.py on a tiny fake parameter list."""

import importlib
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

import tools.model_stats
from tools.model_stats import bias_ranges, routed_share


def params():
    return [
        ("layers.0.mlp.switch_mlp.up_proj.weight", mx.zeros((10,), dtype=mx.uint32)),  # 40 bytes
        ("layers.0.mlp.gate.weight", mx.zeros((5,), dtype=mx.uint32)),  # 20 bytes
        ("layers.1.mixer.gate.e_score_correction_bias", mx.array([40.0, 40.5])),
        ("layers.3.mixer.gate.e_score_correction_bias", mx.array([35.0, 35.25])),
    ]


def test_routed_share_counts_switch_mlp_bytes():
    routed, total = routed_share(params(), ("switch_mlp",))
    assert (routed, total) == (40, 40 + 20 + 8 + 8)


def test_bias_ranges_per_layer():
    assert bias_ranges(params()) == [
        ("layers.1.mixer.gate.e_score_correction_bias", 40.0, 40.5, 0.5),
        ("layers.3.mixer.gate.e_score_correction_bias", 35.0, 35.25, 0.25),
    ]


def test_routed_share_rejects_empty_match():
    with pytest.raises(ValueError):
        routed_share(params(), ("experts.",))


def test_import_leaves_sys_path_alone():
    before = list(sys.path)
    importlib.reload(tools.model_stats)
    assert sys.path == before


def test_module_entry_point_runs_from_the_repo_root():
    root = Path(tools.model_stats.__file__).resolve().parent.parent
    done = subprocess.run(
        [sys.executable, "-m", "tools.model_stats", "--help"], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert "nemotron" in done.stdout
