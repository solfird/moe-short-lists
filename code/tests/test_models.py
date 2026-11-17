"""Model registry and stop-token tests on stub tokenizers: no model download."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import segroute
from models import MODELS, N_EXPERTS, eos_ids


def test_registry_covers_four_tags():
    assert set(MODELS) == set(N_EXPERTS) == {"gemma", "qwen36", "nemotron", "qwen"}
    assert MODELS["qwen36"] == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert MODELS["nemotron"] == "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
    assert N_EXPERTS == {"gemma": 128, "qwen36": 256, "nemotron": 128, "qwen": 128}


def test_eos_ids_unions_every_source():
    tok = SimpleNamespace(eos_token_ids={11}, eos_token_id=2)
    assert eos_ids(tok) == {2, 11}


def test_eos_ids_without_plural_attribute_or_none():
    assert eos_ids(SimpleNamespace(eos_token_id=7)) == {7}
    assert eos_ids(SimpleNamespace(eos_token_ids=set(), eos_token_id=None)) == set()


class ScriptedModel:
    """Greedy-decodes a fixed token script, one id per call, whatever the input."""

    def __init__(self, script: list[int], vocab: int = 16) -> None:
        self.script, self.vocab, self.calls = script, vocab, 0

    def __call__(self, x, cache=None):
        logits = np.zeros((1, 1, self.vocab), dtype=np.float32)
        logits[0, 0, self.script[self.calls]] = 1.0
        self.calls += 1
        return mx.array(logits)


@pytest.mark.parametrize("stop", [2, 11])
def test_generate_stops_at_any_eos_id(monkeypatch, stop):
    monkeypatch.setattr(segroute, "make_prompt_cache", lambda model: None)
    tok = SimpleNamespace(eos_token_ids={11}, eos_token_id=2)
    out = segroute.generate(ScriptedModel([5, 6, stop, 7]), tok, [1, 2], max_tokens=10)
    assert out == [5, 6]


@pytest.mark.parametrize("value", [9, [9, 12]])
def test_eos_ids_adds_config_value(value):
    tok = SimpleNamespace(eos_token_ids={11}, eos_token_id=2)
    assert eos_ids(tok, {"eos_token_id": value}) >= {2, 11, 9}
    assert eos_ids(tok, {}) == {2, 11}


def test_generate_stops_at_config_eos_id(monkeypatch):
    monkeypatch.setattr(segroute, "make_prompt_cache", lambda model: None)
    tok = SimpleNamespace(eos_token_ids=set(), eos_token_id=2)
    out = segroute.generate(ScriptedModel([5, 9, 7]), tok, [1], max_tokens=10, config={"eos_token_id": [9]})
    assert out == [5]
