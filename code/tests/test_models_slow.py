"""Real-model tests: each model loads once per module and is freed before the next one loads."""

import gc

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

from models import MODELS, N_EXPERTS, eos_ids
from segroute import chat, generate, install

pytestmark = pytest.mark.slow

ARTICLES = [
    "The city council voted on Tuesday to extend the tram line by six kilometers. The project will cost about "
    "400 million euros and is due to finish in 2031. Supporters say it will cut car traffic in the old town. "
    "Opponents worry about years of construction noise. A final budget vote is planned for March.",
    "A regional airline said on Monday it will retire its oldest turboprops by the end of next year. The company "
    "ordered twelve newer aircraft to replace them. Executives expect fuel costs to fall by a fifth. Pilots' "
    "unions asked for talks on the new schedules. The first delivery is expected in the spring.",
    "Researchers at a coastal university reported that local oyster beds recovered faster than forecast. The team "
    "counted three times more young oysters than in the previous survey. Cleaner water appears to be the main "
    "cause. The state will fund two more years of monitoring. Fishermen hope the harvest ban can end soon.",
    "A national library opened its digitized newspaper archive to the public on Friday. The collection covers "
    "more than two centuries of print. Visitors can search every page by keyword. Staff spent six years scanning "
    "the material. The library plans to add radio transcripts next year.",
    "Farmers in the northern valley began harvesting wheat two weeks earlier than usual. Warm weather in May sped "
    "up ripening. Yields look average so far. Grain buyers expect prices to stay stable. A storm forecast for the "
    "weekend pushed crews to work through the night.",
]
SUMMARIES = ["Summarize this news article in 2 or 3 sentences.\n\n" + a for a in ARTICLES]
FIXED = [
    "What is the capital of France?",
    "List three uses of a hash map.",
    "Explain in one sentence why the sky is blue.",
]
FORBIDDEN = ("<think>", "</think>", "<|channel|>")


@pytest.fixture(scope="module", params=list(MODELS))
def loaded(request):
    """Load one model, install the controller, free everything at teardown."""
    model, tok = load(MODELS[request.param])
    ctl = install(model)
    yield request.param, model, tok, ctl
    del model, tok, ctl
    mx.clear_cache()
    gc.collect()


def only(*tags):
    """Run a test for the listed model tags only."""
    return pytest.mark.parametrize("loaded", tags, indirect=True)


@only(*MODELS)
def test_n_experts_matches_registry(loaded):
    tag, _, _, ctl = loaded
    assert N_EXPERTS[tag] == ctl.n_experts


@only("nemotron")
def test_stops_at_any_eos(loaded):
    assert eos_ids(loaded[2]) >= {2, 11}


@only("qwen36", "nemotron")
def test_no_thinking_text(loaded):
    _, model, tok, ctl = loaded
    ctl.mode = "off"
    for prompt in SUMMARIES:
        out = generate(model, tok, chat(tok, prompt), 120)
        text = tok.decode(out)
        assert not any(f in text for f in FORBIDDEN), text
        assert len(out) < 120, f"did not stop by EOS: {text}"


@only(*MODELS)
def test_full_budget_is_identity(loaded):
    _, model, tok, ctl = loaded
    for prompt in FIXED:
        ids = chat(tok, prompt)
        ctl.reset()
        ctl.mode = "off"
        ref = generate(model, tok, ids, 24)
        ctl.reset()
        ctl.mode, ctl.k, ctl.s = "mask", ctl.n_experts, 32
        assert generate(model, tok, ids, 24) == ref
    ctl.mode = "off"


@only("qwen36", "nemotron")
def test_mask_reduces_loads(loaded):
    _, model, tok, ctl = loaded
    ctl.reset()
    ctl.mode, ctl.k, ctl.s = "mask", ctl.n_experts // 4, 32
    generate(model, tok, chat(tok, SUMMARIES[0]), 40)
    loads = float(np.mean(ctl.loads))
    ctl.mode = "off"
    assert 0 < loads < ctl.n_experts
