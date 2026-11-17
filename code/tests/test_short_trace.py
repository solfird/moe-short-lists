"""tools/short_trace.py: router trace of stored answers, prompt tokens included (H21)."""

import numpy as np

from tools import short_trace as st


def test_trace_file_round_trip(tmp_path):
    items = {("qa", 3): (np.arange(24, dtype=np.int16).reshape(2, 3, 4), 1, 0.5)}
    p = tmp_path / "t.npz"
    st.save(p, items)
    out = st.load_trace(p)
    assert np.array_equal(out[("qa", 3)][0], items[("qa", 3)][0])
    assert out[("qa", 3)][1] == 1 and out[("qa", 3)][2] == 0.5


def test_forward_ids_leave_out_the_last_answer_token():
    assert st.forward_ids([1, 2], [3, 4, 5]) == [1, 2, 3, 4]
    assert st.forward_ids([1, 2], []) == [1, 2]


def test_answer_part_cuts_the_prompt_tokens():
    a = np.zeros((5, 2, 3), dtype=np.int16)
    assert st.answer_part(a, 2).shape == (3, 2, 3)


def test_agreement_counts_stored_tokens_against_the_argmax():
    logits = np.zeros((1, 3, 5))
    logits[0, 0, 7 % 5] = 1  # position n_prompt - 1 predicts token 2
    logits[0, 1, 1] = 1
    logits[0, 2, 0] = 1
    assert st.agreement(logits[0], n_prompt=1, gen_ids=[2, 1]) == 1.0
    assert st.agreement(logits[0], n_prompt=1, gen_ids=[2, 4]) == 0.5
    assert st.agreement(logits[0], n_prompt=1, gen_ids=[]) == 1.0


def test_a_forward_pass_logs_one_row_per_token_in_every_layer():
    import mlx.core as mx

    import segroute as sr
    from tests.test_ablate import tiny
    from tools.cache_replay import decode_ids

    model = tiny()
    ctl = sr.install(model)
    ctl.mode = "log"
    ctl.reset()
    mx.eval(model(mx.array(st.forward_ids([1, 2, 3], [4, 5]))[None]))
    probs = np.stack([np.array(mx.concatenate(ctl.log[lyr], axis=0)) for lyr in range(ctl.n_layers)])
    top = decode_ids(probs, 0, 2)
    assert top.shape == (4, ctl.n_layers, 2)


def test_memory_note_reports_active_cache_and_peak_in_gib():
    assert st.memory_note(2**30, 3 * 2**30, 2**31) == "active 1.0 GiB, cache 3.0 GiB, peak 2.0 GiB"
