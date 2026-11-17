import json
import struct

import mlx.core as mx
import numpy as np

import device


def write_safetensors(path, tensors):
    """tensors: name -> (dtype string, numpy array of the raw storage type)."""
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_rows_reads_only_the_listed_experts_with_their_dtype(tmp_path):
    w = np.arange(4 * 2 * 3, dtype=np.uint32).reshape(4, 2, 3)
    s = np.arange(4 * 2, dtype=np.uint16).reshape(4, 2)  # bf16 bit patterns
    write_safetensors(tmp_path / "model-1.safetensors", {"a.weight": ("U32", w), "a.scales": ("BF16", s)})
    store = device.ExpertStore(tmp_path)
    got = store.rows("a.weight", [3, 1])
    assert got.dtype == mx.uint32 and got.shape == (2, 2, 3)
    assert np.array_equal(np.array(got), w[[3, 1]])
    sc = store.rows("a.scales", [2])
    assert sc.dtype == mx.bfloat16 and sc.shape == (1, 2)
    assert np.array_equal(np.array(mx.view(sc, mx.uint16)), s[[2]])
    assert store.bytes_read == 2 * 2 * 3 * 4 + 2 * 2


def test_plan_reuses_kept_experts_and_reads_only_new_ones():
    keep, fresh = device.plan({5: 0, 7: 1, 9: 2}, [7, 2, 9])
    assert keep == [(7, 1), (9, 2)]
    assert fresh == [2]


def test_lookup_table_maps_expert_ids_to_buffer_positions():
    lut = device.lookup([7, 9, 2], 10)
    assert lut.tolist()[7] == 0 and lut.tolist()[9] == 1 and lut.tolist()[2] == 2


class FakeLoader:
    def __init__(self):
        self.calls: list[list[int]] = []

    def load(self, layer, allowed):
        self.calls.append(list(allowed))


def _probs(hot):
    p = np.full(8, 0.01, dtype=np.float32)
    p[hot] = 0.45
    return mx.array(p[None])


def _allowed(bias):
    return sorted(np.nonzero(np.array(bias) == 0)[0].tolist())


def test_buffer_always_holds_the_experts_the_returned_mask_allows():
    ctl = device.CompactController(1, 8, lambda: None)
    ctl.mode, ctl.k, ctl.s = "mask", 2, 2
    ctl.loader = FakeLoader()
    seen = []
    for hot in ([0, 1], [6, 7], [6, 7], [6, 7], [6, 7]):
        bias = ctl.observe(0, _probs(hot))
        seen.append((_allowed(bias), ctl.loader.calls[-1]))
    assert all(a == b for a, b in seen), seen
    assert ctl.loader.calls == [[0, 1], [6, 7]]


def test_a_refresh_loads_the_new_list_only_at_the_next_call():
    ctl = device.CompactController(1, 8, lambda: None)
    ctl.mode, ctl.k, ctl.s = "mask", 2, 2
    ctl.loader = FakeLoader()
    ctl.observe(0, _probs([0, 1]))
    ctl.observe(0, _probs([6, 7]))
    ctl.observe(0, _probs([6, 7]))  # reselects here; this token still routes on the old list
    assert ctl.loader.calls == [[0, 1]]
    ctl.observe(0, _probs([6, 7]))
    assert ctl.loader.calls == [[0, 1], [6, 7]]


def test_store_reads_with_pread_and_maps_no_file(tmp_path):
    """Mapped pages count as process memory; the store must copy rows with pread instead."""
    w = np.arange(4 * 5, dtype=np.uint32).reshape(4, 5)
    write_safetensors(tmp_path / "model-1.safetensors", {"a.weight": ("U32", w)})
    store = device.ExpertStore(tmp_path)
    assert not any(isinstance(v, np.memmap) for entry in store.info.values() for v in entry)
    assert np.array_equal(np.array(store.rows("a.weight", [0, 2])), w[[0, 2]])
