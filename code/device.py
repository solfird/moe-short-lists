"""H59: memory and speed with a real loading path (pass rules written 2026-11-03 07:35, before any run).

Expert weights stay in the checkpoint files. Each MoE layer keeps a compact buffer with only its listed experts, read
from the memory-mapped files at each list refresh (experts that stay on the list are reused), and expert ids are
remapped to buffer positions. The full expert tensors are released after load.

device.py run --model-tag gemma --mode full|masked|compact --k 24 --n 20   # results/device_gemma_<mode>_k24.json
device.py report --model-tag gemma                                        # results/device_gemma.json
"""

import argparse
import gc
import json
import os
import resource
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import selrule as sl

RESULTS = Path(__file__).resolve().parents[1] / "results"
DT = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "U8": mx.uint8}
PROJ = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")


class ExpertStore:
    """Byte-level reader of stacked expert tensors in safetensors files, first dimension = expert.

    Rows are copied with pread: mapped file pages would count as process memory."""

    def __init__(self, folder: Path) -> None:
        self.info: dict[str, tuple[int, str, list[int], int, int]] = {}
        self.bytes_read = 0
        for f in sorted(Path(folder).glob("*.safetensors")):
            with f.open("rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                header = json.loads(h.read(n))
            fd = os.open(f, os.O_RDONLY)
            for name, v in header.items():
                if name == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.info[name] = (fd, v["dtype"], v["shape"], 8 + n + a, b - a)

    def rows(self, name: str, experts: list[int]) -> mx.array:
        fd, dt, shape, off, nbytes = self.info[name]
        row = nbytes // shape[0]
        data = np.frombuffer(b"".join(os.pread(fd, row, off + e * row) for e in experts), dtype=np.uint8)
        self.bytes_read += data.nbytes
        return mx.view(mx.array(data), DT[dt]).reshape(len(experts), *shape[1:])


def plan(pos: dict[int, int], allowed: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    """Experts of the new list already in the buffer (with their position), and experts to read."""
    keep = [(e, pos[e]) for e in allowed if e in pos]
    fresh = [e for e in allowed if e not in pos]
    return keep, fresh


def lookup(order: list[int], n_experts: int) -> mx.array:
    t = np.zeros(n_experts, dtype=np.int32)
    t[order] = np.arange(len(order), dtype=np.int32)
    return mx.array(t)


class CompactSwitch(nn.Module):
    """A SwitchGLU whose expert tensors hold only the listed experts; ids are remapped to buffer positions."""

    def __init__(self, inner: nn.Module, n_experts: int) -> None:
        super().__init__()
        self.inner = inner
        self.lut = mx.zeros((n_experts,), dtype=mx.int32)

    def __call__(self, x: mx.array, indices: mx.array, *args: Any, **kw: Any) -> mx.array:
        return self.inner(x, self.lut[indices], *args, **kw)


class Loader:
    def __init__(self, store: ExpertStore, banks: list[tuple[CompactSwitch, str]], n_experts: int) -> None:
        self.store, self.banks, self.n = store, banks, n_experts
        self.pos: list[dict[int, int]] = [{} for _ in banks]
        self.seconds = 0.0
        empties = []
        for sw, _ in banks:  # release the full expert stacks
            for p in PROJ:
                for part in PARTS:
                    getattr(sw.inner, p)[part] = getattr(sw.inner, p)[part][:0]
                    empties.append(getattr(sw.inner, p)[part])
        mx.eval(*empties)  # a lazy slice would keep the full stack referenced
        gc.collect()
        mx.clear_cache()

    def load(self, layer: int, allowed: list[int]) -> None:
        t0 = time.perf_counter()
        sw, prefix = self.banks[layer]
        keep, fresh = plan(self.pos[layer], allowed)
        order = [e for e, _ in keep] + fresh
        new = []
        for p in PROJ:
            for part in PARTS:
                old = getattr(sw.inner, p)[part]
                pieces = []
                if keep:
                    pieces.append(old[mx.array([q for _, q in keep])])
                if fresh:
                    pieces.append(self.store.rows(f"{prefix}.{p}.{part}", fresh))
                arr = mx.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                getattr(sw.inner, p)[part] = arr
                new.append(arr)
        sw.lut = lookup(order, self.n)
        mx.eval(*new, sw.lut)
        self.pos[layer] = {e: i for i, e in enumerate(order)}
        self.seconds += time.perf_counter() - t0


class CompactController(sl.TailController):
    """Loads a layer's list when its mask takes effect: at once for the prompt set, at the next call after a refresh
    (Controller.observe returns the old mask on the token that triggers the refresh)."""

    loader: Loader | None = None

    def reset(self) -> None:
        super().reset()
        self.pending: list[list[int] | None] = [None] * self.n_layers

    def _select(self, layer: int, prob_sum: Any) -> None:
        super()._select(layer, prob_sum)
        self.pending[layer] = sorted(self.allowed[layer])

    def _flush(self, layer: int) -> None:
        if self.loader is not None and self.pending[layer] is not None:
            self.loader.load(layer, self.pending[layer])
        self.pending[layer] = None

    def observe(self, layer: int, probs: Any) -> Any:
        self._flush(layer)  # a list chosen at the previous call
        first = self.mode == "mask" and self.bias[layer] is None
        bias = super().observe(layer, probs)
        if first:
            self._flush(layer)  # the prompt set routes this call
        return bias


def _rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024


def _banks(model: nn.Module) -> list[tuple[nn.Module, str, int]]:
    """(experts owner, checkpoint prefix of its switch_glu, layer number) for each MoE layer, in router order."""
    from adapters import _layers

    out = []
    for n, layer in enumerate(_layers(model)):
        if getattr(layer, "router", None) is not None:
            out.append((layer.experts, f"language_model.model.layers.{n}.experts.switch_glu", n))
    return out


def cmd_run(a: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    import curriculum as cu
    import final
    import segroute as sr
    from models import MODELS

    model, tok, config = load(MODELS[a.model_tag], return_config=True)
    cls = CompactController if a.mode == "compact" else sl.TailController
    ctl = cu.install_with(model, cls)
    ctl.mode, ctl.k, ctl.s = ("off", None, None) if a.mode == "full" else ("mask", a.k, 32)
    if a.mode == "compact":
        store = ExpertStore(Path(snapshot_download(MODELS[a.model_tag], local_files_only=True)))
        banks = []
        for owner, prefix, _ in _banks(model):
            sw = CompactSwitch(owner.switch_glu, ctl.n_experts)
            owner.switch_glu = sw
            banks.append((sw, prefix))
        ctl.loader = Loader(store, banks, ctl.n_experts)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    setup = {"active_mb": mx.get_active_memory() / 2**20, "rss_mb": _rss_mb()}
    rows = []
    for i, prompt, _ in final.summary_items("cnn", a.n, a.start):
        ctl.reset()
        b0 = ctl.loader.store.bytes_read if a.mode == "compact" else 0
        s0 = ctl.loader.seconds if a.mode == "compact" else 0.0
        ids = sr.chat(tok, prompt)
        t0 = time.perf_counter()
        gen = sr.generate(model, tok, ids, 120, config)
        dt = time.perf_counter() - t0
        row = {"i": i, "prompt_tokens": len(ids), "gen": gen, "n_gen": len(gen), "seconds": dt}
        if a.mode == "compact":
            row["load_bytes"] = ctl.loader.store.bytes_read - b0
            row["load_seconds"] = ctl.loader.seconds - s0
        rows.append(row)
    out = {
        "model": a.model_tag,
        "mode": a.mode,
        "k": a.k,
        "setup": setup,
        "peak_mb": mx.get_peak_memory() / 2**20,
        "end_active_mb": mx.get_active_memory() / 2**20,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "end_rss_mb": _rss_mb(),
        "rows": rows,
    }
    (RESULTS / f"device_{a.model_tag}_{a.mode}_k{a.k}.json").write_text(json.dumps(out, indent=1))
    tps = sum(r["n_gen"] for r in rows) / sum(r["seconds"] for r in rows)
    print(a.mode, a.k, "active MB", round(setup["active_mb"]), "peak MB", round(out["peak_mb"]), "tok/s", round(tps, 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("run",))
    p.add_argument("--model-tag", required=True)
    p.add_argument("--mode", choices=("full", "masked", "compact"), default="full")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=600)
    a = p.parse_args()
    cmd_run(a)


if __name__ == "__main__":
    main()
