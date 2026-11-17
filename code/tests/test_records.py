"""Record files: tolerant reading and torn-tail repair before a resume-append."""

import importlib.metadata
import json
import platform
from datetime import datetime

import pytest

import records
import segroute
import short


def _lines(n: int) -> str:
    return "".join(json.dumps({"i": i}) + "\n" for i in range(n))


def test_short_reexports_read_records():
    assert short.read_records is records.read_records


def test_read_records_skips_torn_final_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(_lines(3) + '{"i": 3, "ou')
    assert [r["i"] for r in records.read_records(p)] == [0, 1, 2]


def test_read_records_rejects_corrupt_middle_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(_lines(1) + "not json\n" + _lines(1))
    with pytest.raises(json.JSONDecodeError):
        records.read_records(p)


def test_repair_tail_truncates_torn_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(_lines(3) + '{"i": 3, "ou')
    assert records.repair_tail(p) is True
    assert p.read_text() == _lines(3)


def test_repair_tail_leaves_clean_file(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(_lines(3))
    assert records.repair_tail(p) is False
    assert p.read_text() == _lines(3)


def test_repair_tail_single_torn_line_empties_file(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"i": 0, "o')
    assert records.repair_tail(p) is True
    assert p.read_text() == ""


def test_repair_tail_empty_file(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("")
    assert records.repair_tail(p) is False


def test_gsm8k_resume_drops_torn_line_and_append_stays_valid(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(_lines(3) + '{"i": 3, "ok": tr')
    assert segroute.gsm8k_done(p) == {0, 1, 2}
    assert p.read_text().count("\n") == 3
    with open(p, "a") as f:
        f.write(json.dumps({"i": 3}) + "\n")
    assert [json.loads(x)["i"] for x in p.read_text().splitlines()] == [0, 1, 2, 3]
    assert len(records.read_records(p)) == 4


def test_gsm8k_done_missing_file(tmp_path):
    assert segroute.gsm8k_done(tmp_path / "none.jsonl") == set()


def test_repair_tail_truncates_in_place(tmp_path, monkeypatch):
    """The cut is an in-place truncate: same inode, prefix bytes untouched, the file is never rewritten."""
    p = tmp_path / "r.jsonl"
    prefix = _lines(3).encode()
    p.write_bytes(prefix + b'{"i": 3, "ou')
    inode = p.stat().st_ino

    def no_rewrite(self, data):
        raise AssertionError("repair_tail must not rewrite the file (a kill mid-write would empty it)")

    monkeypatch.setattr(type(p), "write_bytes", no_rewrite)
    assert records.repair_tail(p) is True
    assert p.stat().st_ino == inode
    assert p.read_bytes() == prefix


def test_lock_path_sits_beside_the_output(tmp_path):
    assert records.lock_path(tmp_path / "x.jsonl") == tmp_path / "x.jsonl.lock"


def test_exclusive_refuses_a_held_lock(tmp_path):
    lock = tmp_path / "x.jsonl.lock"
    with records.exclusive(lock), pytest.raises(SystemExit, match=f"busy.*{lock}"), records.exclusive(lock):
        pass
    with records.exclusive(lock):  # released on exit
        pass


def test_exclusive_custom_message(tmp_path):
    lock = tmp_path / "sweep.lock"
    with (
        records.exclusive(lock),
        pytest.raises(SystemExit, match="another sweep is running"),
        records.exclusive(lock, "another sweep is running"),
    ):
        pass


def test_exclusive_releases_on_error(tmp_path):
    lock = tmp_path / "x.lock"
    with pytest.raises(RuntimeError), records.exclusive(lock):
        raise RuntimeError
    with records.exclusive(lock):
        pass


def test_meta_path_sits_beside_the_output(tmp_path):
    assert records.meta_path(tmp_path / "x.jsonl") == tmp_path / "x.jsonl.meta.json"


def test_write_meta_creates_then_keeps_identical(tmp_path):
    out = tmp_path / "x.jsonl"
    meta = {"model": "m", "k": 16}
    assert records.write_meta(out, meta) == tmp_path / "x.jsonl.meta.json"
    assert json.loads((tmp_path / "x.jsonl.meta.json").read_text()) == meta
    assert records.write_meta(out, dict(meta)) == tmp_path / "x.jsonl.meta.json"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.jsonl.meta.json"]


def test_write_meta_never_overwrites_different_content(tmp_path):
    out = tmp_path / "x.jsonl"
    records.write_meta(out, {"model": "m", "start": "t0"})
    second = records.write_meta(out, {"model": "m", "start": "t1"})
    assert json.loads((tmp_path / "x.jsonl.meta.json").read_text()) == {"model": "m", "start": "t0"}
    assert second.name.startswith("x.jsonl.meta.") and second.name.endswith(".json")
    assert second != tmp_path / "x.jsonl.meta.json"
    assert json.loads(second.read_text()) == {"model": "m", "start": "t1"}
    third = records.write_meta(out, {"model": "m", "start": "t2"})
    assert third != second
    assert len(list(tmp_path.glob("x.jsonl.meta*.json"))) == 3


def test_snapshot_hash_from_the_hub_cache(tmp_path):
    ref = tmp_path / "models--org--name" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123\n")
    assert records.snapshot_hash("org/name", tmp_path) == "abc123"
    assert records.snapshot_hash("org/other", tmp_path) is None


def test_provenance_fields(monkeypatch):
    monkeypatch.setattr(records, "snapshot_hash", lambda repo: "abc123")
    monkeypatch.setattr(records, "git_commit", lambda: ("deadbeef", False))
    meta = records.provenance("org/name", {"frac": 0.25, "k": 32, "s": 16, "n": 300})
    assert meta["model"] == "org/name"
    assert meta["snapshot"] == "abc123"
    assert (meta["git_commit"], meta["git_dirty"]) == ("deadbeef", False)
    assert (meta["frac"], meta["k"], meta["s"], meta["n"]) == (0.25, 32, 16, 300)
    assert meta["mlx"] == importlib.metadata.version("mlx")
    assert meta["mlx_lm"] == importlib.metadata.version("mlx-lm")
    assert meta["python"] == platform.python_version()
    datetime.fromisoformat(meta["start"])
    json.dumps(meta)


def test_git_commit_of_this_repo():
    commit, dirty = records.git_commit()
    assert commit is None or len(commit) == 40
    assert isinstance(dirty, bool) or dirty is None
