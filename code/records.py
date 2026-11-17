"""Jsonl record files: tolerant reading, torn-tail repair, one writer per file, and run provenance beside each file."""

import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def read_records(path: Path) -> list[dict]:
    """Records of a jsonl run file; a final line that does not parse (an interrupted write) is skipped."""
    lines = path.read_text().splitlines()
    recs = []
    for pos, line in enumerate(lines):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            if pos != len(lines) - 1:
                raise
    return recs


def repair_tail(path: Path) -> bool:
    """Cut a torn last line (no closing newline) in place so the next append starts on a fresh line; True if it cut."""
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    os.truncate(path, data.rfind(b"\n") + 1)  # in place: a kill here cannot empty the file
    return True


def lock_path(output: Path) -> Path:
    """Lock file that guards one output file: the output name plus .lock, in the same folder."""
    return output.with_name(output.name + ".lock")


@contextmanager
def exclusive(lock: Path, busy: str | None = None) -> Iterator[None]:
    """Hold a non-blocking exclusive flock on a lock file; exit with a busy message when another process holds it."""
    with open(lock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(busy or f"busy: {lock} is held by another process") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def meta_path(output: Path) -> Path:
    """Provenance file of one output file: the output name plus .meta.json, in the same folder."""
    return output.with_name(output.name + ".meta.json")


def write_meta(output: Path, meta: dict) -> Path:
    """Write the run provenance once; an existing file with other content is kept and a timestamped one added."""
    path = meta_path(output)
    if path.exists():
        if json.loads(path.read_text()) == json.loads(json.dumps(meta)):
            return path
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f")
        path = output.with_name(f"{output.name}.meta.{stamp}.json")
    path.write_text(json.dumps(meta, indent=1) + "\n")
    return path


def snapshot_hash(repo: str, cache: Path | None = None) -> str | None:
    """Commit hash of the cached Hugging Face snapshot that mlx_lm.load resolves for a repo, None when not cached."""
    if cache is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)
    ref = cache / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def git_commit() -> tuple[str | None, bool | None]:
    """HEAD commit of this repo and whether tracked files have uncommitted changes; (None, None) without git."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE, capture_output=True, text=True
        )
    except FileNotFoundError:
        return None, None
    if head.returncode != 0:
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def provenance(repo: str, params: dict) -> dict:
    """Run provenance: model repo and snapshot, mlx, mlx_lm and Python versions, git commit, start time, parameters."""
    commit, dirty = git_commit()
    return {
        "model": repo,
        "snapshot": snapshot_hash(repo),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "start": datetime.now().astimezone().isoformat(timespec="seconds"),
        **params,
    }


def _datasets_cache_from_argv() -> None:
    """Honor `--datasets-cache DIR` on any command line of this repository.

    The option names the folder where the `datasets` library keeps the benchmark data (CNN/DailyMail, SQuAD, AG News,
    GSM8K, XSum). It is removed from sys.argv here, before any parser runs, and set before `datasets` is imported.
    Commands that need the dataset text download it from the Hugging Face Hub into that folder on first use.
    """
    argv = sys.argv
    for pos, arg in enumerate(argv):
        if arg == "--datasets-cache" and pos + 1 < len(argv):
            os.environ["HF_DATASETS_CACHE"] = str(Path(argv[pos + 1]).expanduser().resolve())
            del argv[pos : pos + 2]
            return
        if arg.startswith("--datasets-cache="):
            os.environ["HF_DATASETS_CACHE"] = str(Path(arg.split("=", 1)[1]).expanduser().resolve())
            del argv[pos]
            return


_datasets_cache_from_argv()
