"""Project configuration: pinned dependencies match the venv, and the default test run skips slow and data tests."""

import importlib.metadata
import tomllib
from pathlib import Path

from tests import test_c2, test_short

PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
PINNED = ("mlx", "mlx-lm", "datasets", "rouge-score", "numpy")


def _marks(fn: object) -> set[str]:
    return {m.name for m in getattr(fn, "pytestmark", [])}


def test_default_run_excludes_slow_and_data():
    opts = PYPROJECT["tool"]["pytest"]["ini_options"]
    assert "not slow" in opts["addopts"] and "not data" in opts["addopts"]
    assert {m.split(":")[0] for m in opts["markers"]} == {"slow", "data"}


def test_dataset_tests_are_marked_data():
    assert "data" in _marks(test_short.test_items_count_per_task)
    assert "data" in _marks(test_c2.test_prompt_texts_match_the_free_runs)
    assert "data" not in _marks(test_short.test_article_slices_disjoint)


def test_project_table():
    project = PYPROJECT["project"]
    assert (project["name"], project["version"], project["requires-python"]) == (
        "moe-short-lists",
        "1.0.0",
        ">=3.12",
    )


def test_dependencies_pinned_to_the_installed_versions():
    deps = dict(d.split("==") for d in PYPROJECT["project"]["dependencies"])
    assert set(deps) == set(PINNED)
    assert deps["mlx"] == "0.32.4" and deps["mlx-lm"] == "0.32.2"
    for name, version in deps.items():
        assert importlib.metadata.version(name) == version, name
    dev = dict(d.split("==") for d in PYPROJECT["project"]["optional-dependencies"]["dev"])
    assert set(dev) == {"pytest", "ruff"}
    for name, version in dev.items():
        assert importlib.metadata.version(name) == version, name
