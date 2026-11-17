"""Dataset text on demand. The records of this repository hold no article and no reference text.

An item that needs its article carries `source` (dataset, config, split, index). `article_of` fetches the text from the
Hugging Face Hub, or from a local datasets cache given with `--datasets-cache DIR` (see records.py), and cuts it to the
3,000 characters that every prompt of the study uses.
"""

from functools import lru_cache
from typing import Any

CUT = 3000
FIELDS = {"abisee/cnn_dailymail": "article", "EdinburghNLP/xsum": "document"}


def source(dataset: str, index: int, config: str | None = None, split: str = "test") -> dict[str, Any]:
    """The source record that replaces the article in a shipped item."""
    return {"dataset": dataset, "config": config, "split": split, "index": index}


@lru_cache(maxsize=4)
def _split(dataset: str, config: str | None, split: str) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)


def article_of(item: dict[str, Any]) -> str:
    """The article of an item: its own `article` when a new run wrote one, else the text its `source` points to."""
    if "article" in item:
        return item["article"]
    s = item["source"]
    return _split(s["dataset"], s["config"], s["split"])[s["index"]][FIELDS[s["dataset"]]][:CUT]
