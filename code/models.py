"""Model registry and stop tokens for the four 4-bit MLX models under study."""

from typing import Any

MODELS: dict[str, str] = {
    "gemma": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "qwen36": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "nemotron": "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
    "qwen": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
}
N_EXPERTS: dict[str, int] = {"gemma": 128, "qwen36": 256, "nemotron": 128, "qwen": 128}


def eos_ids(tok: Any, config: dict | None = None) -> set[int]:
    """Every token id that ends a generation: tokenizer eos set, single eos id and the config eos_token_id."""
    ids = set(getattr(tok, "eos_token_ids", None) or ())
    single = getattr(tok, "eos_token_id", None)
    if single is not None:
        ids.add(single)
    from_config = (config or {}).get("eos_token_id")
    if from_config is not None:
        ids.update(from_config if isinstance(from_config, list) else [from_config])
    return ids
