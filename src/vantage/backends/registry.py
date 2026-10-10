"""Build a backend from a YAML config file or a dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vantage.backends.base import Backend
from vantage.backends.openai_compat import OpenAICompatBackend
from vantage.backends.simulated import SimulatedBackend
from vantage.config import backend_config_path, load_yaml
from vantage.errors import ConfigError

BACKEND_TYPES: dict[str, type[Backend]] = {
    "openai_compat": OpenAICompatBackend,
    "openai": OpenAICompatBackend,
    "vllm": OpenAICompatBackend,
    "tgi": OpenAICompatBackend,
    "ollama": OpenAICompatBackend,
    "simulated": SimulatedBackend,
    "sim": SimulatedBackend,
}


def build_backend(spec: str | Path | dict[str, Any], overrides: dict[str, Any] | None = None) -> Backend:
    """`spec` is either a config name ('sim.fp16'), a path to a YAML file, or a
    dict already in the config shape."""
    if isinstance(spec, dict):
        cfg = dict(spec)
        source = "<inline>"
    else:
        path = backend_config_path(str(spec))
        cfg = load_yaml(path)
        source = str(path)

    cfg = {**cfg, **(overrides or {})}
    btype = cfg.pop("type", None)
    if btype is None:
        raise ConfigError(f"backend config {source} is missing a 'type' field")
    if btype not in BACKEND_TYPES:
        raise ConfigError(
            f"unknown backend type '{btype}' in {source}. "
            f"Known types: {', '.join(sorted(BACKEND_TYPES))}"
        )
    cfg.setdefault("name", Path(source).stem if source != "<inline>" else btype)
    cfg.pop("description", None)
    return BACKEND_TYPES[btype](**cfg)
