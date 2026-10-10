"""Config loading: YAML files plus environment variables, with ${VAR} expansion.

Design rule: nothing in the codebase reads os.environ directly except this
module. That is what makes a run reproducible -- the resolved config is written
into every run directory, so six weeks later you can see exactly which base URL
and which model produced a given chart.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from vantage.errors import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader. No dependency, no surprises, does not overwrite
    variables that are already set in the real environment."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} inside loaded YAML."""
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level")
    return expand_env(data)


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge, with `override` winning. Used for defaults -> per-run params."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def results_root() -> Path:
    return Path(os.getenv("VANTAGE_RESULTS_DIR", "results"))


def backend_config_path(name: str, search_dirs: list[Path] | None = None) -> Path:
    """Resolve 'sim.fp16' or a path to configs/backends/sim.fp16.yaml."""
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    dirs = search_dirs or [Path("configs/backends"), Path(__file__).parent / "presets"]
    for d in dirs:
        for suffix in (".yaml", ".yml"):
            p = d / f"{name}{suffix}"
            if p.exists():
                return p
    raise ConfigError(
        f"no backend config named '{name}'. Looked in: {', '.join(str(d) for d in dirs)}"
    )
