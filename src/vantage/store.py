"""Where results go.

One directory per sweep run, stamped with a timestamp, containing: the exact
resolved config, the environment it ran in, every individual request, the
per-configuration summaries, the quality scores, and the rendered report.

The rule is that the report must be reproducible from the raw files alone. If
a chart cannot be regenerated six weeks later from `raw.jsonl`, the chart is
not evidence.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vantage.config import results_root


def _json_default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def environment_fingerprint() -> dict[str, Any]:
    """Everything needed to explain why a number might not reproduce."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        info["git_sha"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        info["git_sha"] = None
    try:  # pragma: no cover - only meaningful on a GPU box
        nvidia = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["gpus"] = [line.strip() for line in nvidia.splitlines()]
    except Exception:
        info["gpus"] = []
    return info


class RunStore:
    def __init__(self, name: str, root: Path | None = None, timestamped: bool = True):
        base = Path(root) if root else results_root()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = (base / name / stamp) if timestamped else (base / name)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "charts").mkdir(exist_ok=True)

    # -- writers ------------------------------------------------------------ #

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.dir / filename
        path.write_text(
            json.dumps(payload, indent=2, default=_json_default, allow_nan=True),
            encoding="utf-8",
        )
        return path

    def append_jsonl(self, filename: str, rows: list[Any]) -> Path:
        path = self.dir / filename
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        return path

    def write_text(self, filename: str, text: str) -> Path:
        path = self.dir / filename
        path.write_text(text, encoding="utf-8")
        return path

    @property
    def charts_dir(self) -> Path:
        return self.dir / "charts"

    def __str__(self) -> str:
        return str(self.dir)


def load_summaries(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "summaries.json"
    if not path.exists():
        raise FileNotFoundError(f"no summaries.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run(name: str, root: Path | None = None) -> Path | None:
    base = (Path(root) if root else results_root()) / name
    if not base.exists():
        return None
    runs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)
    return runs[-1] if runs else None
