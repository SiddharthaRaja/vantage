"""Console output. Rich if available, plain print if not -- a benchmark tool
should never fail to run because a pretty-printer is missing."""

from __future__ import annotations

import os
import sys

try:  # pragma: no cover - trivial import guard
    from rich.console import Console

    _console: Console | None = Console(stderr=True)
except Exception:  # pragma: no cover
    _console = None

QUIET = os.getenv("VANTAGE_QUIET", "").lower() in {"1", "true", "yes"}


def log(message: str, style: str = "") -> None:
    """Write a status line to stderr so it never pollutes piped stdout."""
    if QUIET:
        return
    if _console is not None:
        _console.print(message, style=style or None, highlight=False)
    else:  # pragma: no cover
        print(message, file=sys.stderr)


def info(message: str) -> None:
    log(message)


def good(message: str) -> None:
    log(f"[green]{message}[/green]" if _console else message)


def warn(message: str) -> None:
    log(f"[yellow]warning:[/yellow] {message}" if _console else f"warning: {message}")


def error(message: str) -> None:
    log(f"[red]error:[/red] {message}" if _console else f"error: {message}")
