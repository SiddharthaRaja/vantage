"""Servers a sweep starts and stops itself.

A tuning experiment changes the server, not the load: the same requests
against vLLM with a different batch cap, chunked-prefill budget or memory
share. Those settings are launch flags, so comparing them means restarting the
server between configurations. Doing that by hand makes the experiment
unreproducible; declaring it in the sweep YAML makes it one command:

    servers:
      baseline:
        command: vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype half --port 8000
        env: { CUDA_VISIBLE_DEVICES: "0" }
        health_url: http://127.0.0.1:8000/v1/models
    runs:
      - { id: baseline, server: baseline, backend: ..., matrix: ... }

Runs that name a server get it started first (and the previous one stopped),
and the exact command is written into the run's manifest.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from vantage.errors import ConfigError


class ServerStartError(RuntimeError):
    pass


@dataclass
class ServerProfile:
    name: str
    command: str
    health_url: str
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout_s: float = 900.0
    stop_grace_s: float = 10.0

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> ServerProfile:
        if not isinstance(cfg, dict) or not cfg.get("command") or not cfg.get("health_url"):
            raise ConfigError(f"server '{name}' needs a 'command' and a 'health_url'")
        command = cfg["command"]
        if isinstance(command, list):
            command = shlex.join(str(c) for c in command)
        return cls(
            name=name,
            command=" ".join(str(command).split()),     # YAML folded lines -> one line
            health_url=str(cfg["health_url"]),
            env={str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
            startup_timeout_s=float(cfg.get("startup_timeout_s", 900)),
            stop_grace_s=float(cfg.get("stop_grace_s", 10)),
        )


def load_profiles(cfg: dict[str, Any]) -> dict[str, ServerProfile]:
    return {name: ServerProfile.from_config(name, c) for name, c in (cfg.get("servers") or {}).items()}


class ManagedServer:
    """One running server process (and every process it spawned)."""

    def __init__(self, profile: ServerProfile, log_path: Path):
        self.profile = profile
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None
        self.startup_s: float | None = None

    async def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        log.write(f"\n$ {self.profile.command}\n".encode())
        log.flush()
        t0 = time.perf_counter()
        # A new session, so stop() can signal vLLM's engine subprocesses too.
        self.proc = subprocess.Popen(
            shlex.split(self.profile.command),
            env={**os.environ, **self.profile.env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        async with httpx.AsyncClient(timeout=3.0) as client:
            while time.perf_counter() - t0 < self.profile.startup_timeout_s:
                if self.proc.poll() is not None:
                    raise ServerStartError(
                        f"server '{self.profile.name}' exited with {self.proc.returncode} "
                        f"during start-up:\n{self.log_tail()}"
                    )
                try:
                    r = await client.get(self.profile.health_url)
                    if r.status_code == 200:
                        self.startup_s = time.perf_counter() - t0
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2.0)
        await self.stop()
        raise ServerStartError(
            f"server '{self.profile.name}' not healthy after {self.profile.startup_timeout_s:.0f}s:\n"
            f"{self.log_tail()}"
        )

    async def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        t0 = time.perf_counter()
        while self.proc.poll() is None and time.perf_counter() - t0 < 30:
            await asyncio.sleep(0.5)
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=10)
        # GPU memory is released asynchronously after the process exits.
        await asyncio.sleep(self.profile.stop_grace_s)

    def log_tail(self, n: int = 20) -> str:
        try:
            return "".join(self.log_path.read_text(errors="replace").splitlines(keepends=True)[-n:])
        except OSError:
            return "(no log)"

