"""Vantage -- a serving-performance and quality benchmark harness for LLM inference.

The package is organised in layers, each of which can be used on its own:

    backends/   adapters that turn "some way of running an LLM" into one interface
    workload/   what we send at it, and at what arrival pattern
    metrics/    what we measure, how we summarise it, and what it costs
    cache/      an optimisation we implement and measure (exact + semantic)
    quality/    the guard rail: did the optimisation break the answers?
    report/     charts and a written report produced from a finished run

Nothing in here imports torch. Training lives in the top-level `finetune/`
directory and is deliberately kept out of the runtime package.
"""

__version__ = "0.1.0"

from vantage.models import (  # noqa: F401
    GenRequest,
    GenResult,
    LoadConfig,
    RunSummary,
)
