from __future__ import annotations

import json
from pathlib import Path

import pytest

from vantage.workload.dataset import Workload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_workload() -> Workload:
    return Workload.from_jsonl(ROOT / "data" / "workload.sample.jsonl")


@pytest.fixture
def tiny_workload(tmp_path: Path) -> Workload:
    rows = [
        {
            "id": f"t-{i}",
            "question": f"What was the recovery for assay {i}?",
            "context": f"[1] Assay {i} source text. Mean recovery was {90 + i}.0 percent.\n"
            f"[2] Unrelated discussion of storage conditions.",
            "reference_answer": f"Mean recovery was {90 + i}.0% [1].",
            "reference_citations": [1],
            "source_ids": [1, 2],
        }
        for i in range(6)
    ]
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return Workload.from_jsonl(path)
