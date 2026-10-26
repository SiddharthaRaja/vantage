"""Reference answers: export picks a stable subset, merge replaces references."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "references", Path(__file__).resolve().parent.parent / "scripts" / "references.py"
)
refs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refs)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "eval.vectora.jsonl"
    rows = [{"id": f"q{i}", "question": f"question {i}?", "context": f"SOURCE [1] text {i}",
             "prompt": "p", "reference_answer": f"span {i}", "meta": {"x": 1}} for i in range(10)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    path.with_suffix(".system.txt").write_text("SYS\n")
    return path


def _fill(path, answers):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for r in rows:
        r["your_reference"] = answers.get(r["id"], "")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def test_export_is_stable_and_refuses_to_overwrite(tmp_path, source):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    refs.export(source, a, 4)
    refs.export(source, b, 4)
    assert a.read_text() == b.read_text()
    with pytest.raises(SystemExit):
        refs.export(source, a, 4)


def test_merge_uses_written_answers_and_parses_citations(tmp_path, source):
    labels = tmp_path / "todo.csv"
    refs.export(source, labels, 4)
    rows = _fill(labels, {})
    ids = [r["id"] for r in rows]
    _fill(labels, {ids[0]: "They use X [1] and Y [2][1].", ids[1]: "SKIP", ids[2]: "Z [1]."})
    out = tmp_path / "eval.references.jsonl"
    refs.merge(source, labels, out, min_rows=2)
    merged = [json.loads(x) for x in out.read_text().splitlines()]
    assert [m["id"] for m in merged] == [ids[0], ids[2]]
    assert merged[0]["reference_answer"] == "They use X [1] and Y [2][1]."
    assert merged[0]["reference_citations"] == [1, 2]
    assert merged[0]["meta"]["reference_source"] == "written"
    assert out.with_suffix(".system.txt").read_text() == "SYS\n"


def test_merge_demands_enough_rows(tmp_path, source):
    labels = tmp_path / "todo.csv"
    refs.export(source, labels, 4)
    with pytest.raises(SystemExit):
        refs.merge(source, labels, tmp_path / "o.jsonl", min_rows=30)
