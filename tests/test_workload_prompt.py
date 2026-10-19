"""Exported workloads carry the application's own prompt and system prompt."""

import json

from vantage.workload.dataset import DEFAULT_SYSTEM, Workload


def _write(tmp_path, rows, system=None):
    path = tmp_path / "wl.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    if system is not None:
        path.with_suffix(".system.txt").write_text(system, encoding="utf-8")
    return path


def test_prompt_field_overrides_template(tmp_path):
    path = _write(tmp_path, [{"id": "a", "question": "q?", "context": "[1] c", "prompt": "EXACT USER MESSAGE"}])
    wl = Workload.from_jsonl(path)
    assert wl.items[0].prompt() == "EXACT USER MESSAGE"
    assert wl.request(wl.items[0], seq=0).prompt == "EXACT USER MESSAGE"


def test_template_used_without_prompt_field(tmp_path):
    path = _write(tmp_path, [{"id": "a", "question": "q?", "context": "[1] c"}])
    assert Workload.from_jsonl(path).items[0].prompt() == "Sources:\n[1] c\n\nQuestion: q?"


def test_system_sidecar_is_used(tmp_path):
    path = _write(tmp_path, [{"id": "a", "question": "q?", "context": "[1] c"}], system="APP SYSTEM\n")
    wl = Workload.from_jsonl(path)
    assert wl.system == "APP SYSTEM"
    assert wl.request(wl.items[0], seq=0).system == "APP SYSTEM"


def test_explicit_system_beats_sidecar(tmp_path):
    path = _write(tmp_path, [{"id": "a", "question": "q?", "context": "[1] c"}], system="APP SYSTEM")
    assert Workload.from_jsonl(path, system="OVERRIDE").system == "OVERRIDE"


def test_default_system_without_sidecar(tmp_path):
    path = _write(tmp_path, [{"id": "a", "question": "q?", "context": "[1] c"}])
    assert Workload.from_jsonl(path).system == DEFAULT_SYSTEM
