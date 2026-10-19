"""The pure helpers in scripts/export_vectora.py (the export itself needs Vectora)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_vectora", Path(__file__).resolve().parent.parent / "scripts" / "export_vectora.py"
)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_strip_title():
    assert ev.strip_title('In the paper "RobBERT: a Dutch model": What data did they use?') == "What data did they use?"
    assert ev.strip_title("Plain question?") == "Plain question?"


def test_gold_answer():
    assert ev.gold_answer("Gold answer: SemEval-2016 Task 5 BIBREF27 , BIBREF28") == "SemEval-2016 Task 5"
    assert ev.gold_answer("Paper 1801.0 is in the corpus but does not answer this.") == ""
    assert ev.gold_answer("") == ""


def test_clean_placeholders():
    text = "as shown in TABREF3 using INLINEFORM0 and prior work BIBREF1, BIBREF2."
    assert ev.clean(text) == "as shown in the table using [formula] and prior work."
    assert "BIBREF" not in ev.clean("worldwideBIBREF0,")


def test_eval_split_is_stable():
    ids = [f"q-{i:04d}" for i in range(500)]
    first = [i for i in ids if ev.in_eval(i, 0.6)]
    assert first == [i for i in ids if ev.in_eval(i, 0.6)]
    assert 0.5 < len(first) / len(ids) < 0.7
