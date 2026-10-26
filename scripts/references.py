"""
Reference answers for the quality gate, written against the exact prompts.

QASPER's annotator answers are short spans ("BiLSTM", "Yes"). Token F1
between a span and a full cited paragraph is low for every configuration, so
it barely separates a good quantised model from a broken one. A reference
written in the shape of the answer you want — two or three sentences, cited
by source number — makes the gate sensitive to the differences that matter.

Two steps:

    python scripts/references.py export       # -> data/references_todo.csv (40 rows)
    # open it in Excel/Sheets and fill `your_reference` for each row
    python scripts/references.py merge        # -> data/eval.references.jsonl

How to write each reference (≈2 minutes a row):

* Read `context` (the exact sources the model sees) and `question`.
  `qasper_answer` is the annotators' answer, as a hint.
* Write the answer a careful expert would give FROM THESE SOURCES ONLY:
  1–3 sentences, each claim followed by its source number, e.g.
  "They use the SemEval-2017 Task 4 tweets [2] and report macro-F1 [2][4]."
* If the sources do not contain the answer, write SKIP: the row is left
  out (the quality gate scores answers against references; refusals are
  Vectora's refusal metric, not this one). Export gives 40 rows so a few
  SKIPs still leave 30+.
* Do not paste the model's output and edit it: that makes the reference
  agree with one configuration by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
_CITE = re.compile(r"\[(\d{1,2})\]")
COLUMNS = ["id", "question", "qasper_answer", "your_reference", "context"]


def _rank(item_id: str) -> str:
    return hashlib.sha256(f"refs:{item_id}".encode()).hexdigest()


def export(source: Path, out: Path, n: int) -> None:
    rows = [json.loads(x) for x in source.read_text("utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r: _rank(str(r["id"])))
    picked = rows[:n]
    if out.exists():
        raise SystemExit(f"{out} exists — not overwriting your work. Delete it to re-export.")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in picked:
            w.writerow({
                "id": r["id"],
                "question": r["question"],
                "qasper_answer": r.get("reference_answer", ""),
                "your_reference": "",
                "context": r["context"],
            })
    print(f"Wrote {len(picked)} rows to {out} (from {source.name}).")
    print("Fill `your_reference` for every row, save as CSV, then run `merge`.")


def merge(source: Path, labels: Path, out: Path, min_rows: int) -> None:
    by_id = {
        str(r["id"]): r
        for r in (json.loads(x) for x in source.read_text("utf-8").splitlines() if x.strip())
    }
    merged, skipped, skipped_on_purpose = [], [], []
    with labels.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ref = (row.get("your_reference") or "").strip()
            item = by_id.get(row["id"])
            if not ref or item is None:
                skipped.append(row["id"])
                continue
            if ref.upper() == "SKIP":
                skipped_on_purpose.append(row["id"])
                continue
            item = dict(item)
            item["reference_answer"] = ref
            item["reference_citations"] = sorted({int(n) for n in _CITE.findall(ref)})
            item["meta"] = {**item.get("meta", {}), "reference_source": "written",
                            "qasper_answer": row.get("qasper_answer", "")}
            merged.append(item)
    if len(merged) < min_rows:
        raise SystemExit(f"Only {len(merged)} rows have a reference (need {min_rows}). "
                         f"Empty: {', '.join(skipped[:10])}")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", "utf-8")
    system = source.with_suffix(".system.txt")
    if system.exists():
        out.with_suffix(".system.txt").write_text(system.read_text("utf-8"), "utf-8")
    print(f"Wrote {len(merged)} references to {out} "
          f"({len(skipped_on_purpose)} SKIP, {len(skipped)} left empty).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["export", "merge"])
    ap.add_argument("--source", type=Path, default=DATA / "eval.qasper-sentiment.jsonl",
                    help="Rows to pick from (an eval file from scripts/export_vectora.py)")
    ap.add_argument("--labels", type=Path, default=DATA / "references_todo.csv")
    ap.add_argument("--out", type=Path, default=DATA / "eval.references.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-rows", type=int, default=30)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.source, args.labels, args.n)
    else:
        merge(args.source, args.labels, args.out, args.min_rows)


if __name__ == "__main__":
    main()
