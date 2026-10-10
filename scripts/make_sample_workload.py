"""Generate the bundled sample workload.

This is synthetic data in the shape of Sidhartha's Vectora corpus (vitamin D
measurement: immunoassay, LC-MS/MS, electrochemical biosensors, spectroscopy).
It exists so the harness is runnable by anyone who clones the repo, including
CI, without shipping copyrighted paper text.

Replace it with real data by exporting your own retrieval pipeline's
(question, retrieved context, reference answer) triples into the same JSONL
shape -- see docs/WORKLOAD.md. Every number below is invented.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

METHODS = [
    ("chemiluminescent immunoassay", "CLIA"),
    ("liquid chromatography tandem mass spectrometry", "LC-MS/MS"),
    ("electrochemical aptasensor", "aptasensor"),
    ("surface-enhanced Raman spectroscopy", "SERS"),
    ("high-performance liquid chromatography", "HPLC"),
    ("competitive protein binding assay", "CPBA"),
]

MATRICES = ["serum", "plasma", "dried blood spot", "whole blood"]

TEMPLATES = [
    (
        "What inter-assay coefficient of variation was reported for the {long} protocol?",
        "The inter-assay coefficient of variation was {cv}% for the {short} protocol [{cite}].",
        "Across {n} replicate runs the {long} protocol gave an inter-assay coefficient of "
        "variation of {cv}% and an intra-assay coefficient of variation of {cv2}%.",
    ),
    (
        "What limit of detection did the {long} method achieve in {matrix}?",
        "The method achieved a limit of detection of {lod} ng/mL in {matrix} [{cite}].",
        "Using spiked {matrix} samples the {long} method reached a limit of detection of "
        "{lod} ng/mL and a limit of quantification of {loq} ng/mL.",
    ),
    (
        "How many participants were included in the {short} validation cohort?",
        "The validation cohort included {n} participants [{cite}].",
        "A validation cohort of {n} participants was recruited and analysed with the {long} "
        "workflow over an eighteen-month period.",
    ),
    (
        "What recovery was observed for 25(OH)D3 with the {short} workflow?",
        "Mean recovery for 25(OH)D3 was {rec}% with the {short} workflow [{cite}].",
        "Spike-recovery experiments with the {long} workflow produced a mean recovery of "
        "{rec}% for 25(OH)D3 and {rec2}% for 25(OH)D2.",
    ),
    (
        "What correlation with the reference method was reported for {short}?",
        "Correlation with the reference method was r = {corr} [{cite}].",
        "Method comparison against the reference procedure gave a Pearson correlation of "
        "r = {corr} with a mean bias of {bias} ng/mL by Bland-Altman analysis.",
    ),
    (
        "What turnaround time per sample was reported for the {long} approach?",
        "Turnaround time was {tat} minutes per sample [{cite}].",
        "The {long} approach required {tat} minutes per sample including sample preparation, "
        "which the authors contrast with longer chromatographic workflows.",
    ),
    (
        "What threshold did the authors use to define vitamin D sufficiency?",
        "Sufficiency was defined as 25(OH)D above {thresh} ng/mL [{cite}].",
        "Participants were classified using a sufficiency threshold of {thresh} ng/mL of "
        "25(OH)D, consistent with the society guideline cited by the authors.",
    ),
    (
        "What interference was noted for the {short} assay?",
        "Cross-reactivity with the C3 epimer was reported at {xr}% [{cite}].",
        "The {long} assay showed cross-reactivity with the 3-epi-25(OH)D3 epimer of {xr}%, "
        "which the authors note is the dominant interference in paediatric samples.",
    ),
    (
        "What sample volume does the {short} protocol require?",
        "The protocol requires {vol} microlitres of {matrix} [{cite}].",
        "Only {vol} microlitres of {matrix} are required per determination, which the authors "
        "present as the main practical advantage of the {long} protocol.",
    ),
    (
        "What linear range was validated for the {long} method?",
        "The validated linear range was {lo} to {hi} ng/mL [{cite}].",
        "Linearity was demonstrated between {lo} and {hi} ng/mL with a coefficient of "
        "determination above 0.99 across the validated range.",
    ),
]

FILLER = [
    "Sample handling followed the standard pre-analytical protocol, with specimens "
    "centrifuged within two hours of collection and stored at minus eighty degrees "
    "Celsius until analysis. The authors report no measurable degradation over six "
    "months of storage under these conditions.",
    "Quality control materials were analysed at the start and end of each batch. "
    "Westgard rules were applied and no run was rejected during the study period, "
    "although two batches triggered a warning rule that resolved on repeat analysis.",
    "The authors discuss the practical constraints of deploying the workflow in "
    "resource-limited laboratories, noting instrument cost, consumable supply and "
    "the level of operator training required for reliable day-to-day operation.",
    "Statistical analysis was performed with standard software. Agreement between "
    "methods was assessed by Passing-Bablok regression in addition to the ordinary "
    "least squares fit reported in the main table.",
    "A limitation acknowledged by the authors is the single-centre design, which "
    "constrains how far the reported performance characteristics generalise to other "
    "laboratories and other patient populations.",
]


def build(seed: int = 42, n_items: int = 60) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    idx = 0
    while len(rows) < n_items:
        long_name, short_name = METHODS[idx % len(METHODS)]
        q_tpl, a_tpl, evidence_tpl = TEMPLATES[(idx // len(METHODS)) % len(TEMPLATES)]
        matrix = MATRICES[idx % len(MATRICES)]
        values = {
            "long": long_name,
            "short": short_name,
            "matrix": matrix,
            "cv": round(rng.uniform(2.1, 11.4), 1),
            "cv2": round(rng.uniform(1.2, 6.8), 1),
            "lod": round(rng.uniform(0.08, 2.4), 2),
            "loq": round(rng.uniform(2.5, 6.0), 2),
            "n": rng.choice([48, 96, 120, 184, 212, 340, 512]),
            "rec": round(rng.uniform(88.0, 108.0), 1),
            "rec2": round(rng.uniform(84.0, 104.0), 1),
            "corr": round(rng.uniform(0.86, 0.99), 2),
            "bias": round(rng.uniform(-3.4, 3.4), 1),
            "tat": rng.choice([8, 12, 18, 25, 40, 65]),
            "thresh": rng.choice([20, 30]),
            "xr": round(rng.uniform(0.4, 14.0), 1),
            "vol": rng.choice([10, 25, 50, 100, 200]),
            "lo": round(rng.uniform(1.0, 5.0), 1),
            "hi": rng.choice([80, 120, 150, 200]),
        }

        answer_source = rng.randint(1, 3)
        values["cite"] = answer_source

        evidence = evidence_tpl.format(**values)
        blocks = []
        for sid in (1, 2, 3):
            lead = (
                f"Study {sid}: a single-centre evaluation of {long_name} ({short_name}) "
                f"in {matrix} samples. "
            )
            body = evidence if sid == answer_source else rng.choice(FILLER)
            tail = " ".join(rng.sample(FILLER, 2))
            blocks.append(f"[{sid}] {lead}{body} {tail}")

        rows.append(
            {
                "id": f"vd-{idx:03d}",
                "question": q_tpl.format(**values),
                "context": "\n".join(blocks),
                "reference_answer": a_tpl.format(**values),
                "reference_citations": [answer_source],
                "source_ids": [1, 2, 3],
                "meta": {"method": short_name, "matrix": matrix, "synthetic": True},
            }
        )
        idx += 1
    return rows


if __name__ == "__main__":
    out = Path("data/workload.sample.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build()
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    # A held-out slice used as the quality evaluation set. Split by item index
    # so the two files never share an item: the whole point of a held-out set
    # is that nothing you tuned has seen it.
    holdout = rows[::2]
    with Path("data/eval.holdout.jsonl").open("w", encoding="utf-8") as fh:
        for row in holdout:
            fh.write(json.dumps(row) + "\n")

    words = [len(r["context"].split()) for r in rows]
    print(f"wrote {len(rows)} items to {out}")
    print(f"  context words: min {min(words)}, median {sorted(words)[len(words)//2]}, max {max(words)}")
    print(f"  held-out eval items: {len(holdout)} -> data/eval.holdout.jsonl")
