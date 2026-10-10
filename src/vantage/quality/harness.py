"""Run a held-out evaluation set through a backend and score the answers.

This is the gate. Its output is what turns "config B is 3.4x faster" into
"config B is 3.4x faster and loses 1.2 +/- 0.9 composite points, which is
within noise" -- or into "config B is 3.4x faster and loses 11 points, which
is not acceptable".

Two rules enforced here:

1. The eval set is never the set used to tune anything. Nothing in this harness
   writes to the data it reads.
2. A difference is only a difference if the bootstrap interval for the delta
   excludes zero. Reporting "0.81 vs 0.79" on 40 examples as a regression is
   how people fool themselves.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from vantage.backends.base import Backend
from vantage.logging import info
from vantage.metrics.stats import bootstrap_ci
from vantage.models import QualitySummary
from vantage.quality.scorers import ItemScore, score_item
from vantage.workload.dataset import Workload


async def run_quality_eval(
    backend: Backend,
    workload: Workload,
    config_id: str,
    concurrency: int = 4,
    max_tokens: int = 256,
    limit: int | None = None,
    progress: bool = False,
) -> QualitySummary:
    requests = workload.eval_requests(max_tokens=max_tokens)
    if limit:
        requests = requests[:limit]
    if not requests:
        raise ValueError(
            "the workload has no items with a reference_answer, so quality cannot be scored"
        )

    semaphore = asyncio.Semaphore(concurrency)

    async def one(req):
        async with semaphore:
            return await backend.generate(req)

    results = await asyncio.gather(*[one(r) for r in requests])

    scores: list[ItemScore] = []
    for req, res in zip(requests, results, strict=True):
        prediction = res.text if res.ok else ""
        scores.append(
            score_item(
                item_id=str(req.meta.get("item_id", req.id)),
                prediction=prediction,
                reference=str(req.meta.get("reference_answer", "")),
                source_ids=list(req.meta.get("source_ids") or []),
            )
        )

    composites = [s.composite for s in scores]
    ci = bootstrap_ci(composites) if len(composites) >= 3 else None

    summary = QualitySummary(
        config_id=config_id,
        n=len(scores),
        exact_match=float(np.mean([s.exact_match for s in scores])),
        token_f1=float(np.mean([s.token_f1 for s in scores])),
        citation_grounding=float(np.mean([s.citation_grounding for s in scores])),
        number_agreement=float(np.mean([s.number_agreement for s in scores])),
        composite=float(np.mean(composites)),
        composite_ci95=ci,
        per_item=[
            {
                "item_id": s.item_id,
                "composite": round(s.composite, 4),
                "token_f1": round(s.token_f1, 4),
                "citation_grounding": round(s.citation_grounding, 4),
                "number_agreement": round(s.number_agreement, 4),
                "prediction": s.prediction[:400],
            }
            for s in scores
        ],
    )
    if progress:
        info(f"  quality[{config_id}]: composite {summary.composite:.3f} over {summary.n} items")
    return summary


@dataclass
class QualityComparison:
    baseline_id: str
    candidate_id: str
    baseline_composite: float
    candidate_composite: float
    delta: float
    delta_ci95: tuple[float, float]
    significant: bool
    verdict: str  # "pass" | "warn" | "fail"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_id,
            "candidate": self.candidate_id,
            "baseline_composite": round(self.baseline_composite, 4),
            "candidate_composite": round(self.candidate_composite, 4),
            "delta": round(self.delta, 4),
            "delta_ci95": [round(self.delta_ci95[0], 4), round(self.delta_ci95[1], 4)],
            "significant": self.significant,
            "verdict": self.verdict,
            "detail": self.detail,
        }


def compare_quality(
    baseline: QualitySummary,
    candidate: QualitySummary,
    max_drop: float = 0.02,
    n_boot: int = 2000,
    seed: int = 17,
) -> QualityComparison:
    """Paired bootstrap over per-item composite scores.

    Paired, because both configurations answered the same questions: comparing
    independent means throws away the pairing and widens the interval for no
    reason. Items are matched on item_id; anything unmatched is dropped and
    said so.
    """
    base = {row["item_id"]: row["composite"] for row in baseline.per_item}
    cand = {row["item_id"]: row["composite"] for row in candidate.per_item}
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise ValueError("baseline and candidate share no evaluation items")

    diffs = np.asarray([cand[i] - base[i] for i in shared], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    delta = float(diffs.mean())
    significant = not (lo <= 0.0 <= hi)

    if delta >= 0 or not significant:
        verdict = "pass"
        detail = (
            f"no significant regression: delta {delta:+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}) over {len(shared)} paired items"
        )
    elif abs(delta) <= max_drop:
        verdict = "warn"
        detail = (
            f"small but real regression: delta {delta:+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}), within the {max_drop:.3f} budget"
        )
    else:
        verdict = "fail"
        detail = (
            f"quality regression beyond budget: delta {delta:+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}) exceeds the allowed {max_drop:.3f}"
        )

    return QualityComparison(
        baseline_id=baseline.config_id,
        candidate_id=candidate.config_id,
        baseline_composite=baseline.composite,
        candidate_composite=candidate.composite,
        delta=delta,
        delta_ci95=(lo, hi),
        significant=significant,
        verdict=verdict,
        detail=detail,
    )
