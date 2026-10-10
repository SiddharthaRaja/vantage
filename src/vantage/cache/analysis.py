"""Threshold sweep for the semantic cache.

The question this answers: where do you set the similarity threshold?

Set it high and the cache almost never fires, so you paid for an embedding
model and got nothing. Set it low and it fires constantly -- and starts
answering question B with the answer to question A. The usual advice ("try
0.9") is not an engineering decision; this is.

Method: replay a query log through a fresh cache at each candidate threshold.
On a miss, the model is assumed to answer correctly and the reference answer is
stored. On a hit, the cached answer is what the user receives, and it is scored
against *this* query's reference. A hit whose served answer scores below
`quality_floor` is a false hit. Plotting hit rate and false-hit rate against
threshold gives you the operating point, and the cost saving that comes with it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vantage.cache.embedders import Embedder, HashingEmbedder
from vantage.cache.semantic import SemanticCache
from vantage.quality.scorers import token_f1


@dataclass
class ThresholdPoint:
    threshold: float
    n_queries: int
    hit_rate: float
    exact_hit_rate: float
    semantic_hit_rate: float
    false_hit_rate: float
    mean_served_quality: float
    mean_quality_on_hits: float
    estimated_calls_saved: int


@dataclass
class CacheSweep:
    points: list[ThresholdPoint]
    embedder: str
    quality_floor: float
    recommended_threshold: float | None
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder,
            "quality_floor": self.quality_floor,
            "recommended_threshold": self.recommended_threshold,
            "recommendation_reason": self.recommendation_reason,
            "points": [asdict(p) for p in self.points],
        }


def sweep_thresholds(
    queries: list[tuple[str, str]],
    thresholds: list[float] | None = None,
    embedder: Embedder | None = None,
    quality_floor: float = 0.75,
    max_false_hit_rate: float = 0.02,
) -> CacheSweep:
    """`queries` is an ordered log of (question, reference_answer) pairs.

    Order matters: a cache is a stateful, time-dependent thing, and replaying a
    shuffled log would overstate the hit rate of a cold cache.
    """
    thresholds = thresholds or [0.70, 0.75, 0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    embedder = embedder or HashingEmbedder()
    points: list[ThresholdPoint] = []

    for threshold in thresholds:
        cache = SemanticCache(embedder=embedder, threshold=threshold, exact_first=True)
        hits = exact_hits = semantic_hits = false_hits = 0
        served_quality: list[float] = []
        hit_quality: list[float] = []

        for question, reference in queries:
            lookup = cache.lookup(question)
            if lookup.hit and lookup.entry is not None:
                hits += 1
                if lookup.kind == "exact":
                    exact_hits += 1
                else:
                    semantic_hits += 1
                quality = token_f1(lookup.entry.value, reference)
                hit_quality.append(quality)
                served_quality.append(quality)
                if quality < quality_floor:
                    false_hits += 1
            else:
                # Cache miss: the model runs and (by construction of this
                # analysis) returns the reference answer.
                cache.put(question, reference)
                served_quality.append(1.0)

        n = len(queries)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                n_queries=n,
                hit_rate=hits / n if n else 0.0,
                exact_hit_rate=exact_hits / n if n else 0.0,
                semantic_hit_rate=semantic_hits / n if n else 0.0,
                false_hit_rate=(false_hits / hits) if hits else 0.0,
                mean_served_quality=sum(served_quality) / n if n else 0.0,
                mean_quality_on_hits=(sum(hit_quality) / len(hit_quality)) if hit_quality else 0.0,
                estimated_calls_saved=hits,
            )
        )

    eligible = [p for p in points if p.false_hit_rate <= max_false_hit_rate]
    if eligible:
        best = max(eligible, key=lambda p: p.hit_rate)
        reason = (
            f"highest hit rate ({best.hit_rate:.1%}) among thresholds whose false-hit rate "
            f"stays at or below {max_false_hit_rate:.1%}; at this setting "
            f"{best.estimated_calls_saved} of {best.n_queries} model calls are avoided"
        )
        return CacheSweep(points, getattr(embedder, "name", "?"), quality_floor, best.threshold, reason)

    strictest = max(points, key=lambda p: p.threshold)
    return CacheSweep(
        points,
        getattr(embedder, "name", "?"),
        quality_floor,
        strictest.threshold,
        f"no threshold met the false-hit budget of {max_false_hit_rate:.1%}; "
        f"recommending the strictest tested ({strictest.threshold}) and treating the "
        f"semantic cache as unsafe for this traffic without a verification step",
    )


def build_query_log(
    items: list[tuple[str, str]],
    repeats: int = 3,
    paraphrase_rate: float = 0.5,
    seed: int = 5,
) -> list[tuple[str, str]]:
    """Synthesise a realistic repetitive query log from a Q/A set.

    Real traffic repeats: the same question returns, sometimes verbatim,
    sometimes reworded. Without repeats there is nothing for a cache to hit and
    the sweep is meaningless, so when no production log is available this builds
    a stand-in and the report labels it as synthetic.
    """
    import random

    rng = random.Random(seed)
    log: list[tuple[str, str]] = []
    for _ in range(repeats):
        for question, answer in items:
            text = question
            if rng.random() < paraphrase_rate:
                text = _paraphrase(question, rng)
            log.append((text, answer))
    rng.shuffle(log)
    return log


_PREFIXES = ["", "can you tell me ", "quick question: ", "i want to know ", "please explain "]
_SUFFIXES = ["", "?", " exactly?", " in short", " for the 2019 cohort"]


def _paraphrase(question: str, rng) -> str:
    base = question.rstrip("?")
    base = base.replace("What is", "what's").replace("How does", "how do")
    return f"{rng.choice(_PREFIXES)}{base}{rng.choice(_SUFFIXES)}"
