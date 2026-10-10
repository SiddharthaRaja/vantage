from __future__ import annotations

import asyncio

from vantage.backends.simulated import SimulatedBackend
from vantage.cache.analysis import build_query_log, sweep_thresholds
from vantage.cache.embedders import HashingEmbedder
from vantage.cache.semantic import CachedBackend, ExactCache, SemanticCache
from vantage.models import GenRequest


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder()
    a = emb.embed(["what is the reference range for 25(OH)D?"])
    b = emb.embed(["what is the reference range for 25(OH)D?"])
    assert (a == b).all()
    assert abs(float((a[0] * a[0]).sum()) - 1.0) < 1e-5


def test_near_duplicates_are_closer_than_unrelated_text():
    emb = HashingEmbedder()
    v = emb.embed(
        [
            "what is the reference range for 25(OH)D?",
            "what's the reference range of 25(OH)D",
            "how many participants were in the cohort?",
        ]
    )
    near = float(v[0] @ v[1])
    far = float(v[0] @ v[2])
    assert near > far + 0.2


def test_exact_cache_hits_only_on_normalised_equality():
    cache = ExactCache()
    cache.put("What is X?", "X is 5.")
    assert cache.lookup("  what   is x? ").hit
    assert not cache.lookup("What is Y?").hit
    assert cache.stats()["hits"] == 1


def test_exact_cache_evicts_least_recently_used():
    cache = ExactCache(capacity=2)
    cache.put("a", "1")
    cache.put("b", "2")
    cache.lookup("a")           # 'a' becomes most recent
    cache.put("c", "3")         # evicts 'b'
    assert cache.lookup("a").hit
    assert not cache.lookup("b").hit


def test_semantic_cache_threshold_controls_hits():
    strict = SemanticCache(threshold=0.999, exact_first=False)
    loose = SemanticCache(threshold=0.50, exact_first=False)
    for cache in (strict, loose):
        cache.put("what is the limit of detection for the CLIA assay?", "0.9 ng/mL [1].")
    probe = "what's the limit of detection of the CLIA assay"
    assert not strict.lookup(probe).hit
    assert loose.lookup(probe).hit


def test_cached_backend_marks_hits_and_collapses_ttft():
    async def run():
        backend = CachedBackend(SimulatedBackend(max_batch=4), SemanticCache(threshold=0.9))
        req = GenRequest(id="r1", prompt="Sources: [1] x\n\nQuestion: what is x?",
                         max_tokens=24, meta={"question": "what is x?"})
        first = await backend.generate(req)
        second = await backend.generate(req.model_copy(update={"id": "r2"}))
        return first, second

    first, second = asyncio.run(run())
    assert not first.cache_hit
    assert second.cache_hit and second.cache_kind == "exact"
    assert second.e2e_s < first.e2e_s
    assert second.text == first.text


def test_threshold_sweep_trades_hit_rate_against_false_hits():
    pairs = [
        ("What was the recovery for the CLIA assay?", "Recovery was 94.0% [1]."),
        ("How many participants were enrolled?", "212 participants were enrolled [2]."),
        ("What was the limit of detection?", "The limit of detection was 0.8 ng/mL [1]."),
        ("What turnaround time was reported?", "Turnaround was 18 minutes per sample [3]."),
    ]
    log = build_query_log(pairs, repeats=6, paraphrase_rate=0.6)
    sweep = sweep_thresholds(log, thresholds=[0.5, 0.7, 0.9, 0.99])
    rates = [p.hit_rate for p in sweep.points]
    # Hit rate is monotonically non-increasing as the threshold tightens.
    assert rates == sorted(rates, reverse=True)
    assert sweep.points[0].false_hit_rate >= sweep.points[-1].false_hit_rate
    assert sweep.recommended_threshold is not None
