"""Quality scorers. The gate is only as trustworthy as these are."""

from __future__ import annotations

from vantage.models import QualitySummary
from vantage.quality.harness import compare_quality
from vantage.quality.scorers import (
    citation_grounding,
    exact_match,
    number_agreement,
    score_item,
    token_f1,
)


def test_exact_match_ignores_punctuation_articles_and_case():
    assert exact_match("The value was 30 ng/mL.", "value was 30 ng/mL") == 1.0
    assert exact_match("30", "31") == 0.0


def test_token_f1_rewards_overlap():
    assert token_f1("mean recovery was 94%", "mean recovery was 94%") == 1.0
    assert 0.0 < token_f1("recovery was 94%", "the mean recovery was 94% overall") < 1.0
    assert token_f1("completely different text", "nothing alike here") == 0.0


def test_citation_grounding_punishes_invented_sources():
    assert citation_grounding("answer [2]", [1, 2, 3], "ref [2]") == 1.0
    assert citation_grounding("answer [9]", [1, 2, 3], "ref [2]") == 0.0
    assert citation_grounding("answer [1] and [9]", [1, 2, 3], "ref [1]") == 0.5


def test_citation_grounding_punishes_missing_citation():
    assert citation_grounding("an uncited claim", [1, 2], "ref [1]") == 0.0
    # ...but not when the reference itself cites nothing.
    assert citation_grounding("an uncited claim", [1, 2], "plain reference") == 1.0


def test_number_agreement_catches_a_hallucinated_figure():
    assert number_agreement("the CV was 6.4%", "the CV was 6.4%") == 1.0
    assert number_agreement("the CV was 9.1%", "the CV was 6.4%") == 0.0
    # Stating no number is not the same as stating a wrong one.
    assert number_agreement("the CV was acceptable", "the CV was 6.4%") == 1.0


def test_composite_is_bounded_and_ordered():
    good = score_item("a", "Mean recovery was 94.0% [1].", "Mean recovery was 94.0% [1].", [1, 2])
    bad = score_item("a", "Mean recovery was 61.0%.", "Mean recovery was 94.0% [1].", [1, 2])
    assert good.composite == 1.0
    assert 0.0 <= bad.composite < good.composite


def _summary(cid: str, scores: list[float]) -> QualitySummary:
    return QualitySummary(
        config_id=cid, n=len(scores), exact_match=0.0, token_f1=0.0,
        citation_grounding=0.0, number_agreement=0.0,
        composite=sum(scores) / len(scores),
        per_item=[{"item_id": f"i{i}", "composite": s} for i, s in enumerate(scores)],
    )


def test_gate_passes_when_the_difference_is_noise():
    base = _summary("base", [0.80, 0.82, 0.78, 0.81, 0.79, 0.83])
    cand = _summary("cand", [0.79, 0.83, 0.79, 0.80, 0.80, 0.82])
    assert compare_quality(base, cand).verdict == "pass"


def test_gate_fails_on_a_consistent_real_regression():
    base = _summary("base", [0.90] * 20)
    cand = _summary("cand", [0.70] * 20)
    result = compare_quality(base, cand, max_drop=0.02)
    assert result.verdict == "fail"
    assert result.significant
    assert result.delta < 0


def test_gate_warns_inside_the_budget():
    base = _summary("base", [0.900] * 30)
    cand = _summary("cand", [0.890] * 30)
    assert compare_quality(base, cand, max_drop=0.02).verdict == "warn"
