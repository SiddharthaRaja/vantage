"""Answer-quality scoring.

A serving benchmark without a quality gate is a benchmark of how fast you can
return the wrong thing. Every optimisation in this project -- quantisation,
batching, caching, swapping in a smaller fine-tuned model -- trades quality for
speed somewhere. The job of this module is to make that trade visible and
quantified rather than assumed.

Four complementary scorers, cheap and deterministic, no LLM judge required:

  exact_match        strict, mostly useful for short extractive answers
  token_f1           the SQuAD-style overlap score; robust to phrasing
  citation_grounding did it cite, and are the citations real? A RAG answer with
                     no citation is a regression even if the prose is right.
  number_agreement   do the numbers in the answer appear in the reference?
                     In a biomedical corpus a hallucinated figure is the single
                     most dangerous failure, and token F1 barely notices it.

An optional LLM judge is available for nuance, off by default, because a judge
you cannot afford to run on every CI commit is not a gate.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CITATION = re.compile(r"\[(\d+)\]")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def citations(text: str) -> list[int]:
    return [int(m) for m in _CITATION.findall(text)]


def citation_grounding(prediction: str, source_ids: list[int], reference: str = "") -> float:
    """1.0 when the answer cites, and every citation points at a source that
    was actually in the prompt. 0.0 when it cites nothing while the reference
    does, or when it invents a source number.

    Inventing [7] when only [1]-[3] were supplied is the clearest machine-
    checkable hallucination signal a RAG system has, and it costs nothing to
    check.
    """
    cited = citations(prediction)
    expected = bool(citations(reference)) if reference else True
    if not cited:
        return 0.0 if expected else 1.0
    if not source_ids:
        return 1.0
    valid = [c for c in cited if c in source_ids]
    return len(valid) / len(cited)


def number_agreement(prediction: str, reference: str, tolerance: float = 0.001) -> float:
    """Share of numbers in the answer that also appear in the reference.

    Returns 1.0 when the answer states no numbers at all -- absence of a claim
    is not a false claim. The metric only punishes numbers that were invented.
    """
    pred_numbers = [float(x) for x in _NUMBER.findall(prediction)]
    if not pred_numbers:
        return 1.0
    ref_numbers = [float(x) for x in _NUMBER.findall(reference)]
    if not ref_numbers:
        return 0.0
    ok = 0
    for value in pred_numbers:
        if any(abs(value - r) <= max(tolerance, abs(r) * tolerance) for r in ref_numbers):
            ok += 1
    return ok / len(pred_numbers)


@dataclass
class ItemScore:
    item_id: str
    exact_match: float
    token_f1: float
    citation_grounding: float
    number_agreement: float
    composite: float
    prediction: str = ""
    reference: str = ""
    detail: dict[str, float] = field(default_factory=dict)


# Weights: token F1 carries most of the signal, but a hallucinated number or a
# missing citation is weighted heavily enough to sink an otherwise fluent
# answer. These are a judgement call and are stated in the report rather than
# buried, so a reader can re-weight them.
COMPOSITE_WEIGHTS = {
    "token_f1": 0.45,
    "citation_grounding": 0.25,
    "number_agreement": 0.25,
    "exact_match": 0.05,
}


def score_item(
    item_id: str,
    prediction: str,
    reference: str,
    source_ids: list[int] | None = None,
) -> ItemScore:
    em = exact_match(prediction, reference)
    f1 = token_f1(prediction, reference)
    cg = citation_grounding(prediction, source_ids or [], reference)
    na = number_agreement(prediction, reference)
    composite = (
        COMPOSITE_WEIGHTS["token_f1"] * f1
        + COMPOSITE_WEIGHTS["citation_grounding"] * cg
        + COMPOSITE_WEIGHTS["number_agreement"] * na
        + COMPOSITE_WEIGHTS["exact_match"] * em
    )
    return ItemScore(
        item_id=item_id,
        exact_match=em,
        token_f1=f1,
        citation_grounding=cg,
        number_agreement=na,
        composite=composite,
        prediction=prediction,
        reference=reference,
    )


JUDGE_PROMPT = """You are grading an answer against a reference answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {prediction}

Score the candidate from 0 to 10 on factual agreement with the reference.
A candidate that adds unsupported claims scores below 5 no matter how fluent.
Reply with a single integer and nothing else."""
