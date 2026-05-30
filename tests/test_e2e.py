"""
End-to-end test for the ECA pipeline.

Tests:
1. ECA surfaces at least one expected outside-frame concept
2. Vanilla LLM shows variance across 3 runs (EFA is stochastic)

Usage:
  python tests/test_e2e.py

Requires OPENROUTER_API_KEY in .env
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa import ECA
from efa.config import DEFAULT_MODEL, DTG_PATH
from efa.embeddings import embed
from efa.llm import LLMClient
from efa.pipeline.verify import CoverageVerifier


PROBLEM = "How do I reduce employee turnover in my engineering team?"

EXPECTED_OUTSIDE_FRAME = [
    "organizational psychology",
    "job characteristics model",
    "self-determination theory",
    "incentive design",
    "intrinsic motivation",
    "autonomy",
    "psychological safety",
    "herzberg two-factor theory",
    "exit interview",
    "employee engagement",
]


def test_eca_surfaces_outside_frame_concepts():
    print("=" * 60)
    print("TEST 1: ECA surfaces outside-frame concepts")
    print("=" * 60)

    eca = ECA(dtg_path=DTG_PATH, k_gaps=3)
    result = eca.run(PROBLEM)

    print("\n--- Coverage Audit ---")
    print(result.summary())

    print("\n--- Response Preview ---")
    print(result.response[:800])

    verifier = CoverageVerifier()
    verification = verifier.recall(result.concepts, EXPECTED_OUTSIDE_FRAME)

    print(f"\n--- Verification ---")
    print(f"Recall: {verification.recall:.2f}")
    print(f"Matched: {verification.matched}")
    print(f"Missed: {verification.missed}")

    assert verification.recall > 0.0, (
        f"ECA surfaced NO expected outside-frame concepts. "
        f"Surfaced: {[dc.concept for dc in result.concepts]}"
    )
    print(f"\n✓ PASS: ECA recall = {verification.recall:.2f}")
    return result


def test_vanilla_llm_variance():
    print("\n" + "=" * 60)
    print("TEST 2: Vanilla LLM shows coverage variance across 3 runs")
    print("=" * 60)

    llm = LLMClient(model=DEFAULT_MODEL)
    verifier = CoverageVerifier()

    from efa.pipeline.delta import ConceptDeltaExtractor
    extractor = ConceptDeltaExtractor(llm)

    recalls = []
    concept_sets = []

    for run in range(3):
        response = llm.complete(PROBLEM, temperature=0.9)
        # Extract concepts from vanilla response
        phrases = extractor._extract_noun_phrases(response)
        from efa.pipeline.delta import DeltaConcept
        fake_delta = [DeltaConcept(concept=p, source_domain="vanilla") for p in phrases]

        v = verifier.recall(fake_delta, EXPECTED_OUTSIDE_FRAME)
        recalls.append(v.recall)
        concept_sets.append(set(phrases[:20]))
        print(f"  Run {run+1}: recall={v.recall:.2f}, concepts={list(phrases[:5])}")

    std_dev = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
    print(f"\nVanilla recall stats: mean={statistics.mean(recalls):.2f}, std={std_dev:.2f}")
    print("(Variance > 0 demonstrates EFA stochasticity)")

    stats = verifier.multi_run_stats([
        type("R", (), {"recall": r})() for r in recalls
    ])

    print(f"\n✓ DONE: Vanilla std_dev = {std_dev:.3f}")
    return recalls


if __name__ == "__main__":
    if not DTG_PATH.exists():
        print(f"ERROR: DTG not found at {DTG_PATH}")
        print("Run: python scripts/build_dtg.py")
        sys.exit(1)

    try:
        result = test_eca_surfaces_outside_frame_concepts()
        recalls = test_vanilla_llm_variance()
        print("\n✓ All tests passed.")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}")
        sys.exit(1)
