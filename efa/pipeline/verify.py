"""Component 9: Coverage Verifier.

Measures recall of outside-frame concepts against ground truth.
Designed for multi-run averaging — the key metric is consistency (mean ± std dev),
not single-run binary pass/fail.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from efa.config import CONCEPT_MATCH_THRESHOLD
from efa.embeddings import embed
from efa.pipeline.delta import DeltaConcept


@dataclass
class VerificationResult:
    recall: float
    precision: float
    matched: list[str]
    missed: list[str]
    run_count: int = 1


class CoverageVerifier:
    def __init__(self, threshold: float = CONCEPT_MATCH_THRESHOLD):
        self.threshold = threshold

    def recall(
        self,
        surfaced_concepts: list[DeltaConcept],
        ground_truth: list[str],
    ) -> VerificationResult:
        """
        Match surfaced concepts against ground truth.
        Match = cosine_sim(embed(concept), embed(gt)) > threshold
              OR exact phrase match (case-insensitive).

        Returns recall and precision for this run.
        """
        if not ground_truth:
            return VerificationResult(recall=1.0, precision=1.0, matched=[], missed=[])

        gt_vecs = embed.encode(ground_truth)
        surfaced_texts = [dc.concept for dc in surfaced_concepts]
        surfaced_vecs = embed.encode(surfaced_texts) if surfaced_texts else np.array([])

        matched: list[str] = []
        missed: list[str] = []

        for gt_text, gt_vec in zip(ground_truth, gt_vecs):
            gt_lower = gt_text.lower()
            found = False

            # Exact phrase match
            if any(gt_lower in s.lower() or s.lower() in gt_lower for s in surfaced_texts):
                found = True

            # Cosine match
            if not found and len(surfaced_vecs) > 0:
                sims = surfaced_vecs @ gt_vec
                if float(np.max(sims)) >= self.threshold:
                    found = True

            (matched if found else missed).append(gt_text)

        recall_val = len(matched) / len(ground_truth)

        # Precision: fraction of surfaced concepts that match any ground truth
        if surfaced_texts and len(gt_vecs) > 0:
            precision_hits = 0
            for s_text, s_vec in zip(surfaced_texts, surfaced_vecs):
                s_lower = s_text.lower()
                if any(s_lower in gt.lower() or gt.lower() in s_lower for gt in ground_truth):
                    precision_hits += 1
                    continue
                sims = gt_vecs @ s_vec
                if float(np.max(sims)) >= self.threshold:
                    precision_hits += 1
            precision_val = precision_hits / len(surfaced_texts)
        else:
            precision_val = 0.0

        return VerificationResult(
            recall=recall_val,
            precision=precision_val,
            matched=matched,
            missed=missed,
        )

    def multi_run_stats(self, results: list[VerificationResult]) -> dict:
        """Compute mean ± std dev recall across multiple runs."""
        recalls = [r.recall for r in results]
        return {
            "mean_recall": statistics.mean(recalls),
            "std_recall": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            "min_recall": min(recalls),
            "max_recall": max(recalls),
            "run_count": len(recalls),
            "consistency": 1.0 - (statistics.stdev(recalls) if len(recalls) > 1 else 0.0),
        }
