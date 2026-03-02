"""
crag_correction_impact.py

This module evaluates the impact of Corrective Retrieval-Augmented
Generation (CRAG) compared to standard RAG.

Metrics evaluated:
- Correction frequency
- Confidence score improvement
- Answer quality improvement
- Overall gain percentage
"""

##=====================================
## Import Required Libraries
##=====================================
from typing import List, Dict, Any
import time
import pandas as pd
from src.context_engineering.config import CONFIDENCE_THRESHOLD


class CRAGEvaluator:
    """
    Evaluate the performance difference between RAG and CRAG.

    Attributes:
        rag_pipeline: Callable standard RAG system.
        crag_pipeline: Callable CRAG system with correction.
        confidence_threshold (float): Threshold to trigger correction.
    """

    def __init__(
        self,
        rag_pipeline,
        crag_pipeline,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.rag_pipeline = rag_pipeline
        self.crag_pipeline = crag_pipeline
        self.confidence_threshold = confidence_threshold

    def evaluate(self, queries: List[str]) -> pd.DataFrame:
        """
        Evaluate RAG vs CRAG over a set of queries.

        Args:
            queries (List[str]): List of test queries.

        Returns:
            pd.DataFrame: Evaluation results per query.
        """

        results = []

        correction_count = 0

        for query in queries:

            # ---------- Run Standard RAG ----------
            rag_start = time.time()
            rag_result = self.rag_pipeline.generate_response(query,verbose=True)
            rag_latency = time.time() - rag_start

            rag_answer = rag_result["answer"]
            rag_confidence = rag_result.get("confidence_score", 0.5)

            # ---------- Run CRAG ----------
            crag_start = time.time()
            crag_result = self.crag_pipeline.generate_crag_response(query)
            crag_latency = time.time() - crag_start

            crag_answer = crag_result["answer"]
            crag_confidence = crag_result.get("confidence_final", 0.5)
            corrected = crag_result.get("correction_applied", False)

            if corrected:
                correction_count += 1

            # ---------- Quality Improvement ----------
            quality_gain = crag_confidence - rag_confidence

            results.append(
                {
                    "query": query,
                    "rag_confidence": rag_confidence,
                    "crag_confidence": crag_confidence,
                    "confidence_gain": quality_gain,
                    "rag_latency": rag_latency,
                    "crag_latency": crag_latency,
                    "correction_triggered": corrected,
                }
            )

        df = pd.DataFrame(results)

        # Summary Metrics
        summary = {
            "total_queries": len(queries),
            "correction_frequency": correction_count / len(queries),
            "avg_confidence_gain": df["confidence_gain"].mean(),
            "avg_rag_latency": df["rag_latency"].mean(),
            "avg_crag_latency": df["crag_latency"].mean(),
        }

        print("\n===== CRAG Evaluation Summary =====")
        for k, v in summary.items():
            print(f"{k}: {v:.2f}")

        return df