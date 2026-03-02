"""
chunking_evaluator.py
==============================================================

Chunking Strategy Evaluation Module

This module provides utilities to evaluate multiple chunking
strategies.

The evaluation is performed across multiple queries using:

    - Precision@K
    - Recall@K
    - Answer Relevance
    - Latency

Each chunking strategy is filtered via metadata stored in
Qdrant (metadata.chunking_strategy) and evaluated independently.

Designed for research-grade comparison of chunking methods.
"""

##======================================
## Import Required Libraries
##======================================
from typing import List,Dict,Any
from qdrant_client import QdrantClient
import pandas as pd
from qdrant_client.models import FieldCondition,MatchValue,Filter
from src.context_engineering.config import TOP_K
import time


def evaluation(
    queries: List[str],
    collection_name: str,
    client: QdrantClient,
    embedding_model,
    ground_truth: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Evaluate multiple chunking strategies using retrieval metrics.

    Parameters
    ----------
    queries : List[str]
        List of evaluation queries.

    collection_name : str
        Qdrant collection name.

    client : QdrantClient
        Active Qdrant client instance.

    embedding_model : Any
        Embedding model with `embed_query()` method.

    ground_truth : Dict[str, List[str]]
        Dictionary mapping query -> list of relevant document IDs.

    Returns
    -------
    pd.DataFrame
        Aggregated evaluation results for each chunking strategy
        including Precision@K, Recall@K, Relevance, and Latency.
    """

    strategies = [
        "Fixed Strategy",
        "Semantic Strategy",
        "Sliding Strategy",
        "Child Strategy",
        "Late Strategy"
    ]

    final_results = []

    for strategy in strategies:

        precision_scores = []
        recall_scores = []
        latency_scores = []

        for query in queries:

            query_vector = embedding_model.embed_query(query)

            strategy_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.chunking_strategy",
                        match=MatchValue(value=strategy),
                    )
                ]
            )

            start_time = time.time()

            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=strategy_filter,
                limit=TOP_K
            )

            latency = time.time() - start_time
            latency_scores.append(latency)

            retrieved_ids = [
                point.id for point in results.points
            ]

            relevant_ids = ground_truth.get(query, [])

            # Precision@K
            relevant_retrieved = len(
                set(retrieved_ids).intersection(set(relevant_ids))
            )

            precision = relevant_retrieved / TOP_K if TOP_K > 0 else 0
            recall = (
                relevant_retrieved / len(relevant_ids)
                if len(relevant_ids) > 0
                else 0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)

        final_results.append({
            "Strategy": strategy,
            "Precision@K": sum(precision_scores) / len(precision_scores),
            "Recall@K": sum(recall_scores) / len(recall_scores),
            "Latency": sum(latency_scores) / len(latency_scores),
        })

    return pd.DataFrame(final_results)