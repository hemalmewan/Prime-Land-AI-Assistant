"""
cache_effectiveness.py
==========================================================

Cache-Augmented Generation (CAG) Evaluation Module

This module evaluates cache effectiveness by simulating
multiple queries and measuring:

    - Cache Hit Rate
    - Average Latency
    - Latency Improvement
    - Estimated Cost Savings from avoided LLM calls

Designed for performance and cost analysis of AI systems.

Evaluation Flow:
----------------
1. Send query to CAG service.
2. Detect if response came from cache or RAG/LLM.
3. Record latency.
4. Track avoided API calls.
5. Compute summary statistics.

Use Case:
---------
Academic benchmarking or production performance testing.
"""

##========================================
## Import Required Libraries
##========================================
import time
import random
import pandas as pd
from typing import List,Dict,Any

class CAGEvaluator:
    """
    Cache-Augmented Generation Evaluator.

    Evaluates the effectiveness of a CAG system by
    simulating multiple queries.

    Metrics:
        - Hit Rate
        - Avg Latency (Cache vd RAG)
        - Latency Improvement %
        - Estimated Cost Savings

    Prameters

    cag_service : Any
        Your CAG service instance with generate_response().
    
    cost_per_llm_call : float
        Estimated API cost per non-caches generation.
    
    """

    def __init__(
            self,
            cag_service,
            cost_per_llm_call:float=0.002
    ):
        self.cag_service=cag_service
        self.cost_per_llm_call=cost_per_llm_call
    
    
    def evaluate(
            self,
            queries:List[str],
            simulate_n:int=100
    )->Dict[str,Any]:
        
        """
        Simulate queries and compute cache effectiveness metrics.
        
        Parameters:
        ===================
        queries: List[str]
            Base list of queries to sample from.
        
        simulate_n : int
             Number of simulated queries.
        
        Returns:
        ====================

        Dict[str,Any]
            Dictionary containing evaluation metrics.
      
        """

        total_latency=0
        cache_latency=[]
        rag_latency=[]

        cache_hits=0
        llm_calls=0

        simulated_queries=[
            random.choice(queries) for _ in range(simulate_n)
        ]

        for query in simulated_queries:
            
            start=time.time()
            response=self.cag_service.generate_response(query=query)
            latency=time.time()-start

            total_latency+=latency

            if response["source"]=="cache":
                cache_hits+=1
                cache_latency.append(latency)
            else:
                llm_calls+=1
                rag_latency.append(latency)
        
        hit_rate=cache_hits/simulate_n

        avg_latency_total=total_latency/simulate_n
        avg_cache_latency=(
            sum(cache_latency)/len(cache_latency)
            if cache_latency else 0
        )

        avg_rag_latency=(
            sum(rag_latency)/len(rag_latency)
            if rag_latency else 0
        )

        latency_improvement=(
            ((avg_rag_latency-avg_cache_latency)/avg_rag_latency)*100
            if avg_rag_latency>0 else 0
        )

        cost_without_cache = simulate_n * self.cost_per_llm_call
        cost_with_cache = llm_calls * self.cost_per_llm_call
        cost_saved = cost_without_cache - cost_with_cache

        return {
            "Total Queries": simulate_n,
            "Cache Hits": cache_hits,
            "Cache Hit Rate": round(hit_rate, 3),
            "Avg Total Latency (s)": round(avg_latency_total, 3),
            "Avg Cache Latency (s)": round(avg_cache_latency, 3),
            "Avg RAG Latency (s)": round(avg_rag_latency, 3),
            "Latency Improvement (%)": round(latency_improvement, 2),
            "Estimated Cost Without Cache ($)": round(cost_without_cache, 4),
            "Estimated Cost With Cache ($)": round(cost_with_cache, 4),
            "Estimated Cost Saved ($)": round(cost_saved, 4),
        }

