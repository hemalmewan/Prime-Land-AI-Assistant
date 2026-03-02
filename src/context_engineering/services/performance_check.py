"""
  Chunking Strategy Evaluation Module

  This script evaluates the performance and storage metrics of various text chunking
  strategies ingested into a Qdratnt vector database.It is designed to generate a comparison table that measures:
    1.Total Chunk Count(retrieved via Qdrant ount API)
    2.Average Tokens size(calculated from raw chunk data)
    3.Estimated Index Size in MB(mathematical estimation based on vector dimensions)
    4.Retrieval Time in milliseconds(measured via Qdrant search API)

    The output is a pandas DataFrame suitable for final reporting and analysis in a Jupyter Notebook environment.
"""

## ===================================
## Import Required Libraries
## ===================================
import time
import pandas as pd
from typing import List,Dict,Any
from qdrant_client.http.models import Filter,FieldCondition,MatchValue
def generate_comparison_table(
        qdrant_client,
        collection_name:str,
        embedding_model,
        total_chunks:List[Dict[str,Any]]
        )->pd.DataFrame:
    """
    Generate a performance compraison table for different chunking strategies.
    
    This fucntion queries the Qdrant database and processes the raw chunk list
    to computee evaluation metrics for each implemented chunking strategy
    (Fixed,Sementic,Sliding,Parent-Child and Late-Chunking).

    Args:
        qdrant_client: An instance of the Qdrant client to interact with the vector database.
        collection_name (str): The name of the Qdrant collection where the chunks are stored.
        embedding_model: The embedding model used to generate vector representations of the chunks.
        total_chunks (List[Dict[str,Any]]): A list of all chunk dictionaries ingested into Qdrant, used for token size calculations.

    Returns:
        pd.DataFrame: A DataFrame containing the performance metrics for each chunking strategy
        (total chunk count, average tokens size, estimated index size in MB, and retrieval time in milliseconds).
    
    """

    print("🔍 Gathering metrics for all strategies... Please wait.\n")

    ## calculate the average token size from raw chunk list usinf Pandas
    df_chunks=pd.DataFrame(total_chunks)

    ## define the chunking strategies to evaluate
    strategies=[
        "Fixed Strategy",
        "Sementic Strategy",
        "Sliding Strategy",
        "Child Strategy",
        "Late Strategy"

    ]

    metrics=[] ## to store the metrics for each strategy

    ## create a dimmy query vector to measure retrieval time.
    query_text="Looking for a 3 bedroom luxury house in Colombo under 50,000,000"
    query_vector=embedding_model.embed_query(query_text)

    ## define the default dimentions for open-ai embedding model (like text-embedding-3-small)
    VECTOR_DIMENSIONS=3072

    for strategy in strategies:
        ## get the average token count for this specific strategy
        strategy_chunks=df_chunks[df_chunks['chunking_strategy']==strategy]
        if strategy_chunks.empty:
            continue ## skip if no chunks for this strategy

        avg_tokens=strategy_chunks["token_count"].mean()

        ## chunk count
        strategy_filter=Filter(
            must=[FieldCondition(key="metadata.chunking_strategy",match=MatchValue(value=strategy))]
        
        )

        count_result=qdrant_client.count(
            collection_name=collection_name,
            count_filter=strategy_filter
        )
        chunk_count=count_result.count

        ## Measure exactly how long Qdrant takes to filter and search 10 records
        start_time=time.perf_counter()
        _=qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=strategy_filter,
            limit=5

        )

        end_time=time.perf_counter()
        retrieval_time_ms=(end_time-start_time)*1000  ## convert to milliseconds

        ## Estimate index size in MB
        estimated_size_mb=(chunk_count*VECTOR_DIMENSIONS*4)/(1024*1024) ## 4 bytes per float32 vector element


        metrics.append({
            "Strategy": strategy,
            "Chunk Count": chunk_count,
            "Avg Size(Tokens)": round(avg_tokens,2),
            "Estimated(Index )Size(MB)": round(estimated_size_mb,2),
            "Retrieval Time(ms)": round(retrieval_time_ms,2)
        })

    ## make the results as DataFrame
    df_metrics=pd.DataFrame(metrics)

    ## Sort the DataFrame by Retrieval Time
    df_metrics=df_metrics.sort_values(by="Retrieval Time(ms)")

    return df_metrics
