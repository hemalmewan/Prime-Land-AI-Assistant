"""
    cag_cache.py

    Sementic Cache Layer for Cache-Augmented Generation (CAG)

    1. Warm (pre-load) FAQ data into a vector-based semantic cache.
    2. Perform sementic similarity lookup against cached entries.

    The cache is implemented using a dedicated Qdrant collection
    and supports high-speed semeatic retrieval for frequently asked questions.

    Intended Usage:

        - Used as a fast pre-check layer before triggering RAG.
        - Reduces latency and LLM usage cost.
        - Improves response consistency for FAQs.
    
    Dependencies:

        - Qdrant Client Library
        - OpenAI Embedding Models
        - UUID for unique point IDs

"""

##=====================================
## Import Required Libraries
##====================================
import uuid
from typing import List,Dict,Any,Optional
from qdrant_client.models import PointStruct,Filter,FieldCondition,Range
from datetime import datetime
from src.context_engineering.config import(
    load_faq,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL,
    MAX_CACHE_SIZE,
    HISTORY_TTL_HOURS
)


def warm_faq_cache(
        faq_list:dict,
        rag_service,
        embedding_model,
        client,
        collection_name:str
)->None:
    """
      Warm semantic cache using FAQ YAML (question-only format).

    This function:
        1. Iterates through FAQ categories.
        2. Generates answers using RAG.
        3. Embeds the FAQ question.
        4. Stores question + generated answer in Qdrant cache.

    Args:
        faq_dict (dict): YAML-loaded FAQ dictionary.
        rag_service: Initialized RAG service.
        embedding_model: Embedding model instance.
        client: Qdrant client.
        collection_name (str): Cache collection name.

    Returns:
        None
    """

    for category,questions in faq_list.items():
        for question in questions:
            
            print(f"Generating answer for FAQ:{question}")

            ##generating answer using RAG
            rag_result=rag_service.generate_response(question)
            answer=rag_result["answer"]

            ## embedding auestion
            vector=embedding_model.embed_query(question)

            ## store in chache
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "question":question,
                            "answer":answer,
                            "category":category,
                            "type":"faq",
                            "created_at":datetime.utcnow().timestamp()
                        }
                    )

                ]
            )

    print("✅ FAQ cache warming completed.")



def check_cache(
        query:str,
        embedding_model,
        client,
        collection_name:str,
        threshold:float=CACHE_SIMILARITY_THRESHOLD,
        cache_ttl:int=CACHE_TTL

)->Dict[str,Any]:
    """
     Perform semantic similarity search against the cache.

     The function retrieves the most similar cached entry
     and returns a chache hit only if similarity score exceeds
     the defined threshold.

     Args:
        query (str):
            user query.
        
        embedding_model:
            Embedding model instance used for query embedding.

        Client:
            Initialized Qdrant client.
        
        Collection_name (str):
            Cache collection name.
        
        threshold (float,optional):
            Minimum cosine similarity score required
            to consider a cache hit.Default is 0.90.
    
    Returns:
        Dict[str,Any]:
            {
              "hit":bool,
              "answer":str | None,
              "score": float | None
            
            }
    
    
    """

    query_vector=embedding_model.embed_query(query)

    results=client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1
    )

    if results.points:
        top_result=results.points[0]

        ## TTL check
        created_at=top_result.payload.get("created_at")
        current_time=datetime.utcnow().timestamp()

        if created_at and (current_time-created_at)>cache_ttl:
            print("Cache expired ❌")
            return {"hit":False}

        if top_result.score>=threshold:
            return{
                "hit":True,
                "answer":top_result.payload.get("answer"),
                "score":top_result.score
            }
    
    return{
        "hit":False,
        "answer":None,
        "score":None
    }


def enforce_max_cache_size(
        client,
        collection_name:str,
        max_cache_size:int=MAX_CACHE_SIZE
)->Optional[int]:
    """
    Enforce maximum cache size by removing oldest entries if limit exceed.

    This function:
        1. Counts current number of cached entries.
        2. Compares it with 'max_cache_size'.
        3. If exceeded,deletes the oldest records based on the 'created_at'
        timestamp.
    
    Args:
        client (QdrantClient):
            Initialized Qdrant client instance.
        
        collection_name (str):
            Name of the sementic cache collection.
        
        max_cache_size (int):
            Maximum allowed number of caches entries.
    
    Returns:
        Optional[int]:
            Number of deleted entries if cleanup occured.
            None if no deletion was necessary.
    
    """

    ## get current cache size
    count_result=client.count(collection_name=collection_name)
    current_size=count_result.count

    if current_size<=max_cache_size:
        return None ## No need to take an action
    
    ## number of exceed entries to remove
    exceed_entries=(current_size-max_cache_size)

    ## Retrive oldest entries
    oldest_points=client.scroll(
        collection_name=collection_name,
        limit=exceed_entries,
        with_payload=True,
        with_vectors=False,
        scroll_filter=None,
        order_by="created_at" ## payload index
    )

    points_to_delete=[point.id for point in oldest_points[0]]

    if points_to_delete:
        client.delete(
            collection_name=collection_name,
            points_selector=points_to_delete
        )
    
    return len(points_to_delete)