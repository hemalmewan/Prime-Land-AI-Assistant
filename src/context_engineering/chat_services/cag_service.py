"""
cag_service.py
======================================

Cache-Augmented Generation (CAG) Service Layer.

This module defines the CAGService class which wraps an
existing RAG service with a semantic cache pre-check layer.

Workflow:
    1. Receive user query.
    2. Enforce max cache size.
    3. Check semantic cache (FAQ memory).
    4. If cache hit -> return cached response.
    5. If cache miss -> run RAG pipeline.
    6. Return structured response with source indicator.

Benefits:
    - Reduces LLM usage cost
    - Improves latency
    - Guarantees consistent answers for FAQs
    - Production-ready modular design
"""

from typing import Dict,Any,Optional
from .cag_cache import (check_cache,
                       warm_faq_cache,
                       enforce_max_cache_size)
import time

class QdrantCAGService:
    """
      Cache-Augmented Generation Service

      This class enahaces a RAG pipeline by introducing a
      semantic chache layer before LLM execution.

      Attributes:
        rag_service:
            Existing RAG service instance responsible for
            retrieval and answer generation.
        
        embedding_model:
            Embedding model used for sementic similarity search.
        
        client:
            Qdrant client instance.
        
        cache_collection (str):
            Name of the semantic cache collection.
        
        threshold (float):
            similarity threshold for cache matching.
        
        cache_ttl (int):
            Time-to-live for dynamic cache entries (seconds).
        
        max_cache_size (int):
            Maximum number of cache entries allowd.
    """
    def __init__(
            self,
            rag_service,
            embedding_model,
            client,
            cache_collection:str,
            ):
        
        self.rag_service=rag_service
        self.embedding_model=embedding_model
        self.client=client
        self.cache_collection=cache_collection



    ##====================
    ## FAQ Cache Warming
    ##====================

    def warm_faqs(self,faq_quetions:list[str])->None:
        """
         warm the semantic cache with predefined FAQ questions.

         This method:
            1. Enforces max cache size.
            2. If space available -> stores FAQ embeddings.
            3. Skips insertion if cache size exceeds limit.

        Args:
            faq_questions(list[str]):
                List of FAQ questions to embed and cache.
        
        Returns:
            None
        
        """

        deleted=enforce_max_cache_size(
            client=self.client,
            collection_name=self.cache_collection
        )

        ## If enforce_max_cache_size returns None,
        ## cache size is within limits.
        warm_faq_cache(
            client=self.client,
            embedding_model=self.embedding_model,
            collection_name=self.cache_collection,
            faq_list=faq_quetions,
            rag_service=self.rag_service
        )

        ##====================================
        ## Response Generation
        ##====================================
        
    def generate_response(self,query:str)-> Dict[str,Any]:
        """
            Generate response using Cache-Augmented Generation.

            steps:
                1. Check semantic cache.
                2. If hit -> return cached answer.
                3. If miss -> call RAG service.
                4. Return structured response wtih sourced flag.
            
            Args:
                query (str):
                    User input question.
            
            Return:
                Dict[str,Any]:
                    {
                     "answer":str,
                     "source":"cache" | "rag
                    
                    }
         """
        start_time=time.time() 

        cached_response: Optional[str] = check_cache(
            client=self.client,
            embedding_model=self.embedding_model,
            collection_name=self.cache_collection,
            query=query
        )

        elapsed=time.time()-start_time

        if cached_response["hit"]:
            print("✅ Cache Hit")
            return{
                "answer":cached_response["answer"],
                "source":"cache",
                "score": cached_response["score"],
                "generation_time":elapsed,
            }
        
        print("❌ Cache Miss → Generating answer using RAG...")
        
        ## Cache miss --> Run RAG
        rag_response=self.rag_service.generate_response(query)

        return{
            "answer":rag_response,
            "source":"rag"
        }



        

