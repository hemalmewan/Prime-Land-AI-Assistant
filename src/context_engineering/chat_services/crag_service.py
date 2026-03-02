"""
crag_service.py
===========================================================

Corrective Retrieval-Augmented Generation (CRAG) Service.

This module implements a self-correcting Retrieval-Augmented
Generation (RAG) pipeline designed to improve answer quality
by dynamically evaluating retrieval confidence and applying
corrective retrieval strategies when necessary.

Core Features:
--------------
- Initial semantic retrieval with configurable top-k
- Confidence scoring of retrieved documents
- Automatic corrective retrieval when confidence is low
- Optional query expansion for improved grounding
- Structured response with metrics and diagnostics

Workflow:
---------
1. Perform initial retrieval (k = initial_k).
2. Compute retrieval confidence score.
3. If confidence < threshold:
      - Apply corrective retrieval (k = expanded_k)
      - Optionally expand or reformulate query
      - Recalculate confidence
4. Generate final answer using best evidence set.
5. Return answer with confidence metrics and metadata.

Benefits:
---------
- Improves factual grounding
- Reduces hallucinations
- Automatically adapts to weak retrieval
- Provides transparent confidence diagnostics
- Production-ready modular design

Intended Usage:
---------------
This service is designed to be used after a Cache-Augmented
Generation (CAG) layer. If cache miss occurs, CRAG handles
self-correcting retrieval before answer generation.

Dependencies:
-------------
- LangChain retriever
- LLM interface
- Prompt templates
- Custom confidence scoring utility
"""

##=========================================
## Import Required Libraries
##=========================================
from typing import Dict,Any,List
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from src.context_engineering.config import(
    CONFIDENCE_THRESHOLD,
    EXPANDED_K,
    TOP_K
)
from src.context_engineering.utils import format_docs,calculate_confidence
from src.context_engineering.prompts.rag_template import PRIME_LAND_RAG_TEMPLATE

class QdrantCRAGService:
    """
    Corrective Retrievel-Augmented Generation Service.

    CRAG enhances traditional RAG by introducing a retrieval
    confidence evaluation step and automatically applying
    corrective retrieval strategies when inital results
    are insufficient.

    This class is responsible for:
        - Performing intital semantic retrieval
        - Evaluating retrieval confidence
        - Triggering expanded or corrective retrieval
        - Generating grounded answers using selected evidence
        - Returning detailed performance metrics
    
    Attributes:
    ------------------
    retriever: VectorStoreRetriever
        vector database retriever used for semantic search.
    
    llm: Any
        Language model instance used for response generation
        and optional query expansion.
    
    initial_k : int
        Number of documents retrieved during initial search.

    expanded_k : int
        Number of documents retrieved during corrective search.

    prompt : ChatPromptTemplate
        Prompt template used for final answer generation.

    Design Philosophy:
    ------------------
    Instead of blindly trusting the first retrieval step,
    CRAG measures retrieval strength and dynamically adjusts
    search depth to improve grounding quality.

    Example:
    --------
    >>> crag = CRAGService(retriever, llm)
    >>> result = crag.generate("What is Prime Lands?")
    >>> print(result["answer"])
    >>> print(result["confidence_final"])

    """

    def __init__(self,
                 retriever: VectorStore,
                 llm:Any,
                 expanded_k:int=EXPANDED_K,
                 initial_k:int=TOP_K

                 ):
         """
        Initialize CRAG service.
        
        Args:
            retriever: Vector store retriever
            llm: LangChain LLM instance
            initial_k: Number of docs for initial retrieval
            expanded_k: Number of docs for corrective retrieval
        """
         
         self.retriever= retriever
         self.llm=llm
         self.expanded_k=expanded_k
         self.initial_k=initial_k

         self.prompt = ChatPromptTemplate.from_template(PRIME_LAND_RAG_TEMPLATE)

    def generate_crag_response(self,
                               query:str,
                               confidence_threshold:float=CONFIDENCE_THRESHOLD,
                               verbose: bool=True
                               )->Dict[str,Any]:
        """
        Generate an answer using Corrective RAG.

        Parameters
        ----------
        query : str
            User question or input query.

        confidence_threshold : float
            Minimum confidence score required to accept
            initial retrieval results (range: 0.0 – 1.0).

        verbose : bool, optional
            If True, prints diagnostic logs during execution.

        Returns
        -------
        Dict[str, Any]
            Structured response containing:

            - answer : str
                Final generated response.
            - confidence_initial : float
                Confidence score of initial retrieval.
            - confidence_final : float
                Final confidence score used for generation.
            - correction_applied : bool
                Whether corrective retrieval was triggered.
            - docs_used : int
                Number of documents used in final generation.
            - generation_time : float
                Time taken for answer generation (seconds).
            - evidence_urls : List[str]
                List of unique source URLs.
            - evidence : List[Document]
                Retrieved document objects used for grounding. 
         """
        
        if verbose:
            print(f"🔍 Query: {query}")
            print(f"🎯 Confidence threshold: {confidence_threshold}\n")
        
        # Step 1: Initial retrieval
        if verbose:
            print(f"1️⃣  Initial retrieval (k={self.initial_k})...")
        
        self.retriever.search_kwargs["k"] = self.initial_k
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)
        
        if verbose:
            print(f"   📊 Confidence: {confidence_initial:.2f}")
        
        # Step 2: Check if correction needed
        if confidence_initial >= confidence_threshold:
            if verbose:
                print(f"   ✅ Confidence sufficient - proceeding with initial retrieval")
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                print(f"   ⚠️  Low confidence - applying corrective retrieval...\n")
            
            # Step 3: Corrective retrieval
            if verbose:
                print(f"2️⃣  Corrective retrieval (k={self.expanded_k}, expanded)...")
            
            # Expand k for more diverse results
            self.retriever.search_kwargs["k"] = self.expanded_k
            docs_corrected = self.retriever.invoke(query)
            confidence_final = calculate_confidence(docs_corrected, query)
            
            if verbose:
                print(f"   📊 Corrected confidence: {confidence_final:.2f}")
                improvement = (confidence_final - confidence_initial) * 100
                print(f"   📈 Confidence improved by {improvement:.1f}%")
            
            final_docs = docs_corrected
            correction_applied = True
        
        # Step 4: Generate answer
        if verbose:
            print(f"\n3️⃣  Generating answer...")
        
        start = time.time()
        
        # Format docs and generate
        context = format_docs(final_docs)
        prompt_input = {"context": context, "question": query}
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)
        
        elapsed = time.time() - start
        
        # Extract evidence URLs
        evidence_urls = list(set([doc.metadata['url'] for doc in final_docs]))
        
        return {
            'answer': answer,
            'confidence_initial': confidence_initial,
            'confidence_final': confidence_final,
            'correction_applied': correction_applied,
            'docs_used': len(final_docs),
            'generation_time': elapsed,
            'evidence_urls': evidence_urls,
            'evidence': final_docs
        }
