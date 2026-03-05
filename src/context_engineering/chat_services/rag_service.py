"""
RAG Service Module
This module implements the RAGService class, which provides functionalities for performing Retrieval-Augmented Generation (RAG) tasks. 

"""

## import necessary libraries
import time
from typing import List, Dict, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough
)

## user define prompt template for RAG
from src.context_engineering.prompts.rag_template import PRIME_LAND_RAG_TEMPLATE
from src.context_engineering.utils import format_docs,calculate_confidence,precision_at_5_keyword
from src.context_engineering.config import TOP_K


## define RAG chain function
def rag_chain(
        retriever,
        llm,
        k:int =TOP_K,
        prompt_template:str =PRIME_LAND_RAG_TEMPLATE

)-> Runnable:
    """
     Construct a Retrieval-Augmented Generation (RAG) chain using
     LangChain Expression Language (LECL) with a Qdrant based retriever.

     This function builds a modern runnable pipeline that performs:

         Query
           - Vector Retrieval (Qdrant)
           - Document Formatting (format_docs)
           - Prompt Injection
           - LLM Generation
           - Output Parsing

    The chain enforces grounded generation by injecting structured
    real estate property context into the provided prompt template.

    Unlike legacy LangChain chains (e.g., RetrievalQA), this implementation
    uses LECL primitives such as RunnableParallel and the '|' operator
    for composable,production-grade orchestration.

    Args:
        retriever:
            A LangChain-compatible retriever instance (e.g., QdrantRetriever)
        
        llm:
            A LangChain LLM instance (e.g., ChatOpenAI) responsible for generating
            grounded responses.
        
        k (int,optional):
            Number of documents to retrieve per query.
            Defaults to 5.
        
        template (str,optional):
            Prompt template string used for grounded generation.
            Must contain '{context}' and '{question}' variables.
    
    Returns:
        Runnable:
            A LangChain Runnable that takes a user query as input and returns
            a grounded answer based on retrieved documents.
  
    """

    ## filter retrieved documents to top k
    if k!=TOP_K:
        retriever.search_kwargs["k"]=k
    
    ## create the RAG prompt template
    rag_template=ChatPromptTemplate.from_template(PRIME_LAND_RAG_TEMPLATE)

    ## define the RAG chain using LECL
    rag_pipeline=(
        RunnableParallel(
            {
            "context":retriever | format_docs, ## retrieve and format documents in parallel
            "question":RunnablePassthrough() ## pass the user query through unchanged
            }
        )
        | rag_template
        | llm
        | StrOutputParser()
    )

    return rag_pipeline


class QdrantRAGService:
    """
     High-level RAG servicee abstraction using Qdrant Cloud as 
     the underlying vector store.

     This class encapsulates:

         - Vector retrieval
         - LCEL-based RAG chain execution
         - Evidence tracking
         - URL extraction
         - Response timing metrics
    
    It provides a clean interface for generating grounded answers
    in a real estate domain (Prime Lands) while preserving evidence
    trasparency and evaluation metrics.

    Attributes: 
        retriever:
            LangChain retriever backed by Qdrant Cloud.
        
        llm:
            LangChain LLM instance used for response generation.
        
        k(int):
            Number of documents retrieved per query.

        chain:
            Composed LCEL RAG chain built via 'rag_chain'.
    
    Example:
        >>> service = QdrantRAGService(retriever, llm, k=5)
        >>> result = service.generate_response("What is the price of GREATE 10 - GALLE?")
        >>> print(result["answer"])
        >>> print(result["evidence_urls"])
    
    """

    def __init__(
            self,
            retriever,
            llm,
            k:int=TOP_K):
        
        """
          Initialize the Qdrant RAG service.

          Args:
            retriever:
                Qdrant-backed retriever instance.
            
            llm:
                LangChain LLM used for grounded answer generation.
            
            k (int,optional):
                Number of documents to retrieve per query.
                Defaults to 5.
        """

        self.retriever=retriever
        self.llm=llm
        self.k=k
        self.chain=rag_chain(retriever,self.llm,k=self.k)
    
    def generate_response(
            self,
            query:str,
            verbose:bool=False)->Dict[str,Any]:
        """
        Generate a grounded answer for a user query using RAG.

        This method performs:
            1. Document retrieval from Qdrant
            2. Context formatting
            3. LLM-based grounded generation
            4. Evidence extraction
            5. Latency measurement

        Args:
            query (str):
                User question related to Prime Lands properties.
            
            If verbose=True
                print step by step execution logs.

        Returns:
            Dict[str, Any]:
                Dictionary containing:

                - answer (str):
                    The generated grounded response.

                - evidence (List[Document]):
                    Retrieved Document objects used as context.

                - evidence_urls (List[str]):
                    Unique source URLs extracted from metadata.

                - generation_time (float):
                    Total execution time in seconds.

                - num_docs (int):
                    Number of retrieved documents.
        
        """

        start=time.time()

        if verbose:
            print("🔎 Step 1: Retrieving relevant property documents from Qdrant...")
            print(f"➤ User Query: {query}")

        evidence=self.retriever.invoke(query)

        if verbose:
            print(f"📚 Retrieved {len(evidence)} documents.")
            print("🧮 Step 2: Calculating retrieval confidence score...")

        ##calculate confidence score for retrieve documents
        rag_confidence_score=calculate_confidence(docs=evidence,query=query)
        ## calculate the precision@5 for retrived documents
        precision=precision_at_5_keyword(retrieved_docs=evidence,query=query)

        if verbose:
             print(f"✅ Confidence Score: {rag_confidence_score:.4f}")
             print(f"✅ Pricision@5: {precision:.4f}")
             print("🧠 Step 3: Generating grounded response using LLM...")
        
        answer=self.chain.invoke(query)

        if verbose:
            print("✍️ Step 4: Extracting evidence URLs...")

        elapsed=time.time()-start

        evidence_urls=list({
            doc.metadata.get("url") 
            for doc in evidence
        })

        if verbose:
             print("📊 Step 5: Finalizing response...")
             print(f"⏱️ Total Generation Time: {elapsed:.2f} seconds")
             print("🎉 RAG Response Ready!\n")


        return{
            "answer":answer,
            "confidence_score":rag_confidence_score,
            "Precision@5":precision,
            "evidence":evidence,
            "evidence_urls":evidence_urls,
            "generation_time":elapsed,
            "num_docs":len(evidence)
        }