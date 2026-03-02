"""
   Embedding Factory Module
   ==============================

   This module provides factory utilities for creating embedding model
   Instances used in vectorization pipelines.

   It supports:
   - Multiple providers (OpenAI,OpenRouter)
   - Tier-based model selection
   - Config-driven defaults
   - Batch configuration
   - Progress bar control

   The returned embedding model is compatible with LangChain and
   vector databases such as Qdrant.

   Design Goals:
    - Centralized embedding configuration
    - Provider-agnostic interface
    - Production-ready initialization
    - Clean separation of concerns
"""
##=============================================
## Import Required Libries
##=============================================
from typing import Dict,List,Any,Optional
from langchain_openai import OpenAIEmbeddings
from src.context_engineering.config import (
    EMBEDDING_MODEL,
    PROVIDER,
    OPENROUTER_URL,
    BATCH_SIZE,
    SHOW_PROGRESS,
    get_api_keys,
    get_embedding_model
    
)

##==============================
## Embedding Function
##==============================

def _get_embedding_model(
        model:Optional[str]=None,
        provider:Optional[str]=None,
        tier:str="default",
        batch_size:Optional[int]=None,
        show_progress:Optional[bool]=None,
        env_path:Optional[str]=None,
        **kwargs:Any
)->OpenAIEmbeddings:
    """
     Factory function to create and configure an embedding model instance.

     This function supports multiple providers(openAI and OpenRouter)
     and dynamically resolves the embedding model based on configuration.

     Parametrs
      - model: Optional[str]
            Explicit model name overide (e.g., "text-embedding-3-large").
            If None,model is resolved from config based on provider and tier.

      - provider: Optional[str]
            Provider override.Supported:
                -"open-ai"
                -"openrouter"

      - tier: str,default="default"
            Embedding tier configuration:
                -"default"
                -"small"

      - batch_size: Optional[int]
            Number of texts processed in parallel during embedding.

      -show_progress: Optional[bool]
            Whether to display progress bar for large embedding batches.
      
      -env_path: Optional[str]
            enviromental variable path .env file path
    
      **kwargs: Any
            Additional keyword arguments
    
    Returns
      -OpenAIEmbeddings
        A fully configured embedding model instance ready for vectorization.

    Example
      >>> embedder=get_embedding_model()
      >>> embedder=get_embedding_model(tier="small")
      >>> embedder=get_embedding_model(model="text-embedding-3-large")
      
    
    """
    ## determine provider
    use_provider=provider or PROVIDER

    ## determine model
    if model:
        use_model=model
    else:
        use_model=get_embedding_model(provider=use_provider,tier=tier)
        

    ## remove provider prefix if exists(e.g., "openai/text-embedding-3-large")
    if "/" in use_model:
        use_model=use_model.split("/")[-1]
    
    ## retrive API key
    api_key=get_api_keys(provider=use_provider,env_path=env_path)

    if not api_key:
        raise ValueError(f"API key not found for provider:{use_provider}")
    
    ## Resolve batch configuration
    use_batch_size=batch_size if batch_size is not None else BATCH_SIZE
    use_show_progress=show_progress if show_progress is not None else SHOW_PROGRESS

    ## configure based on provider
    if use_provider=="openrouter":
        return OpenAIEmbeddings(
            model='openai/'+use_model,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_URL,
            chunk_size=use_batch_size,
            show_progress_bar=SHOW_PROGRESS,
            **kwargs

        )
    elif use_provider=="open-ai":
        return OpenAIEmbeddings(
            model=use_model,
            openai_api_key=api_key,
            chunk_size=use_batch_size,
            show_progress_bar=use_show_progress,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported embedding provider:{use_provider}")

