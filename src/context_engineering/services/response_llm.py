"""
LLM provider with OpenRouter support for multi-provider access.

OpenRouter (https://openrouter.ai) provides unified API access to:
- OpenAI (GPT-4, GPT-4o, o1, o3)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini 2.0)
- Meta (Llama 3)
- DeepSeek, Mistral, and many more

Set OPENROUTER_API_KEY in .env to use any supported model.
"""

## Import necessary libraries
import os
from aiofiles import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI

## Import user defined configuration and utility functions
from src.context_engineering.config import (
    LLM_MODEL,
    OPENROUTER_URL,
    PROVIDER,
    TEMPERATURE,
    MAX_TOKENS,
    STREANING,
    get_llm_model,
    get_api_keys,

)

def get_chat_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = None,
    path: Optional[str] = None,
    **kwargs: Any
) -> ChatOpenAI:
    """
    Factory function to create a chat LLM instance.
    
    Supports multiple providers via OpenRouter unified API,
    or direct provider access if configured.
    
    Args:
        model: Model name (e.g., "openai/gpt-4o-mini"). If None, uses config.
        provider: Override provider ("openrouter", "openai", etc.)
        tier: Model tier ("general", "strong", "reason")
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens to generate
        streaming: Enable streaming responses
        **kwargs: Additional provider-specific parameters
    
    Returns:
        ChatOpenAI: An LLM instance ready for chat completions
    
    Examples:
        # Use default from config
        >>> llm = get_chat_llm()
        
        # Use specific model
        >>> llm = get_chat_llm(model="anthropic/claude-3-5-sonnet")
        
        # Use different tier
        >>> llm = get_chat_llm(tier="strong")
        
        # Use specific provider directly
        >>> llm = get_chat_llm(provider="openai")
    """
    # Determine provider
    use_provider = provider or PROVIDER
    
    # Determine model
    if model:
        use_model = model
    elif tier:
        use_model = get_llm_model(provider=use_provider, tier=tier)
    else:
        use_model = LLM_MODEL
    
    # Get API key
    api_key = get_api_keys(provider=use_provider,env_path=path)
    if not api_key:
        # Fallback to OPENAI_API_KEY for backward compatibility
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Set defaults
    use_temperature = temperature if temperature is not None else TEMPERATURE
    use_max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS
    use_streaming = streaming if streaming is not None else STREANING
    
    # Configure based on provider
    if use_provider == "openrouter":
        # OpenRouter configuration
        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/zuu-crew/context-engineering",
                "X-Title": "Context Engineering RAG"
            },
            **kwargs
        )
    else:
        # Direct provider access (OpenAI compatible)
        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            **kwargs
        )