##====================================================
## Application configuration.Load all the YAML files.
##======================================================

##=============================
## import requires libraries
##=============================
import yaml
from pathlib import Path
from typing import Dict,Any,Optional,Union
import sys
import os
from dotenv import load_dotenv

##=====================================
## set the current working directory
##=====================================
main_root=Path.cwd().parent.parent
yaml_dir=main_root/"config"

##==========================================================================
## load the YAML(config/config.yaml and config/llm_models.yaml) files
##==========================================================================
def load_yaml(file_name:str)-> Dict[str,Any]:
    """
      Load both the yaml files
       1.Config.yaml
       2.llm_models.yaml
    """
    config_dir=yaml_dir/file_name
    if not config_dir.exists():
        print(f"Directory not exists:{config_dir}")
        return {}
    
    with open(config_dir,'r') as f:
        return yaml.safe_load(f)

##================================================================
## get the yaml configuration setup indices through this function
##================================================================
def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default

##==================================================
## Loads config files "call the load yaml function"
##==================================================
config=load_yaml("config.yaml")
llm_models=load_yaml("llm_models.yaml")
    

##=====================================
## Provider Configuration
##=====================================
PROVIDER=_get_nested(config,"provider"," default",default="openrouter")
MODEL_TIER=_get_nested(config,"provider","tier",default="general")
OPENROUTER_URL=_get_nested(config,"provider","openrouter_url",default="https://openrouter.ai/api/v1")


##===================================
## Model Cofiguration
##===================================

def get_llm_model(provider:Optional[str]=None,tier:Optional[str]=None)->str:
    provider=PROVIDER
    tier=MODEL_TIER
    return _get_nested(llm_models,provider,"chat",tier,default="openai/gpt-4o-mini")

def get_embedding_model(provider:Optional[str]=None,tier:str="default")->str:
    provider=PROVIDER
    return _get_nested(llm_models,provider,"embedding",tier,default="openai/text-embedding-3-large")


LLM_MODEL=get_llm_model()
EMBEDDING_MODEL=get_embedding_model()

##===================================
## LLM Configuration
##===================================
TEMPERATURE=_get_nested(config,"llm","temperature",default=0.0)
MAX_TOKENS=_get_nested(config,"llm","max_tokens",default=2000)
STREANING=_get_nested(config,"llm","streaming",default=False)

##===================================
##  Embedding Model Configuration
##===================================
TIER=_get_nested(config,"embedding","tier",default="default")
BATCH_SIZE=_get_nested(config,"embedding","batch_size",default=20)
SHOW_PROGRESS=_get_nested(config,"embedding","show_progress",default=False)

##==================================
## Crawling Configuration
##==================================
BASE_URL=_get_nested(config,"crawling","base_url",default=" https://www.primelands.lk")
MAX_DEPTH=_get_nested(config,"crawling","max_depth",default=3)
MAX_PAGES=_get_nested(config,"crawling","max_pages",default=500)
TIMEOUT=_get_nested(config,"crawling","timeout",default=30000)
RATE_LIMIT_SECONDS=_get_nested(config,"crawling","rate_limit_seconds",default=2.0)

##==================================
## Data Paths Configuration
##==================================
DATA_DIR=_get_nested(config,"data_dir",default="data")
CHUNKS_DIR=_get_nested(config,"chunks_dir",default="data/chunkings")
MARKDOWN_DIR=_get_nested(config,"markdown_dir",default="data/markdown")
CRAWL_OUT_DIR=_get_nested(config,"processed_dir",default="data/processed")
CACHE_DIR=_get_nested(config,"cache_dir",default="data/cache")

##==================================
## Chunking Startegy Configuration
##==================================
FIXED_CHUNK_SIZE=_get_nested(config,"chunking","fixed_chunk","chunk_size",default=800)
FIXED_CHUNK_OVERLAP=_get_nested(config,"chunking","fixed_chunk","chunk_overlap",default=100)
SEMENTIC_CHUNK_SIZE=_get_nested(config,"chunking","semantic_chunk","chunk_size",default=1000)
SEMENTIC_MIN_CHUNK=_get_nested(config,"chunking","sementic_chunk","min_chunk_size",default=200)
SLIDING_CHUNK_SIZE=_get_nested(config,"chunking","sliding_chunk","chunk_size",default=512)
SLIDING_CHUNK_OVERLAP=_get_nested(config,"chunking","sliding_chunk","overlap",default=256)
PARENT_CHUNK_SIZE=_get_nested(config,"chunking","parent_child_chunk","parent_chunk",default=1200)
CHILD_CHUNK_SIZE=_get_nested(config,"chunking","parent_child_chunk","child_chunk",default=250)
CHILD_OVERLAP=_get_nested(config,"chunking","parent_child_chunk","child_overlap",default=50)
LATE_CHUNK_BASE_SIZE=_get_nested(config,"chunking","late_chunk","base_size",default=1000)
LATE_CHUNK_SPLIT_SIZE=_get_nested(config,"chunking","late_chunk","split_size",default=300)
LATE_CONTEXT_WINDOW=_get_nested(config,"chunking","late_chunk","context_window",default=150)

##=================================
## Retrivel Configuration
##=================================
TOP_K=_get_nested(config,"retrieval","top_k",default=5)
SIMILARITY_THRESHOLD=_get_nested(config,"retrieval","similarity_threshold",default=0.7)

##================================
## CAG Configuration
##================================
CACHE_TTL=_get_nested(config,"cag","cache_ttl",default=86400)
MAX_CACHE_SIZE=_get_nested(config,"cag","max_cache_size",default=1000)
CACHE_SIMILARITY_THRESHOLD=_get_nested(config,"cag","similarity_threshold",default=0.9)
HISTORY_TTL_HOURS=_get_nested(config,"cag","history_ttl_hours",default=24)

##=================================
## CRAG Configuration
##=================================
CONFIDENCE_THRESHOLD=_get_nested(config,"crag","confidence_threshold",default=0.6)
EXPANDED_K=_get_nested(config,"crag","expanded_k",default=8)


##===============================
## Load FAQ yaml file
##===============================
def load_faq(file_name:str)->Dict[str,Any]:
    """
    Load the FAQ yaml file and return the content as a dictionary.
    Args:
        file_name (str): The name of the FAQ yaml file to load.

    Returns:
        Dict[str, Any]: A dictionary containing the FAQ data loaded from the yaml file.

    """
    faq_dir=yaml_dir/file_name
    if not faq_dir.exists():
        print(f"FAQ file does not exist: {faq_dir}")
        return {}
    
    with open(faq_dir,"r") as f:
        return yaml.safe_load(f)



##=================================
## API Keys Configurations
##=================================

def get_api_keys(
    provider: Optional[str],
    env_path: str
) -> Optional[Union[str, Dict[str, str]]]:
    """
    Retrieve secret API credentials for a given provider from a specified `.env` file.

    Parameters
    ----------
    provider : Optional[str]
        Name of the provider. Supported values:
            - "open-ai"
            - "openrouter"
            - "gemini"
            - "qdrant"

    env_path : str
        Path to the `.env` file containing API credentials.

    Returns
    Optional[Union[str, Dict[str, str]]]
        - Returns a string API key for:
            "open-ai", "openrouter", "gemini"
        - Returns a dictionary for:
            "qdrant" → {"url": str, "key": str}
        - Returns None if provider is None.

    Raises
    ValueError
        If:
            - Provider is unsupported
            - Required environment variables are missing
            - .env file cannot be loaded
    """

    if provider is None:
        return None

    # Load .env into environment
    loaded = load_dotenv(str(env_path))

    if not loaded:
        raise ValueError(f"Could not load environment file from path: {env_path}")

    keys_providers = {
        "open-ai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "qdrant": {
            "url": "QDRANT_URL",
            "key": "QDRANT_API_KEY"
        }
    }

    if provider not in keys_providers:
        raise ValueError(f"Unsupported provider: {provider}")

    provider_config = keys_providers[provider]

    # Single-key providers
    if isinstance(provider_config, str):
        secret = os.getenv(provider_config)
        if not secret:
            raise ValueError(f"Missing {provider_config} in {env_path}")
        return secret

    # Multi-key provider (qdrant)
    elif isinstance(provider_config, dict):
        result = {}
        for key_name, env_var in provider_config.items():
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Missing {env_var} in {env_path}")
            result[key_name] = value
        return result


def show_confiurations():
    """
    Docstring for show_confiurations
    Print all the configurations(non secrets)
    """

    print("="*60)
    print("Pirinting All the Configuation(Non-Secrets)")
    print("="*60)

    print("PROVIDER")
    print(f"LLM PROVIDER:{PROVIDER}")
    print(f"LLM MODEL TIER:{MODEL_TIER}")
    print(f"LLM CHAT MODEL:{LLM_MODEL}")
    print(f"EMBEDDING MODEL:{EMBEDDING_MODEL}")

    print("="*60)

    print("DIRECTRIES")
    print(f"DATA DIRECTORTY:{DATA_DIR}")
    print(f"VECTOR DB STORE DIRECTORY:{CHUNKS_DIR}")
    print(f"WEB CRAWLING OUTPUT DIRECTORY:{CRAWL_OUT_DIR}")
    print(f"MARKDWON DIRECTORY:{MARKDOWN_DIR}")


    print("="*60)


