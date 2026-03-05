
##=================================
## Import Required Libraries
##=================================
from typing import List
from langchain_core.documents import Document
from src.context_engineering.config import TOP_K


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved LangChain Document objects into a structured
    context string optimized for Real Estate RAG prompting.

    This function converts structured metadata + page content into a
    clean, LLM-friendly format that improves:

    - Grounded generation
    - Citation accuracy
    - Property-specific reasoning
    - Hallucination reduction

    Important real estate attributes such as price, bedrooms,
    bathrooms, and amenities are explicitly injected to help the LLM
    reason over structured property data.

    Internal metadata fields (e.g., chunk_index, chunking_strategy)
    are intentionally excluded to avoid prompt noise.

    Args:
        docs (List[Document]):
            A list of LangChain Document objects retrieved from
            the vector store (e.g., Qdrant Cloud).

    Returns:
        str:
            A single formatted context string containing structured
            property information separated by clear delimiters.
            This string is injected into the RAG prompt template
            under the {context} variable.
    """

    formatted_docs = []

    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}

        # Extract relevant metadata
        url = metadata.get("url", "N/A")
        title = metadata.get("title", "Not Specified")
        property_id = metadata.get("property_id", "Not specified")
        price = metadata.get("price", "Not specified")
        bedrooms = metadata.get("bedrooms") or "Not specified"
        bathrooms = metadata.get("bathrooms") or "Not specified"
        amenities = metadata.get("amenities") or []

        # Clean description text
        description = doc.page_content.strip()

        # Correct: use parentheses, NOT {}
        block = (
            f"[Property {i+1}]\n"
            f"Property Name: {title}\n"
            f"Property ID: {property_id}\n"
            f"Price: {price}\n"
            f"Bedrooms: {bedrooms}\n"
            f"Bathrooms: {bathrooms}\n"
            f"Amenities: {', '.join(amenities) if amenities else 'None'}\n"
            f"\nDescription:\n{description}\n"
            f"\nSource: {url}"
        )

        formatted_docs.append(block)

    return "\n\n===================\n\n".join(formatted_docs)


def calculate_confidence(docs: list, query: str) -> float:
    """
    Calculate confidence score for retrieved documents.
    
    Multi-factor heuristic:
    1. Keyword overlap (query ∩ docs)
    2. Content richness (avg doc length)
    3. Strategy diversity (multiple chunking strategies)
    
    Args:
        docs: List of retrieved documents
        query: User query string
    
    Returns:
        Confidence score 0.0 to 1.0
    """
    if not docs:
        return 0.0
    
    # Extract query keywords
    query_words = set(query.lower().split())
    
    # Factor 1: Keyword overlap
    overlaps = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
        overlaps.append(overlap)
    keyword_score = sum(overlaps) / len(overlaps)
    
    # Factor 2: Content richness
    avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
    length_score = min(avg_length / 500, 1.0)
    
    # Factor 3: Strategy diversity
    strategies = set([doc.metadata.get('strategy', 'unknown') for doc in docs])
    diversity_score = len(strategies) / 3.0  # We have 3 strategies max
    
    # Weighted average
    confidence = (
        0.5 * keyword_score +
        0.3 * length_score +
        0.2 * diversity_score
    )
    
    return confidence


def precision_at_5_keyword(retrieved_docs: list, query: str) -> float:
    """
    Approximate Precision@5 using keyword overlap heuristic.
    """

    if not retrieved_docs:
        return 0.0

    query_words = set(query.lower().split())
    relevant_count = 0

    for doc in retrieved_docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words)
        
        if overlap > 2:  # threshold for relevance
            relevant_count += 1

    return relevant_count / TOP_K