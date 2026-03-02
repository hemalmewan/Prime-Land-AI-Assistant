"""
   Store Embedding Result + Qdrant Ingestion pipeline

   This module:
   1. Converts chunk dictionaries to LangChain Documents
   2. Generates OpenAI embeddings (3072-dim)
   3. Stores vectors in Qdarnt Cloud
   4. Uses batched upserts for scalability

   Assumptions:
   - Collection already created with size=3072
   - Distance metric=COSINE
   - Using text-embedding-3-large

"""

##===================================
## Import Required Libraries
##===================================
import uuid
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
from qdrant_client.models import PointStruct


##===============================================
## Convert combine chunks to LangChain Documents
##===============================================

def build_documents(total_chunks:List[dict])->List[Document]:
    """
      Convert raw chunk dictionaries into LangChain Document objects.

      Args:
        total_chunk (List[Dict]) : List of chunk dictionaries.

      Returns:
        List[Document]: Cleaned Document objects ready for embedding.
    """

    documents=[] ## to store all the chunks with the metadata
    ## loop through each chunk
    for chunk in total_chunks:
        doc=Document(
            page_content=chunk.get("text",""),
            
            metadata={
                "url":chunk.get("url"),
                "title":chunk.get("title"),
                "chunking_strategy":chunk.get("chunking_strategy"),
                "chunk_index":chunk.get("chunk_idx"),
                "property_id":chunk.get("property_id"),
                "price":chunk.get("price"),
                "bedrooms":chunk.get("bedrooms"),
                "bathrooms":chunk.get("bathrooms"),
                "aminities":chunk.get("amenities"),
                "agent":chunk.get("agent"),
            }
        )
        documents.append(doc)
    
    return documents

##======================================
## Embed and Store in Qdrant
##======================================

def ingest_documents_to_qdrant(
        documents:List[Document],
        client,
        embedding_model,
        collection_name:str,
        batch_size:int
):
    
    """
      Embed documents using OpenAI embedding model and store them in Qdrant.

      Args:
        documents (List[Document]): Documents to embed.
        client (QdrantClient): Initialized Qdrant client.
        embedder (OpenAIEmbeddings): Embedding model instance.
        collection_name (str): Target Qdrant collection.
        batch_size (int): Batch size for embedding and upsert.

    Returns:
        None
    """

    print("="*60)
    print("Starting embedding + ingestion pipeline......")
    print(f"Total documents:{len(documents)}")
    print("="*60)

    for i in tqdm(range(0,len(documents),batch_size)):
        batch_docs=documents[i:i+batch_size]

        ## Filter out documents that have empty or whitespace-only
        valid_docs=[doc for doc in batch_docs if doc.page_content and doc.page_content.strip()]

        if not valid_docs:
            print(f"Skipping batch {i//batch_size}:No valid text content found.")
            continue

        ## Extract text from the valid documents only
        clean_text=[doc.page_content.strip() for doc in valid_docs]

        try:
            embeddings=embedding_model.embed_documents(clean_text)                

        except Exception as e:
            print(f"Error during embedding at batch {i}: {e}")
            continue

        ## prepare Qdrant points
        points=[]

        for doc,vector in zip(valid_docs,embeddings):
            point=PointStruct(
                id=str(uuid.uuid4()), ## Unique ID per chunk
                vector=vector,
                payload={
                    "text":doc.page_content,
                    "metadata":doc.metadata
                }
            )
            points.append(point)
        
        ## upsert into qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print("Ingestion completed successfully ✅")