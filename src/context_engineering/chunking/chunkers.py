"""
Docstring for src.context_engineering.crawler.chunkers

  Text Chunking Strategies Module

  This module implements multiple text chunking strategies for document ingestion
  within the Context Engineering pipeline.These strategies transform raw documents
  into strcutured chunks optimized for embedding,indexing,and retrieval.

  Implemented Strategies:
  1.Semantic/ Heading-Aware Chunking
    -Splits documents based on structural elements such as headings and sectons.
    -Preserves semantic coherence and document hierarchy.

  2.Fixed-Window Chunking
    -Splits text into uniform-sized chunks.
    -supports configrable overlap to maintain contextual continuity.

  3.Sliding-Window Chunking
    -Creates overlapping rolling windows across text.
    -Improves recall during retrieval by preservinb boundry context.
  
  4.Parent-Child (Two-Tier) Chunking
    -Generates small child chunks for embedding.
    -Retains larger parent chunks to privide expanded context at retrieval time.
  
  5.Query-Focused Late Chunking
    -Stores large base passages.
    -Applies fine-grained chunking dunamically during retrieval based on the query.

"""

##======================================
## Import Required Libraries
##======================================
from typing import Dict,Any,List,Tuple
import tiktoken
from langchain_text_splitters import(
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from src.context_engineering.config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    SEMENTIC_CHUNK_SIZE,
    SEMENTIC_MIN_CHUNK,
    SLIDING_CHUNK_SIZE,
    SLIDING_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    LATE_CHUNK_BASE_SIZE,
    LATE_CHUNK_SPLIT_SIZE,
    LATE_CONTEXT_WINDOW
)


##=================================
## Token Count for each Chunking
##=================================
def count_token(text:str,model:str)-> int:
    """
    Docstring for count_token
    
    :param text: Description
    :type text: str
    :param model: Description
    :type model: str
    :return: Description
    :rtype: int

    calculate the token count for each chunking strategies
    """
    try:
        encoding=tiktoken.encoding_for_model(model_name=model)
    except KeyError:
        encoding=tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
        



##=======================================
## Fixed Chunking Strategy
##=======================================
def fixed_chunk(documents:List[Dict[str,Any]])-> List[Dict[str,Any]]:
    """
    Docstring for fixed_chunk
    
    :param documents: Description
    :type documents: List[Dict[str, Any]]
    :return: Description
    :rtype: List[Dict[str, Any]]

    Split the each document into the fixed-chunk size

    Args:
      Each document is a list of a Dict containing title,url,headings and content.

    Return:
      List of chunk with Dict containing url,title,chunk text,chunking strategy,token count
    """

    chunks=[]  ##to store the each chunk
    chunk_idx=0 ## define the unique chunk index

    chunk_size_chars=FIXED_CHUNK_SIZE*4
    chunk_size_overlap=FIXED_CHUNK_OVERLAP*4

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_size_overlap,
        length_function=len,
        separators=["\n\n","\n"," ."," ",""]
    )

    ## loop through each document
    for doc in documents:
        url=doc["url"]
        content=doc["content"]
        title=doc["title"]

        ## extract structured meta data 
        property_id=doc.get("property_id")
        price=doc.get("price")
        bedrooms=doc.get("bedrooms")
        bathrooms=doc.get("bathrooms")
        amenities=doc.get("amenities",[])
        agent=doc.get("agent")

        ## split the content
        doc_split_chunk=splitter.split_text(content)

        ##loop through each document chunking
        for chunk in doc_split_chunk:
            if chunk.strip():
                token_count=count_token(chunk,"gpt-4o")
            
                chunks.append({
                  "url":url,
                  "title":title,
                  "property_id":property_id,
                  "price":price,
                  "bedrooms":bedrooms,
                  "bathrooms":bathrooms,
                  "amenities":amenities,
                  "agent":agent,
                  "text":chunk.strip(),
                  "chunking_strategy":"Fixed Strategy",
                  "token_count":token_count,
                  "chunk_idx":chunk_idx
                })
            chunk_idx+=1
    
    return chunks
            

##=======================================
## Semantic Chunking Strategy
##=======================================
def sementic_chunk(documents:List[Dict[str,Any]])->List[Dict[str,Any]]:
    """
    Docstring for sementic_chunk
    
    :param document: Description
    :type document: List[Dict[str, Any]]
    :return: Description
    :rtype: List[Dict[str, Any]]

     Heading aware chunking strategy.
     Split the document by html headings

     Args:
      List of the documents with Dict containing url,content and headings
     
     Return:
      List of the metadata as Dict 
    """

    chunks=[] ##to store each chunk
    chunk_idx=0 ## Assign unique chunk index for each chunk

    ## define the heading hierarchy
    heading_splitter=[
        ("#","h1"),
        ("##","h2"),
        ("###","h3")
    ]

    splitter=MarkdownHeaderTextSplitter(
        headers_to_split_on=heading_splitter,
        strip_headers=False
    )

    ## loop through each document
    for doc in documents:
        url=doc["url"]
        content=doc["content"]
        title=doc["title"]

        ## extract structured meta data 
        property_id=doc.get("property_id")
        price=doc.get("price")
        bedrooms=doc.get("bedrooms")
        bathrooms=doc.get("bathrooms")
        amenities=doc.get("amenities",[])
        agent=doc.get("agent")

        ## split the document by headings
        try:
           ## split the document by headings
          sections=splitter.split_text(content)

          ##loop through each section
          for section in sections:
              text=section.page_content.strip()

              if not text or len(text)< SEMENTIC_MIN_CHUNK:
                  continue
              ## if text is too large futher split
              if count_token(text=text,model="gpt-4o")> SEMENTIC_CHUNK_SIZE:
                  char_size=SEMENTIC_CHUNK_SIZE*4

                  splitter=RecursiveCharacterTextSplitter(
                      char_size=char_size,
                      chunk_overlap=100,
                      length_function=len
                  )

                  ## get the further small chunks
                  sub_chunks=splitter.split_text(text)

                  ## loop through each chunk
                  for sub_chunk in sub_chunks:
                      if sub_chunk.strip():
                          chunks.append({
                            "url":url,
                            "title":title,
                            "property_id":property_id,
                            "price":price,
                            "bedrooms":bedrooms,
                            "bathrooms":bathrooms,
                            "amenities":amenities,
                            "agent":agent,
                            "text":sub_chunk.strip(),
                            "chunking_strategy":"Sementic Strategy",
                            "token_count":count_token(text=sub_chunk,model="gpt-4o"),
                            "chunk_idx":chunk_idx,
                            "heading": section.metadata.get('h1', '') or section.metadata.get('h2', '')

                          })
                      chunk_idx+=1
              else:
                  chunks.append({
                    "url":url,
                    "title":title,
                    "property_id":property_id,
                    "price":price,
                    "bedrooms":bedrooms,
                    "bathrooms":bathrooms,
                    "amenities":amenities,
                    "agent":agent,
                    "text":text.strip(),
                    "chunking_strategy":"Sementic Strategy",
                    "token_count":count_token(text=text,model="gpt-4o"),
                    "chunk_idx":chunk_idx,
                    "heading": section.metadata.get('h1', '') or section.metadata.get('h2', '')
                  })
                  chunk_idx+=1
        except Exception as e:
            if content.strip():
                chunks.append({
                    "url":url,
                    "title":title,
                    "property_id":property_id,
                    "price":price,
                    "bedrooms":bedrooms,
                    "bathrooms":bathrooms,
                    "amenities":amenities,
                    "agent":agent,
                    "content":content.strip(),
                    "chunking_strategy":"Sementic Strategy",
                    "token_count":count_token(text=content,model="gpt-4o"),
                    "chunk_ids":chunk_idx,
                    "heading":""
                })
                chunk_idx+=1
    return chunks

##==================================
## Sliding Chunking Strategy
##==================================
                      
def sliding_chunk(documents:List[Dict[str,Any]])-> List[Dict[str,Any]]:
    """
    Docstring for sliding_chunk
    
    :param List: Description

     split the document with the overlap.

    Args:
      List of the documents with Dict containing url,content and headings
     
    Return:
      List of the metadata as Dict 
    """

    chunks=[] ##to store each chunks
    chunk_idx=0 ## assign the unique index value for each chunk

    ## parameter configuration
    window_char_size= SLIDING_CHUNK_SIZE*4
    stride_char_size=SLIDING_CHUNK_OVERLAP*4

    ## loop through each document
    for doc in documents:
        url=doc["url"]
        content=doc["content"]
        title=doc["title"]

         ## extract structured meta data 
        property_id=doc.get("property_id")
        price=doc.get("price")
        bedrooms=doc.get("bedrooms")
        bathrooms=doc.get("bathrooms")
        amenities=doc.get("amenities",[])
        agent=doc.get("agent")

        ## sliding window parameter configuration
        pos=0
        window_idx=0
        content_len=len(content)

        while pos < content_len:
            end=min(pos+window_char_size,content_len)
            window_text=content[pos:end]

            if window_text.strip():
                chunks.append({
                    "url":url,
                    "title":title,
                    "text":window_text.strip(),
                    "property_id":property_id,
                    "price":price,
                    "bedrooms":bedrooms,
                    "bathrooms":bathrooms,
                    "amenities":amenities,
                    "agent":agent,
                    "chunking_strategy":"Sliding Strategy",
                    "token_count":count_token(text=window_text,model="gpt-4o"),
                    "chunk_idx":chunk_idx,
                    "window_idx":window_idx
                })
                chunk_idx+=1
                window_idx+=1
            pos+=stride_char_size

            if pos>=content_len:
                break
    return chunks

##===============================
## Parent Child Chunking Srategy
##===============================
def parent_child_chunk(documents:List[Dict[str,Any]])-> Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:
    """
    Docstring for parent_child_chunk
    
    :param documents: Description
    :type documents: List[Dict[str, Any]]
    :return: Description
    :rtype: List[Dict[str, Any]]

    This method controll the over chunking and less chunking situation.

    Args:
      List of the documents with Dict containing url,content and headings
     
    Return:
      List of the metadata as Dict 
    """

    parent_idx=0 ## assign unique parent index
    child_idx=0 ## assign unique child index
    parent_chunks=[] ## to store the parent chunks
    child_chunks=[] ##to store the child chunks

    ## parameter cofiguration
    parent_char_size=PARENT_CHUNK_SIZE*4
    child_char_size=CHILD_CHUNK_SIZE*4
    child_overlap_size=CHILD_OVERLAP*4

    ## Parent splitter
    parent_splitter=RecursiveCharacterTextSplitter(
        chunk_size=parent_char_size,
        chunk_overlap=200,
        length_function=len

    )

    ## Child splitter
    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=child_char_size,
        chunk_overlap=child_overlap_size,
        length_function=len
    )

    ## loop through each document
    for doc in documents:
        url=doc["url"]
        title=doc['title']
        content=doc["content"]

         ## extract structured meta data 
        property_id=doc.get("property_id")
        price=doc.get("price")
        bedrooms=doc.get("bedrooms")
        bathrooms=doc.get("bathrooms")
        amenities=doc.get("amenities",[])
        agent=doc.get("agent")

        ## Create the parent chunk
        _parent_chunks=parent_splitter.split_text(content)

        ##loop through each parent chunks
        for parent_chunk in _parent_chunks:
            if not parent_chunk.strip():
                continue
            
            parent_id=f"{url}::Parent::{parent_idx}"

            parent_chunks.append({
                "parent_id":parent_id,
                "url":url,
                "title":title,
                "property_id":property_id,
                "price":price,
                "bedrooms":bedrooms,
                "bathrooms":bathrooms,
                "amenities":amenities,
                "agent":agent,
                "text":parent_chunk.strip(),
                "chunking_strategy":"Parent Strategy",
                "token_count":count_token(text=parent_chunk,model="gpt-4o"),
                "parent_idx":parent_idx
            })

            ## create the child chunk
            _child_chunks=child_splitter.split_text(parent_chunk)

            for child_chunk in _child_chunks:
                if child_chunk.strip():
                    child_chunks.append({
                        "child_id":f"{parent_id}::Child::{child_idx}",
                        "parent_idx":parent_idx,
                        "url":url,
                        "title":title,
                        "property_id":property_id,
                        "price":price,
                        "bedrooms":bedrooms,
                        "bathrooms":bathrooms,
                        "amenities":amenities,
                        "agent":agent,
                        "text":child_chunk.strip(),
                        "chunking_strategy":"Child Strategy",
                        "token_count":count_token(text=child_chunk.strip(),model="gpt-4o"),
                        "child_idx":child_idx

                    })
                child_idx+=1
            parent_idx+=1

    return parent_chunks,child_chunks
            

##==================================
## Late Chunking Strategy
##==================================
def late_base_chunk(document:List[Dict[str,Any]])->List[Dict[str,Any]]:
    """
    Docstring for late_chunk
    
    :param document: Description
    :type document: List[Dict[str, Any]]
    :return: Description
    :rtype: List[Dict[str, Any]]

    create large base chunks for late chunking strategy.

    This function runs during ingestion.
    Only large base passages are created and stored.
    No fine-grained splitting happens here.

    Args:
     document: List of documents with url,title,content.
    
    Retunrs:
      List of base chunks ready for vector indexing.
    """
    chunks=[] ##to store each chunk
    chunk_idx=0 ## assign unique chunk id for each chink

    ## define late chunking parameters
    base_char_size=LATE_CHUNK_BASE_SIZE*4


    ## Base splitter
    base_splitter=RecursiveCharacterTextSplitter(
        chunk_size=base_char_size,
        chunk_overlap=200,
        length_function=len
    )

    ## loop through each document
    for doc in document:
        url=doc["url"]
        title=doc["title"]
        content=doc["content"]

        ## extract structured meta data 
        property_id=doc.get("property_id")
        price=doc.get("price")
        bedrooms=doc.get("bedrooms")
        bathrooms=doc.get("bathrooms")
        amenities=doc.get("amenities",[])
        agent=doc.get("agent")

        ## apply base splitter(large chunks)
        base_chunks=base_splitter.split_text(content)

        ## loop through each base chunk
        for base_chunk in base_chunks:
            if not base_chunk.strip():
                continue
            
            chunks.append({
                "url":url,
                "titile":title,
                "property_id":property_id,
                "price":price,
                "bedrooms":bedrooms,
                "bathrooms":bathrooms,
                "amenities":amenities,
                "agent":agent,
                "text":base_chunk.strip(),
                "chunking_strategy":"Late Strategy",
                "token_count":count_token(text=base_chunk,model="gpt-4o"),
                "chunk_idx":chunk_idx

            }) 
            chunk_idx+=1   

    return chunks        

def late_split_with_context(base_text:str,relevent_idx:int)->str:
    """
    Docstring for late_split_with_context
    
    :param base_text: Description
    :type base_text: str
    :param relevent_idx: Description
    :type relevent_idx: int
    :return: Description
    :rtype: str

    Perform late splitting and apply context window expansion.

    This function runs during retrieval.

    steps:
    1. Split the retrieved base chunk into smaller pieces.
    2. Select the relevant split(base on similarity ranking).
    3. Expand using LATE_CONTEXT_WINDOW.
    4. Merge and return final context for LLM.

    Args:
      base_text: Retrieved base chunk text.
      relevant_idx: Index of most relvent split.

    Returns:
      Expanded context string.
    """

    ## define late chunknig paramter
    late_split_char_size=LATE_CHUNK_SPLIT_SIZE*4

    ## late splitter
    late_splitter=RecursiveCharacterTextSplitter(
        chunk_size=late_split_char_size,
        chunk_overlap=100,
        length_function=len
    )

    late_chunks=late_splitter.split_text(base_text)

    ## safety checks
    if relevent_idx>len(late_chunks):
        raise ValueError("relevent idx out of range")
    
    ##Apply context window
    start=max(0,relevent_idx-LATE_CONTEXT_WINDOW)
    end=min(len(late_chunks),relevent_idx+LATE_CONTEXT_WINDOW+1)

    expanded_context=" ".join(late_chunks[start:end])

    return expanded_context


##====================================
## All Chunking Strategies
##====================================

class Chunking:
    """
    Docstring for Chunking

    Chunking Class
    
    Wrapper class that exposes different chunking strategies.

    This class does not implement chunking logic.
    It simply calls the previously defined functions.

    Support strategies:
      -semantic
      -fixed
      -sliding
      -parent_child
      -late
    """

    def __ini__(self):
        pass
    
    def chunk_strategy(self,documents:List[Dict[str,Any]],
                       strategy:str)->List[Dict[str,Any]]:
        """
        Docstring for chunk_strategy
        
        :param self: Description
        :param documents: Description
        :type documents: List[Dict[str, Any]]
        :param strategy: Description
        :type strategy: str
        :return: Description
        :rtype: List[Dict[str, Any]]

        Apply selected chunking strategy.
        """

        if strategy=="fixed":
            return fixed_chunk(documents=documents)
        
        elif strategy=="semantic":
            return sementic_chunk(documents=documents)
        
        elif strategy=="sliding":
            return sliding_chunk(documents=documents)
        
        elif strategy=="parent_child":
            return parent_child_chunk(documents=documents)
        
        elif strategy=="late":
            return late_base_chunk(document=documents)
        
        else:
            raise ValueError(f"Unsupported chunking srategy:{strategy}")
        

        def late_retrieval(self,base_text:str,relevant_idx:int)->str:
            """
            Docstring for late_retrieval
            
            :param self: Description
            :param base_text: Description
            :type base_text: str
            :param relevant_idx: Description
            :type relevant_idx: int
            :return: Description
            :rtype: str

            Applying late splitting + context window expansion.
            Used during retrieval phase
            """

            return late_split_with_context(
                base_text=base_text,
                relevent_idx=relevant_idx
            )
    











                  
              

    