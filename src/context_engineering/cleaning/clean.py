"""
  clean.py

  Data Transformation Module for Contxt Engineering.
  This script cleans raw crawled HTML documents,removes useless navigation pages,
  and injects structured metadata dirextly into the text for optimal LLM retrieval.

"""
from typing import Dict,Any,Optional
from bs4 import BeautifulSoup
import markdownify
import re

def clean_and_enrich_document(doc:Dict[str,Any])-> Optional[Dict[str,Any]]:
    """
      Cleans the HTML content of a scraped document,prepends available metadata,
      and formats it into clean Markdown suitable for sementic chunking.

      Args:
        doc: The raw dictionary output from the web crawler.

      Returns:
        The cleaned dictionary ready for the Chunking class,or None if the document is a low-value
        page (like a general navigation menu) that should be dropped from the vector database entirely.
    
    """
    url=doc.get("url","").lower()
    ## check if the URL belongs to an index,city,district,or generic page.
    invalid_url_patterns=[
        "/district/",
        "/city/",
        "/ongoing/",
        "/completed/",
        "portfolio-property",
        "contact-us"
    ]
    if any(pattern in url for pattern in invalid_url_patterns):
        return None
    
    url_parts = url.rstrip("/").split("/")
    
    # If URL ends with language tag, the property name is one level up (-2)
    if url_parts[-1] in ["en", "sin", "tam"]:
        actual_slug = url_parts[-2]
    else:
        actual_slug = url_parts[-1]
        
    # If the extracted slug is just a main category drop it
    if actual_slug in ["land", "house", "apartment", "residential", "commercial"]:
        return None
        
    # Overwrite the broken "en" property ID with the real one
    doc["property_id"] = actual_slug.upper().replace("-", " ")
  
    ## raw html content
    raw_html=doc.get("content","")

    ##pass it through BeautifulSoup to strip out non-text elements completely.
    soup=BeautifulSoup(raw_html,"html.parser")

    ## use markdownify to convert HTML tags(<h1>,<h2>) into markdown.
    clean_markdown=markdownify.markdownify(
        str(soup),
        heading_style="ATX", #forces headings to use '#' symbols
        strip=['a','img'] ## strip links and images to save token space
    )

    clean_markdown=re.sub(r'\n{3,}','\n\n',clean_markdown).strip()

    ## inject structured data into the texts.
    metadata_lines=[]

    ## map the exact keys we want to inject
    target_metadata_keys=[
        "title","property_id","address","price",
        "bedrooms","bathrooms","sqft","agent"
    ]

    for key in target_metadata_keys:
        value=doc.get(key)

        if value is not None and value!="":
            ## format the key to be human-readable 
            formatted_key=key.replace("_"," ").title()
            metadata_lines.append(f"{formatted_key}:{value}")

    ## amenities
    amenities=doc.get("amenities",[])
    if amenities:
        ## join the list into a comma-separated string
        amenities_str=", ".join(amenities)
        metadata_lines.append(f"Amenities:{amenities_str}")

    
    if metadata_lines:
        enriched_text="\n".join(metadata_lines)+"\n---\n"+clean_markdown

    else:
        enriched_text=clean_markdown
    
    ##update the document dictionary with the newly enriched text.
    doc["content"]=enriched_text

    return doc










