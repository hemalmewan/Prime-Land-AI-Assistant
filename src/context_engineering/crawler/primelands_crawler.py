"""
Prime Lands Web Crawler Service

Asynchronous Playwright-based web crawler for extracting property listings
from https://www.primelands.lk.

Featrue:
-JavaScript-rendered content handling
-Breadth-First Search (BFS) traversal with depth control
-Configurable rate limiting and timeout management
-Structured metadata extraction
-JSONL corpus generation for vector indexing
-Markdown export for inspection and debugging
-Robust error handling and graceful failure

This module represents the Data Ingestion Layer of the
Real Estate Intlligence platform(RAG system)
"""

## Import Rquired Libraries

import asyncio
import json
import re
from pathlib import Path
from collections import deque
from urllib.parse import urljoin,urlparse
import sys

from playwright.async_api import async_playwright
import nest_asyncio
import asyncio
from bs4 import BeautifulSoup
import aiofiles

class PrimeLandsCrawler:
    """
    Docstring for PrimeLandsCrawler

    Async web crawler for Prime Lands property listings using Playwrigth.

    Features:
    -JavaScript-rendered page support via headless Chromium
    -Breadth-First Search(BFS) traversal with configurable depth control
    -Maximum page limit to prevent excessive crawling
    -Configurable timeout handling for slow-loading pages
    -Rate limiting between requests for polite crawling
    -Structured metadata extraction for RAG ingestion
    -JSONL corpus generation and Markdown export

    Configuration parameters:
        base_url(str): Root domain to restrict crawling scope.
        max_depth(int): Maximum BFS depth level for link traversal.
        max_pages(int):  number of pages to crawl.
        timeout(int): Page load timeout in milliseconds.
        rate_limit_seconds(float):Delay between requests to avoid overloading the server.

    Usage:
       crawler=PrimeLandsCrawler(base_url,max_depth,max_pages,timeout,rate_limit_seconds)
       asyncio.run(crawler.crawl())

    """

    def __init__(self,base_url:str,max_depth:int,max_pages:int,timeout:int,rate_limit_seconds:float):
        self.base_url=base_url
        self.max_depth=max_depth
        self.max_pages=max_pages
        self.timeout=timeout
        self.rate_limit_seconds=rate_limit_seconds

        self.visited:set[str]=set()
        self.documents:list[dict]=[]
    
    def should_crawl(self,url:str)->bool:
        """
         Check if a URL is eligible to crawl.
        """
        if url in self.visited:
            return False
        
        ## Restrict to base main
        if not url.startswith(self.base_url):
            return False
        
        ## Skip media files
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|zip|exe|svg|css|js)$',url,re.I):
            return False
        
        parsed_url=urlparse(url)
        path=parsed_url.path.lower()

        if path in ['','/']:
            return True
        
        ## This covers /land/en,house/sin/,/apartment/...
        allowed_categories=['/land','/house','/apartment']

        ## check if the current URL's path start with any of the allowed categories
        is_valid_property_path=any(path.startswith(category) for category in allowed_categories)

        if is_valid_property_path:
            return True
        
        return False

        
        return True

    def extract_property_metadata(self, soup, url):
        
        ## 1 Property ID
        property_id = url.rstrip("/").split("/")[-1]

        ## 2 Title and Address
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else None

        address = None
        icon_pin = soup.find(class_=re.compile("fa-map-marker|location-icon", re.I))
        if icon_pin and icon_pin.parent:
            address = icon_pin.parent.get_text(strip=True)

        ## 3 Price
        price = None 
     
        price_tag = soup.find(string=re.compile(r'(LKR|Rs\.|Price)', re.I))
        if price_tag and price_tag.parent:
            price = price_tag.parent.get_text(separator=" ", strip=True)

        ## 4 Bedrooms and Bathrooms
        bedrooms = None
        bathrooms = None

        specs = soup.find_all(["li", "span", "div"])
        for spec in specs:
            text = spec.get_text(separator=" ", strip=True).lower()
            
            if "bed" in text and not bedrooms:
                match = re.search(r'(\d+[\-\d]*)\s*bed', text, re.I)
                if match: bedrooms = match.group(1)
                
            elif "bath" in text and not bathrooms:
                match = re.search(r'(\d+[\-\d]*)\s*bath', text, re.I)
                if match: bathrooms = match.group(1)
        
        ## 5 Amenities
        amenities = []
        amenity_section = soup.find(["div", "section"], class_=re.compile("amenities|facilities|features", re.I))

        if amenity_section:
            items = amenity_section.find_all("li")
            amenities = [item.get_text(strip=True) for item in items if item.get_text(strip=True)]

        return {
            "property_id": property_id,
            "title": title,
            "address": address,
            "price": price,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": None,
            "amenities": amenities,
            "agent": None
        }




    def extract_content(self,soup:BeautifulSoup,url:str)->dict:
        """
        Docstring for extract_content
        
        :param self: Description
        :param soup: Description
        :type soup: BeautifulSoup
        :param url: Description
        :type url: str
        :return: Description
        :rtype: dict

        Extract property listing content and links from a page.
        """

        for el in soup(["script","style","nav","footer","aside","noscript","iframe"]):
            el.decompose()
        
        title=soup.title.string.strip() if soup.title else url.split("/")[-1]
        headgins=[h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3","h4"])]

        links=[]
        for a in soup.find_all('a',href=True):
            href=a["href"]
            if href.startswith('/'):
                href=urljoin(self.base_url,href)
            
            elif not href.startswith('http'):
                href=urljoin(url,href)
            
            if href.startswith(self.base_url):
                href=href.split('#')[0].split('?')[0]
                if href and href!=url:
                    links.append(href)
        
        main_content=(
            soup.find('div',{'id':'root'}) or
            soup.find('main') or 
            soup.find('article') or
            soup.find('div',{'class':re.compile('content|main|container',re.I)}) or
            soup.body
        )

        content_md=str(main_content) if main_content else str(soup)
        content_md=re.sub(r'\n{3,}','\n\n',content_md).strip()

        ## call the property meta data file
        if re.search(r'/land/|/house/|/apartment/', url) and url.count("/") > 4:
            property_metadata = self.extract_property_metadata(soup, url)
        else:
            property_metadata = {
            "property_id": None,
            "price": None,
            "bedrooms": None,
            "bathrooms": None,
            "sqft": None,
            "amenities": [],
            "agent": None
        }

        return{
            "title":title,
            "headings":headgins,
            "content":content_md,
            "links":list(set(links)),
            **property_metadata
        }
    
    async def crawl_async(self,start_urls:list[str])->list[dict]:
        """
        Docstring for crawl_async
        
        :param self: Description
        :param start_urls: Description
        :type start_urls: list[str]
        :return: Description
        :rtype: list[dict]

        Async BFS crawler with playwright.
        """
        queue=deque([(url,0) for url in start_urls])

        async with async_playwright() as p:
            browser=await p.chromium.launch(headless=True)
            page=await browser.new_page()
            page.set_default_timeout(self.timeout)

            while queue and len(self.documents)< self.max_pages:
                url,depth=queue.popleft()
                if depth > self.max_depth or not self.should_crawl(url):
                    continue

                try:
                    print(f"[{depth}] Crawling:{url}")
                    self.visited.add(url)

                    await page.goto(url,wait_until="networkidle")
                    ##await page.wait_for_timeout(2000) ## small delay for SPA render

                    html= await page.content()
                    soup=BeautifulSoup(html,"html.parser")

                    doc_data=self.extract_content(soup,url)
                    doc_data['url']=url
                    doc_data['depth_level']=depth

                    if len(doc_data['content'])>=100:
                        self.documents.append(doc_data)
                        print(f" Saved ({len(doc_data['content'])} chars)")
                    
                    ## Add new linnks to queue
                    if depth < self.max_depth:
                        for link in doc_data['links']:
                            if link not in self.visited and link not in [u for u,_ in queue]:
                                queue.append((link,depth+1))
                    
                    await asyncio.sleep(self.rate_limit_seconds)

                except Exception as e:
                    print(f"Error:{str(e)[:100]}")
                    continue
            await browser.close()
        return self.documents
    
    def crawl(self,start_urls:list[str])->list[dict]:
        """
        Docstring for crawl
        
        :param self: Description
        :param start_urls: Description
        :type start_urls: list[str]
        :return: Description
        :rtype: list[dict]

        Synchronous wrapper for async crawl.
        """
        # 1. Apply Windows-specific event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        else:
            # For Mac/Linux, use the standard loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        nest_asyncio.apply(loop)

        try:
            return loop.run_until_complete(self.crawl_async(start_urls))
        finally:
            # Optional: clean up the loop if we created a new one, 
            # though in notebooks we often leave it open.
            pass


    

            




    
