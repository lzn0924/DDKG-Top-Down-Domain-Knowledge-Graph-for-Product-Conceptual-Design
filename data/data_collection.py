"""
Data collection module: Scrapy-based web crawler for gathering publicly available
domain knowledge data in the home furnishing / product conceptual design domain.

Paper reference: Section 2 (Technical Architecture, Step 1 – Data Acquisition)
  "PYTHON-based web crawling (Scrapy) is employed for gathering publicly
   available domain data."
"""

import json
import os
import re
from typing import Generator, List, Dict, Any
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy import Spider, Request
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response

from config import DATA_DIR


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

class DesignKnowledgeItem(scrapy.Item):
    """Scraped design knowledge item."""
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    category = scrapy.Field()      # e.g., furniture, appliance, material, style
    images = scrapy.Field()        # List of image URLs
    attributes = scrapy.Field()    # Dict of product attributes
    source = scrapy.Field()


# ---------------------------------------------------------------------------
# Spiders
# ---------------------------------------------------------------------------

class DesignKnowledgeSpider(Spider):
    """
    Generic spider for home furnishing design knowledge portals.

    Collects:
      - Product descriptions (furniture, appliances, materials)
      - Design case articles (style, layout, color schemes)
      - User reviews and feedback
      - Engineering/installation guides

    Usage:
        process = CrawlerProcess(settings=SCRAPY_SETTINGS)
        process.crawl(DesignKnowledgeSpider, start_urls=[...])
        process.start()
    """

    name = "design_knowledge"
    custom_settings = {
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 1.5,        # Respect rate limits
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": (
            "Mozilla/5.0 (compatible; DDKGBot/1.0; "
            "+https://example.com/ddkg)"
        ),
        "ITEM_PIPELINES": {
            "data.data_collection.JsonWriterPipeline": 300,
        },
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, start_urls: List[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls or []
        self._seen_urls: set = set()

    def parse(self, response: Response) -> Generator:
        """Dispatch to appropriate parser based on page type."""
        page_type = self._classify_page(response)
        if page_type == "product":
            yield from self._parse_product(response)
        elif page_type == "article":
            yield from self._parse_article(response)
        elif page_type == "listing":
            yield from self._parse_listing(response)

    def _classify_page(self, response: Response) -> str:
        url = response.url
        if any(kw in url for kw in ["/product/", "/item/", "/goods/"]):
            return "product"
        if any(kw in url for kw in ["/article/", "/case/", "/design/"]):
            return "article"
        return "listing"

    def _parse_product(self, response: Response) -> Generator:
        """Extract structured product knowledge."""
        title = response.css("h1::text, .product-title::text").get("").strip()
        description = " ".join(
            response.css(".product-desc *::text").getall()
        ).strip()
        attributes = {}
        for row in response.css(".attr-table tr, .spec-row"):
            key = row.css("td:first-child::text, th::text").get("").strip()
            val = row.css("td:last-child::text").get("").strip()
            if key and val:
                attributes[key] = val

        image_urls = response.css("img.product-img::attr(src)").getall()
        image_urls = [urljoin(response.url, u) for u in image_urls]

        category = self._infer_category(title + " " + description)

        yield DesignKnowledgeItem(
            url=response.url,
            title=title,
            content=description,
            category=category,
            images=image_urls,
            attributes=attributes,
            source="product_page",
        )

        # Follow pagination / related product links
        for href in response.css("a.related-product::attr(href)").getall():
            url = urljoin(response.url, href)
            if url not in self._seen_urls:
                self._seen_urls.add(url)
                yield Request(url, callback=self.parse)

    def _parse_article(self, response: Response) -> Generator:
        """Extract design case articles and knowledge descriptions."""
        title = response.css("h1::text, .article-title::text").get("").strip()
        paragraphs = response.css("article p::text, .content p::text").getall()
        content = " ".join(p.strip() for p in paragraphs if p.strip())
        image_urls = [
            urljoin(response.url, u)
            for u in response.css("article img::attr(src)").getall()
        ]

        category = self._infer_category(title + " " + content)

        yield DesignKnowledgeItem(
            url=response.url,
            title=title,
            content=content,
            category=category,
            images=image_urls,
            attributes={},
            source="article",
        )

    def _parse_listing(self, response: Response) -> Generator:
        """Follow links from listing/category pages."""
        for href in response.css("a::attr(href)").getall():
            url = urljoin(response.url, href)
            if self._is_valid_domain(url, response.url) and url not in self._seen_urls:
                self._seen_urls.add(url)
                yield Request(url, callback=self.parse)

    @staticmethod
    def _infer_category(text: str) -> str:
        """Heuristic category inference from text keywords."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["sofa", "bed", "wardrobe", "cabinet", "desk", "沙发", "床", "柜"]):
            return "furniture"
        if any(k in text_lower for k in ["lamp", "light", "灯"]):
            return "lighting"
        if any(k in text_lower for k in ["tile", "wood", "marble", "paint", "瓷砖", "木材", "大理石"]):
            return "material"
        if any(k in text_lower for k in ["style", "design", "风格", "设计"]):
            return "design_style"
        if any(k in text_lower for k in ["kitchen", "bathroom", "厨房", "卫生间"]):
            return "space"
        return "general"

    @staticmethod
    def _is_valid_domain(url: str, base_url: str) -> bool:
        return urlparse(url).netloc == urlparse(base_url).netloc


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

class JsonWriterPipeline:
    """Writes scraped items to a JSONL file."""

    def open_spider(self, spider: Spider):
        path = os.path.join(DATA_DIR, "raw_data.jsonl")
        self.file = open(path, "w", encoding="utf-8")
        spider.logger.info(f"Writing data to {path}")

    def close_spider(self, spider: Spider):
        self.file.close()

    def process_item(self, item: DesignKnowledgeItem, spider: Spider):
        line = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(line)
        return item


# ---------------------------------------------------------------------------
# Crawler runner
# ---------------------------------------------------------------------------

SCRAPY_SETTINGS = {
    "BOT_NAME": "ddkg_crawler",
    "SPIDER_MODULES": ["data"],
    "NEWSPIDER_MODULE": "data",
    "LOG_LEVEL": "INFO",
    "ROBOTSTXT_OBEY": True,
    "CONCURRENT_REQUESTS": 8,
    "DOWNLOAD_DELAY": 1.5,
    "COOKIES_ENABLED": False,
}


def run_crawler(start_urls: List[str]) -> str:
    """
    Run the DDKG web crawler.

    Args:
        start_urls: Seed URLs for the crawler.

    Returns:
        Path to the output JSONL file.
    """
    process = CrawlerProcess(settings=SCRAPY_SETTINGS)
    process.crawl(DesignKnowledgeSpider, start_urls=start_urls)
    process.start()
    output_path = os.path.join(DATA_DIR, "raw_data.jsonl")
    return output_path


# ---------------------------------------------------------------------------
# Utility: load collected data
# ---------------------------------------------------------------------------

def load_raw_data(path: str = None) -> List[Dict[str, Any]]:
    """Load raw collected items from a JSONL file."""
    path = path or os.path.join(DATA_DIR, "raw_data.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
