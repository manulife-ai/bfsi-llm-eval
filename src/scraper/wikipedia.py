"""Wikipedia scraper using wikipedia-api library."""

from __future__ import annotations

import logging
import re

import wikipediaapi

from .base import BaseScraper, ScrapedContent

logger = logging.getLogger(__name__)

# Map Wikipedia categories to BFSI domains and geography
CATEGORY_DOMAIN_MAP: dict[str, tuple[str, str]] = {
    "canadian_financial_institutions": ("banking", "canada"),
    "insurance_in_canada": ("insurance", "canada"),
    "banking_in_the_united_states": ("banking", "usa"),
    "retirement_in_canada": ("tax_retirement", "canada"),
    "financial_regulation_in_canada": ("regulatory_compliance", "canada"),
    "insurance_in_the_united_states": ("insurance", "usa"),
    "financial_regulation_in_the_united_states": ("regulatory_compliance", "usa"),
    "investment_in_canada": ("investments_wealth", "canada"),
    "taxation_in_canada": ("tax_retirement", "canada"),
    "taxation_in_the_united_states": ("tax_retirement", "usa"),
}


def _normalize_category(cat: str) -> str:
    """Normalize 'Category:Foo_Bar' -> 'foo_bar'."""
    return re.sub(r"^Category:", "", cat).lower().strip()


def _infer_domain_geo(category: str) -> tuple[str, str]:
    """Map a category string to (domain, geography)."""
    norm = _normalize_category(category)
    for key, val in CATEGORY_DOMAIN_MAP.items():
        if key in norm:
            return val
    return ("general_financial_literacy", "general")


class WikipediaScraper(BaseScraper):
    """Scrape Wikipedia articles by category."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="BFSIEvalHarness/1.0 (research; +https://github.com/sabyasm/bfsi-llm-eval)",
            language="en",
        )
        self.categories: list[str] = config.get("categories", [])
        self.max_per_category: int = config.get("max_articles_per_category", 50)

    def scrape(self) -> list[ScrapedContent]:
        results: list[ScrapedContent] = []
        for cat_name in self.categories:
            domain, geo = _infer_domain_geo(cat_name)
            logger.info("Scraping Wikipedia category: %s", cat_name)

            cat = self.wiki.page(cat_name)
            if not cat.exists():
                logger.warning("Category not found: %s", cat_name)
                continue

            members = cat.categorymembers
            count = 0
            for title, page in members.items():
                if count >= self.max_per_category:
                    break
                if page.ns != wikipediaapi.Namespace.MAIN:
                    continue

                cache_key = self._cache_key("wikipedia", title)
                cached = self._get_cached(cache_key)
                if cached is not None:
                    results.append(cached)
                    count += 1
                    continue

                text = page.text
                if not self._passes_length_filter(text):
                    continue

                sc = ScrapedContent(
                    source_type="wikipedia",
                    source_name=title,
                    domain=domain,
                    geography=geo,
                    text=text,
                    url=page.fullurl if hasattr(page, "fullurl") else "",
                )
                sc.chunk()
                self._save_cache(cache_key, sc)
                results.append(sc)
                count += 1

            logger.info("Got %d articles from %s", count, cat_name)

        return results
