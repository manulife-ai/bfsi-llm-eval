"""API-based scrapers for SEC EDGAR and OSFI guidelines."""

from __future__ import annotations

import logging

import requests

from .base import BaseScraper, ScrapedContent

logger = logging.getLogger(__name__)


class APIScraper(BaseScraper):
    """Scrape content from free financial APIs (SEC EDGAR, OSFI)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.sec_config = config.get("sec_edgar", {})
        self.osfi_config = config.get("osfi_guidelines", {})

    def scrape(self) -> list[ScrapedContent]:
        results: list[ScrapedContent] = []

        if self.sec_config.get("enabled", False):
            results.extend(self._scrape_sec_edgar())

        if self.osfi_config.get("enabled", False):
            results.extend(self._scrape_osfi())

        return results

    def _scrape_sec_edgar(self) -> list[ScrapedContent]:
        """Scrape SEC EDGAR full-text search for financial filings."""
        base_url = self.sec_config.get(
            "base_url", "https://efts.sec.gov/LATEST/search-index"
        )
        results: list[ScrapedContent] = []

        queries = [
            "insurance regulation filing",
            "bank holding company annual report",
            "investment adviser compliance",
            "retirement plan fiduciary",
        ]

        for query in queries:
            cache_key = self._cache_key("api_sec", query)
            cached = self._get_cached(cache_key)
            if cached is not None:
                results.append(cached)
                continue

            logger.info("Querying SEC EDGAR: %s", query)
            try:
                self._rate_limit("efts.sec.gov")
                resp = requests.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={"q": query, "dateRange": "custom",
                            "startdt": "2023-01-01", "enddt": "2025-12-31"},
                    headers={"User-Agent": "BFSIEvalHarness/1.0 research@example.com"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning("SEC EDGAR query failed for '%s': %s", query, e)
                continue

            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[:10]:  # limit per query
                source = hit.get("_source", {})
                text = source.get("file_description", "") or source.get("display_names", [""])[0]
                filing_text = f"{text}\n\n{source.get('form_type', '')} filing"

                if not self._passes_length_filter(filing_text):
                    continue

                sc = ScrapedContent(
                    source_type="api",
                    source_name=f"SEC-{source.get('file_num', 'unknown')}",
                    domain="regulatory_compliance",
                    geography="usa",
                    text=filing_text,
                )
                sc.chunk()
                self._save_cache(cache_key, sc)
                results.append(sc)

        return results

    def _scrape_osfi(self) -> list[ScrapedContent]:
        """Scrape OSFI guidelines page for regulatory content."""
        base_url = self.osfi_config.get(
            "base_url", "https://www.osfi-bsif.gc.ca/en/guidance"
        )
        results: list[ScrapedContent] = []

        cache_key = self._cache_key("api_osfi", "guidance_index", base_url)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return [cached]

        logger.info("Scraping OSFI guidelines index")
        html = self._fetch_url(base_url, respect_robots=True)
        if html is None:
            return results

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Extract guideline links and summaries
        blocks = []
        for tag in soup.find_all(["p", "h2", "h3", "li", "td"]):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text.split()) > 5:
                blocks.append(text)

        combined = "\n\n".join(blocks)
        if not self._passes_length_filter(combined):
            return results

        sc = ScrapedContent(
            source_type="api",
            source_name="OSFI-Guidelines",
            domain="regulatory_compliance",
            geography="canada",
            text=combined,
            url=base_url,
        )
        sc.chunk()
        self._save_cache(cache_key, sc)
        results.append(sc)

        return results
