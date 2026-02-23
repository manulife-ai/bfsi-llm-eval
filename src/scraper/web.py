"""Web scraper for bank, insurer, and regulator public pages."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from .base import BaseScraper, ScrapedContent

logger = logging.getLogger(__name__)


class WebScraper(BaseScraper):
    """Scrape public web pages from financial institutions and regulators."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.targets: list[dict] = config.get("targets", [])
        self.respect_robots: bool = config.get("respect_robots_txt", True)

    def scrape(self) -> list[ScrapedContent]:
        results: list[ScrapedContent] = []
        for target in self.targets:
            name = target["name"]
            url = target["url"]
            domain = target.get("domain", "general_financial_literacy")
            geo = target.get("geography", "general")

            cache_key = self._cache_key("web", name, url)
            cached = self._get_cached(cache_key)
            if cached is not None:
                results.append(cached)
                continue

            logger.info("Scraping web: %s (%s)", name, url)
            html = self._fetch_url(url, respect_robots=self.respect_robots)
            if html is None:
                continue

            text = self._extract_text(html)
            if not self._passes_length_filter(text):
                logger.info("Skipping %s: content too short", name)
                continue

            sc = ScrapedContent(
                source_type="web",
                source_name=name,
                domain=domain,
                geography=geo,
                text=text,
                url=url,
            )
            sc.chunk()
            self._save_cache(cache_key, sc)
            results.append(sc)

        return results

    @staticmethod
    def _extract_text(html: str) -> str:
        """Extract clean paragraph text from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer, header elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get text from remaining paragraphs and headings
        blocks = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                blocks.append(text)

        combined = "\n\n".join(blocks)
        # Collapse whitespace
        combined = re.sub(r"[ \t]+", " ", combined)
        combined = re.sub(r"\n{3,}", "\n\n", combined)
        return combined.strip()
