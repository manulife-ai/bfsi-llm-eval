"""Base scraper with rate limiting, robots.txt checking, and content filtering."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from .chunker import Chunk, chunk_text

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")


@dataclass
class ScrapedContent:
    source_type: str  # "wikipedia", "web", "api"
    source_name: str
    domain: str  # one of the 7 BFSI domains
    geography: str  # "canada", "usa", "general"
    text: str
    url: str = ""
    chunks: list[Chunk] = field(default_factory=list)

    def chunk(self, target_words: int = 500) -> None:
        self.chunks = chunk_text(self.text, target_words=target_words)


class BaseScraper(ABC):
    """Abstract base for all scrapers."""

    MIN_CONTENT_WORDS = 100
    REQUEST_DELAY = 2.0  # seconds between requests to same domain

    def __init__(self, config: dict):
        self.config = config
        self._last_request_time: dict[str, float] = {}
        self._robot_parsers: dict[str, RobotFileParser | None] = {}
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def scrape(self) -> list[ScrapedContent]:
        """Scrape content from this source. Returns list of ScrapedContent."""

    # --- Rate limiting ---

    def _rate_limit(self, domain: str) -> None:
        last = self._last_request_time.get(domain, 0.0)
        elapsed = time.time() - last
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time[domain] = time.time()

    # --- Robots.txt ---

    def _check_robots(self, url: str) -> bool:
        """Return True if URL is allowed by robots.txt."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain not in self._robot_parsers:
            rp = RobotFileParser()
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
                self._robot_parsers[domain] = rp
            except Exception:
                logger.warning("Could not fetch robots.txt for %s", domain)
                self._robot_parsers[domain] = None
        parser = self._robot_parsers[domain]
        if parser is None:
            return True  # allow if robots.txt unavailable
        return parser.can_fetch("*", url)

    # --- Content filtering ---

    @staticmethod
    def _passes_length_filter(text: str, min_words: int | None = None) -> bool:
        threshold = min_words if min_words is not None else BaseScraper.MIN_CONTENT_WORDS
        return len(text.split()) >= threshold

    # --- Caching ---

    @staticmethod
    def _cache_key(source_type: str, source_name: str, url: str = "") -> str:
        raw = f"{source_type}:{source_name}:{url}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> ScrapedContent | None:
        path = RAW_DATA_DIR / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            sc = ScrapedContent(**{k: v for k, v in data.items() if k != "chunks"})
            sc.chunk()
            return sc
        except Exception:
            return None

    def _save_cache(self, key: str, content: ScrapedContent) -> None:
        path = RAW_DATA_DIR / f"{key}.json"
        data = {
            "source_type": content.source_type,
            "source_name": content.source_name,
            "domain": content.domain,
            "geography": content.geography,
            "text": content.text,
            "url": content.url,
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    # --- HTTP helper ---

    def _fetch_url(self, url: str, respect_robots: bool = True) -> str | None:
        """Fetch URL with rate limiting and robots.txt check. Returns text or None."""
        parsed = urlparse(url)
        domain = parsed.netloc

        if respect_robots and not self._check_robots(url):
            logger.info("Blocked by robots.txt: %s", url)
            return None

        self._rate_limit(domain)
        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": "BFSIEvalHarness/1.0 (research; +https://github.com/sabyasm/bfsi-llm-eval)"
            })
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None
