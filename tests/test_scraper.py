"""Tests for scraper layer: chunking, content filtering, caching."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.scraper.chunker import chunk_text, Chunk
from src.scraper.base import BaseScraper, ScrapedContent


# --- Chunker tests ---


class TestChunkText:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ") == []

    def test_single_short_paragraph(self):
        text = "This is a short paragraph with a few words."
        chunks = chunk_text(text, target_words=500)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0

    def test_multiple_paragraphs_under_target(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = chunk_text(text, target_words=500)
        assert len(chunks) == 1  # both fit in one chunk

    def test_splits_at_paragraph_boundary(self):
        # Create two paragraphs of ~100 words each, target 80
        para1 = " ".join(["word"] * 100)
        para2 = " ".join(["other"] * 100)
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text(text, target_words=80)
        assert len(chunks) >= 2

    def test_handles_long_single_paragraph(self):
        # 1000 words in a single paragraph (no double newlines)
        sentences = [f"Sentence number {i} has several words in it." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, target_words=100)
        assert len(chunks) >= 2

    def test_word_counts_are_accurate(self):
        text = "Hello world this is four words.\n\nAnother paragraph with five words here."
        chunks = chunk_text(text, target_words=500)
        for chunk in chunks:
            assert chunk.word_count == len(chunk.text.split())

    def test_indices_sequential(self):
        text = "\n\n".join([" ".join(["word"] * 100) for _ in range(5)])
        chunks = chunk_text(text, target_words=80)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


# --- Content filtering tests ---


class TestContentFilter:
    def test_passes_with_enough_words(self):
        text = " ".join(["word"] * 150)
        assert BaseScraper._passes_length_filter(text) is True

    def test_fails_with_too_few_words(self):
        text = " ".join(["word"] * 50)
        assert BaseScraper._passes_length_filter(text) is False

    def test_exact_threshold(self):
        text = " ".join(["word"] * 100)
        assert BaseScraper._passes_length_filter(text) is True

    def test_custom_threshold(self):
        text = " ".join(["word"] * 50)
        assert BaseScraper._passes_length_filter(text, min_words=50) is True
        assert BaseScraper._passes_length_filter(text, min_words=51) is False


# --- Cache key tests ---


class TestCacheKey:
    def test_deterministic(self):
        k1 = BaseScraper._cache_key("web", "RBC", "https://rbc.com")
        k2 = BaseScraper._cache_key("web", "RBC", "https://rbc.com")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = BaseScraper._cache_key("web", "RBC", "https://rbc.com")
        k2 = BaseScraper._cache_key("web", "TD", "https://td.com")
        assert k1 != k2

    def test_key_length(self):
        k = BaseScraper._cache_key("web", "test", "http://example.com")
        assert len(k) == 16


# --- ScrapedContent tests ---


class TestScrapedContent:
    def test_chunk_method(self):
        sc = ScrapedContent(
            source_type="web",
            source_name="Test",
            domain="banking",
            geography="canada",
            text=" ".join(["word"] * 1000),
        )
        assert sc.chunks == []
        sc.chunk(target_words=200)
        assert len(sc.chunks) >= 2

    def test_chunk_empty_text(self):
        sc = ScrapedContent(
            source_type="web",
            source_name="Test",
            domain="banking",
            geography="canada",
            text="",
        )
        sc.chunk()
        assert sc.chunks == []


# --- Caching round-trip ---


class TestCachingRoundTrip:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.scraper.base.RAW_DATA_DIR", tmp_path)

        class DummyScraper(BaseScraper):
            def scrape(self):
                return []

        s = DummyScraper({})
        sc = ScrapedContent(
            source_type="web",
            source_name="TestBank",
            domain="banking",
            geography="canada",
            text=" ".join(["word"] * 200),
            url="https://example.com",
        )

        key = s._cache_key("web", "TestBank", "https://example.com")
        s._save_cache(key, sc)

        loaded = s._get_cached(key)
        assert loaded is not None
        assert loaded.source_name == "TestBank"
        assert loaded.domain == "banking"
        assert len(loaded.chunks) > 0

    def test_cache_miss(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.scraper.base.RAW_DATA_DIR", tmp_path)

        class DummyScraper(BaseScraper):
            def scrape(self):
                return []

        s = DummyScraper({})
        assert s._get_cached("nonexistent_key") is None


# --- WebScraper HTML extraction ---


class TestWebScraperExtraction:
    def test_extract_text_strips_scripts(self):
        from src.scraper.web import WebScraper
        html = """
        <html><body>
            <script>var x = 1;</script>
            <style>.foo { color: red; }</style>
            <nav>Navigation stuff</nav>
            <p>This is the actual content paragraph.</p>
            <p>Another paragraph with useful information here.</p>
        </body></html>
        """
        text = WebScraper._extract_text(html)
        assert "var x = 1" not in text
        assert "Navigation stuff" not in text
        assert "actual content paragraph" in text
        assert "useful information" in text

    def test_extract_text_empty_html(self):
        from src.scraper.web import WebScraper
        assert WebScraper._extract_text("<html><body></body></html>") == ""


# --- Wikipedia category mapping ---


class TestWikipediaCategoryMapping:
    def test_known_category(self):
        from src.scraper.wikipedia import _infer_domain_geo
        domain, geo = _infer_domain_geo("Category:Insurance_in_Canada")
        assert domain == "insurance"
        assert geo == "canada"

    def test_unknown_category_defaults(self):
        from src.scraper.wikipedia import _infer_domain_geo
        domain, geo = _infer_domain_geo("Category:Random_Stuff")
        assert domain == "general_financial_literacy"
        assert geo == "general"

    def test_us_banking(self):
        from src.scraper.wikipedia import _infer_domain_geo
        domain, geo = _infer_domain_geo("Category:Banking_in_the_United_States")
        assert domain == "banking"
        assert geo == "usa"
