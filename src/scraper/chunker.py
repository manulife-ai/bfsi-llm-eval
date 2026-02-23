"""Text chunking into ~500-word passages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int  # position within source document
    word_count: int


def chunk_text(text: str, target_words: int = 500) -> list[Chunk]:
    """Split text into chunks of approximately target_words words.

    Splits on paragraph boundaries (double newlines) when possible,
    falling back to sentence boundaries, then hard word-count splits.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If single paragraph exceeds target, split it by sentences
        if para_words > target_words * 1.5 and not current_parts:
            sentence_chunks = _split_long_paragraph(para, target_words)
            chunks.extend(sentence_chunks)
            continue

        # If adding this paragraph would exceed target, flush current
        if current_words + para_words > target_words * 1.3 and current_parts:
            combined = "\n\n".join(current_parts)
            chunks.append(Chunk(
                text=combined,
                index=len(chunks),
                word_count=current_words,
            ))
            current_parts = []
            current_words = 0

        current_parts.append(para)
        current_words += para_words

    # Flush remaining
    if current_parts:
        combined = "\n\n".join(current_parts)
        chunks.append(Chunk(
            text=combined,
            index=len(chunks),
            word_count=current_words,
        ))

    # Re-index
    for i, c in enumerate(chunks):
        c.index = i

    return chunks


def _split_long_paragraph(text: str, target_words: int) -> list[Chunk]:
    """Split a long paragraph by sentence boundaries, falling back to word splits."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # If no sentence boundaries found, fall back to hard word-count splits
    if len(sentences) <= 1:
        return _split_by_words(text, target_words)

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > target_words * 1.3 and current_sentences:
            combined = " ".join(current_sentences)
            chunks.append(Chunk(
                text=combined,
                index=len(chunks),
                word_count=current_words,
            ))
            current_sentences = []
            current_words = 0
        current_sentences.append(sent)
        current_words += sent_words

    if current_sentences:
        combined = " ".join(current_sentences)
        chunks.append(Chunk(
            text=combined,
            index=len(chunks),
            word_count=current_words,
        ))

    return chunks


def _split_by_words(text: str, target_words: int) -> list[Chunk]:
    """Hard split by word count when no other boundaries exist."""
    words = text.split()
    chunks: list[Chunk] = []
    for i in range(0, len(words), target_words):
        batch = words[i:i + target_words]
        chunks.append(Chunk(
            text=" ".join(batch),
            index=len(chunks),
            word_count=len(batch),
        ))
    return chunks
