# src/utils/tokenizer.py
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """Load encoder once, reuse forever."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a string.
    Uses cl100k_base encoding (GPT-4 / Groq Llama compatible approximation).
    """
    if not text or not text.strip():
        return 0
    encoder = _get_encoder()
    return len(encoder.encode(text))


def truncate_to_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum token count.
    Truncates at token boundary — no partial tokens.
    """
    if not text:
        return ""
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def chunk_text_by_tokens(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text into token-based chunks with overlap.
    Used by FixedChunker in ingestion layer.

    Args:
        text: Input text to split
        chunk_size: Max tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    encoder = _get_encoder()
    tokens = encoder.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end == len(tokens):
            break
        start += step

    return chunks
