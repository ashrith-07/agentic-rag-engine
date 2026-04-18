# src/retrieval/bm25_index.py
import json
import pickle
import string
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed

# Default path to persist the BM25 index
_INDEX_PATH = Path("data/processed/bm25_index.pkl")
_META_PATH = Path("data/processed/bm25_meta.json")


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer for BM25.
    Lowercases, removes punctuation, splits on whitespace.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    # Remove very short tokens (noise)
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """
    BM25 sparse retrieval index built on top of rank_bm25.

    Complements dense vector search in the hybrid retrieval pipeline.
    The index is persisted to disk so it survives process restarts.

    Usage:
        index = BM25Index()
        index.build(chunks)
        index.save()

        # Later:
        index = BM25Index.load()
        results = index.search("my query", top_k=20)
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def doc_count(self) -> int:
        return len(self._chunks)

    @timed("bm25_build")
    def build(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 index from a list of Chunk objects.
        Replaces any existing index.
        """
        cid = get_correlation_id()

        if not chunks:
            logger.warning(f"[{cid}] BM25: build called with empty chunk list — clearing index")
            self.clear()
            return

        self._chunks = chunks
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        logger.info(f"[{cid}] BM25 index built: {len(chunks)} documents")

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        """
        Add new chunks to an existing index by rebuilding.
        BM25Okapi doesn't support incremental updates — full rebuild.
        """
        all_chunks = self._chunks + new_chunks
        self.build(all_chunks)

    def clear(self) -> None:
        """Clear the full index from memory."""
        self._bm25 = None
        self._chunks = []
        self._corpus_tokens = []
        logger.info("BM25 index cleared in memory.")

    @timed("bm25_search")
    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 search.

        Args:
            query: Raw query string (will be tokenized internally)
            top_k: Number of results to return

        Returns:
            List of result dicts with chunk_id, score, text, metadata
            Sorted by BM25 score descending, zero-score results excluded.
        """
        cid = get_correlation_id()

        if not self.is_built:
            raise RuntimeError("BM25 index not built. Call build() first.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning(f"[{cid}] BM25: empty query after tokenization")
            return []

        scores = self._bm25.get_scores(query_tokens)  # type: ignore

        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue  # BM25 score of 0 = no term overlap
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.metadata.chunk_id,
                "score": round(score, 4),
                "text": chunk.text,
                "metadata": chunk.metadata.to_qdrant_payload(),
            })

        logger.debug(
            f"[{cid}] BM25 search: {len(results)} results "
            f"(top score={results[0]['score'] if results else 0:.3f})"
        )
        return results

    def save(self, index_path: Path = _INDEX_PATH, meta_path: Path = _META_PATH) -> None:
        """Persist the BM25 index and chunk metadata to disk."""
        if not self.is_built and not self._chunks:
            # If cleared, simply remove the cache files if they exist
            if index_path.exists():
                index_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            logger.info("Cleared BM25 persisted files.")
            return

        if not self.is_built:
            raise RuntimeError("Cannot save — index not built.")

        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the BM25 object + corpus tokens
        with open(index_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "corpus_tokens": self._corpus_tokens,
            }, f)

        # Save chunk texts + metadata as JSON (human readable)
        meta = [
            {
                "text": c.text,
                "metadata": c.metadata.to_qdrant_payload(),
            }
            for c in self._chunks
        ]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"BM25 index saved: {index_path} ({len(self._chunks)} docs)")

    @classmethod
    def load(
        cls,
        index_path: Path = _INDEX_PATH,
        meta_path: Path = _META_PATH,
    ) -> "BM25Index":
        """Load a persisted BM25 index from disk."""
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        instance = cls()

        with open(index_path, "rb") as f:
            saved = pickle.load(f)
            instance._bm25 = saved["bm25"]
            instance._corpus_tokens = saved["corpus_tokens"]

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # Reconstruct minimal Chunk objects from saved metadata
        from src.ingestion.chunker import Chunk
        from src.ingestion.metadata import ChunkMetadata

        instance._chunks = [
            Chunk(
                text=item["text"],
                metadata=ChunkMetadata(**item["metadata"]),
            )
            for item in meta
        ]

        logger.info(f"BM25 index loaded: {index_path} ({len(instance._chunks)} docs)")
        return instance


# Module-level singleton
bm25_index = BM25Index()
