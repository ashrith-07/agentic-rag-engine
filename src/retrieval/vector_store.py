# src/retrieval/vector_store.py

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from src.config import settings
from src.ingestion.chunker import Chunk
from src.retrieval.cache import embedding_cache
from src.retrieval.embeddings import embedding_engine
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed


def _make_client() -> QdrantClient:
    from src.config import settings

    host = settings.qdrant_host.strip()
    if host.startswith("http"):
        # User provided a full URL in QDRANT_HOST
        if settings.qdrant_api_key:
            return QdrantClient(
                url=host,
                api_key=settings.qdrant_api_key,
                timeout=30,
            )
        return QdrantClient(
            url=host,
            timeout=30,
        )

    # User provided just a hostname
    # Cloud mode - uses HTTPS + API key
    if settings.qdrant_api_key:
        return QdrantClient(
            url=f"https://{host}",
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

    # Local mode - plain HTTP
    return QdrantClient(
        host=host,
        port=settings.qdrant_port,
        timeout=30,
    )


class VectorStore:
    """
    Qdrant vector store wrapper.

    Manages two collections:
      - bge_chunks  (768-dim, primary model)
      - minilm_chunks (384-dim, secondary model)

    Features:
      - Idempotent collection creation
      - Embedding cache integration
      - Batch upsert (100 vectors per request)
      - Metadata filtering
      - Cosine similarity search

    Usage:
        store = VectorStore()
        store.ensure_collections()
        store.upsert_chunks(chunks)
        results = store.search("my query", top_k=10)
    """

    BATCH_SIZE = 100

    def __init__(self) -> None:
        self._client = _make_client()

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collections(self) -> None:
        """
        Create both collections if they don't exist.
        Idempotent — safe to call on every startup.
        """
        existing = {c.name for c in self._client.get_collections().collections}

        for collection_name, vector_size in [
            (settings.primary_collection, 768),
            (settings.secondary_collection, 384),
        ]:
            if collection_name not in existing:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name} (dim={vector_size})")
            else:
                logger.debug(f"Collection exists: {collection_name}")

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entirely (use for re-ingestion)."""
        self._client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    def get_collection_info(self, collection_name: str) -> dict:
        """Return count and status for a collection."""
        info = self._client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }

    # ── Upsert ────────────────────────────────────────────────────────────────

    @timed("upsert_chunks")
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        model: str = "primary",
        collection_name: str | None = None,
    ) -> int:
        """
        Embed and upsert chunks into Qdrant.

        Uses embedding cache — if a chunk's text was previously embedded
        with the same model, the cached vector is used directly.

        Args:
            chunks: List of Chunk objects from chunker.py
            model: "primary" or "secondary"
            collection_name: Override default collection

        Returns:
            Number of vectors upserted
        """
        cid = get_correlation_id()

        if not chunks:
            return 0

        col = collection_name or (
            settings.primary_collection if model == "primary"
            else settings.secondary_collection
        )
        model_name = (
            settings.primary_embedding_model if model == "primary"
            else settings.secondary_embedding_model
        )

        texts = [c.text for c in chunks]

        # Check cache for existing embeddings
        cached_vectors, miss_indices = embedding_cache.get_many(texts, model_name)

        # Embed only cache misses
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            logger.info(
                f"[{cid}] Embedding {len(miss_texts)} new chunks "
                f"({len(texts) - len(miss_indices)} cache hits)"
            )
            new_vectors = embedding_engine.embed_texts(miss_texts, model=model)
            embedding_cache.set_many(miss_texts, model_name, new_vectors)

            # Merge back
            new_iter = iter(new_vectors)
            for i in miss_indices:
                cached_vectors[i] = next(new_iter)

        vectors: list[list[float]] = cached_vectors  # type: ignore

        # Batch upsert
        total_upserted = 0
        for batch_start in range(0, len(chunks), self.BATCH_SIZE):
            batch_chunks = chunks[batch_start: batch_start + self.BATCH_SIZE]
            batch_vectors = vectors[batch_start: batch_start + self.BATCH_SIZE]

            points = [
                PointStruct(
                    id=chunk.metadata.chunk_id,
                    vector=vector,
                    payload={
                        **chunk.metadata.to_qdrant_payload(),
                        "text": chunk.text,
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]

            self._client.upsert(collection_name=col, points=points)
            total_upserted += len(points)
            logger.debug(
                f"[{cid}] Upserted batch {batch_start}–"
                f"{batch_start + len(points)} into {col}"
            )

        logger.info(f"[{cid}] Upserted {total_upserted} vectors into {col}")
        return total_upserted

    # ── Search ────────────────────────────────────────────────────────────────

    @timed("vector_search")
    def search(
        self,
        query: str,
        top_k: int | None = None,
        model: str = "primary",
        collection_name: str | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search in Qdrant.

        Args:
            query: Query string
            top_k: Number of results (default: settings.top_k_retrieval)
            model: "primary" or "secondary"
            collection_name: Override default collection
            filters: Optional metadata filters e.g. {"doc_id": "abc123"}

        Returns:
            List of result dicts with text, score, and metadata
        """
        cid = get_correlation_id()
        k = top_k or settings.top_k_retrieval
        col = collection_name or (
            settings.primary_collection if model == "primary"
            else settings.secondary_collection
        )

        query_vector = embedding_engine.embed_query(query, model=model)

        qdrant_filter = _build_filter(filters) if filters else None

        results: list[ScoredPoint] = self._client.search(
            collection_name=col,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        logger.debug(f"[{cid}] Vector search: {len(results)} results from {col}")

        return [_scored_point_to_dict(r) for r in results]

    def delete_by_doc_id(self, doc_id: str, collection_name: str | None = None) -> None:
        """Delete all vectors belonging to a document."""
        from qdrant_client.models import FilterSelector

        for col in [
            collection_name or settings.primary_collection,
            settings.secondary_collection,
        ]:
            self._client.delete(
                collection_name=col,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )]
                    )
                ),
            )
        logger.info(f"Deleted vectors for doc_id={doc_id[:12]}...")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_filter(filters: dict) -> Filter:
    """Convert a plain dict of filters to a Qdrant Filter object."""
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=conditions)


def _scored_point_to_dict(point: ScoredPoint) -> dict:
    """Convert a Qdrant ScoredPoint to a plain dict."""
    payload = point.payload or {}
    return {
        "chunk_id": str(point.id),
        "score": round(float(point.score), 4),
        "text": payload.get("text", ""),
        "metadata": payload,
    }


# Module-level singleton
vector_store = VectorStore()
