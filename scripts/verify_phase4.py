import sys
from src.utils.correlation_id import set_correlation_id
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_document
from src.retrieval.vector_store import vector_store
from src.retrieval.bm25_index import bm25_index
from src.retrieval.hybrid_retriever import hybrid_retriever, reciprocal_rank_fusion
from src.retrieval.embeddings import embedding_engine

set_correlation_id()

# ── 1. Parse + chunk ──────────────────────────────────────────────────────────
print("\n── Step 1: Parse + Chunk ──")
doc = parse_pdf("data/raw/RAG-Assignment.pdf")
chunks = chunk_document(doc, strategy="fixed")
print(f"✓ {len(chunks)} chunks ready for indexing")

# Check if chunks are empty to avoid min/max errors, just in case
if not chunks:
    print("❌ No chunks found. Cannot proceed with verification.")
    sys.exit(1)

# ── 2. Embedding engine ───────────────────────────────────────────────────────
print("\n── Step 2: Embedding Engine ──")
sample_texts = [c.text for c in chunks[:2]]
vectors = embedding_engine.embed_texts(sample_texts, model="primary")
print(f"✓ Primary model: {len(vectors)} vectors, dim={len(vectors[0])}")
vectors_s = embedding_engine.embed_texts(sample_texts, model="secondary")
print(f"✓ Secondary model: {len(vectors_s)} vectors, dim={len(vectors_s[0])}")

# ── 3. Qdrant collections ─────────────────────────────────────────────────────
print("\n── Step 3: Qdrant Collections ──")
vector_store.ensure_collections()
info_p = vector_store.get_collection_info("bge_chunks")
info_s = vector_store.get_collection_info("minilm_chunks")
print(f"✓ bge_chunks: {info_p}")
print(f"✓ minilm_chunks: {info_s}")

# ── 4. Upsert ─────────────────────────────────────────────────────────────────
print("\n── Step 4: Upsert ──")
n_primary = vector_store.upsert_chunks(chunks, model="primary")
n_secondary = vector_store.upsert_chunks(chunks, model="secondary")
print(f"✓ Upserted {n_primary} vectors (primary)")
print(f"✓ Upserted {n_secondary} vectors (secondary)")

# ── 5. BM25 index ─────────────────────────────────────────────────────────────
print("\n── Step 5: BM25 Index ──")
bm25_index.build(chunks)
bm25_results = bm25_index.search("RAG pipeline chunking strategy", top_k=3)
print(f"✓ BM25 built: {bm25_index.doc_count} docs")
print(f"✓ BM25 search: {len(bm25_results)} results")
if bm25_results:
    print(f"  top score={bm25_results[0]['score']:.4f}")
bm25_index.save()
print(f"✓ BM25 index saved to disk")

# ── 6. Dense search ───────────────────────────────────────────────────────────
print("\n── Step 6: Dense Search ──")
dense = vector_store.search("document chunking strategy", top_k=3)
print(f"✓ Dense search: {len(dense)} results")
if dense:
    print(f"  top score={dense[0]['score']:.4f}")

# ── 7. Hybrid search (RRF) ────────────────────────────────────────────────────
print("\n── Step 7: Hybrid Search (RRF) ──")
hybrid = hybrid_retriever.search("What is the chunking strategy?", top_k=3)
print(f"✓ Hybrid search: {len(hybrid)} results")
if hybrid:
    print(f"  top rrf_score={hybrid[0]['rrf_score']:.6f}")

# ── 8. Cache hit test ─────────────────────────────────────────────────────────
print("\n── Step 8: Cache Hit ──")
hybrid2 = hybrid_retriever.search("What is the chunking strategy?", top_k=3)
print(f"✓ Second query returned {len(hybrid2)} results (should be cache hit — check logs)")

# ── 9. RRF unit test ─────────────────────────────────────────────────────────
print("\n── Step 9: RRF Function ──")
list_a = [{"chunk_id": "A", "score": 0.9}, {"chunk_id": "B", "score": 0.8}, {"chunk_id": "C", "score": 0.7}]
list_b = [{"chunk_id": "B", "score": 5.2}, {"chunk_id": "D", "score": 4.1}, {"chunk_id": "A", "score": 3.0}]
fused = reciprocal_rank_fusion([list_a, list_b])
assert fused[0]["chunk_id"] == "B", f"Expected B on top (appears in both lists), got {fused[0]['chunk_id']}"
print(f"✓ RRF fusion: B ranks first (appears in both lists) — {[r['chunk_id'] for r in fused]}")

print("\n✅ Phase 4 complete — vector store + hybrid retrieval working")
