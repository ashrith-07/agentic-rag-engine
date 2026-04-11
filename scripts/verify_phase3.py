#!/usr/bin/env python
"""
Phase 3 verification script.
Usage: PYTHONPATH=. .venv/bin/python scripts/verify_phase3.py <path-to-pdf>
       PYTHONPATH=. .venv/bin/python scripts/verify_phase3.py  (auto-finds first PDF in data/raw/)
"""
import sys
from pathlib import Path

from src.utils.correlation_id import set_correlation_id
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_document, AdaptiveChunker

set_correlation_id()

# ── Resolve PDF path ──────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    PDF = sys.argv[1]
else:
    raw_dir = Path("data/raw")
    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No PDF found in data/raw/. Drop any PDF there and re-run.")
        sys.exit(1)
    PDF = str(pdfs[0])

# ── Parse ─────────────────────────────────────────────────────────────────────
doc = parse_pdf(PDF)
print(f"\n✓ Parsed: {doc.source_file}")
print(f"  pages={doc.total_pages}, chars={len(doc.raw_markdown)}, doc_id={doc.doc_id[:12]}...")

# ── Test all 4 strategies ─────────────────────────────────────────────────────
strategies = ["fixed", "semantic", "hierarchical", "structure"]
for strategy in strategies:
    chunks = chunk_document(doc, strategy=strategy)
    token_counts = [c.metadata.token_count for c in chunks]
    print(f"\n✓ {strategy.upper()} strategy: {len(chunks)} chunks")
    print(f"  min={min(token_counts)} max={max(token_counts)} avg={sum(token_counts)//len(token_counts)} tokens")
    print(f"  sample: '{chunks[0].text[:80]}...'")

# ── Test AdaptiveChunker ──────────────────────────────────────────────────────
print("\n── AdaptiveChunker ──")
adaptive = AdaptiveChunker()
chunks = adaptive.chunk(doc)
det = adaptive.last_detection
print(f"✓ Auto-selected: {det.strategy} (doc_type={det.doc_type}, confidence={det.confidence:.2f})")
print(f"  signals: {det.signals}")
print(f"  total chunks: {len(chunks)}")

# ── Verify metadata integrity ─────────────────────────────────────────────────
sample = chunks[0]
print(f"\n✓ Metadata sample:")
print(f"  chunk_id={sample.metadata.chunk_id}")
print(f"  doc_id={sample.metadata.doc_id[:12]}...")
print(f"  strategy={sample.metadata.strategy_used}")
print(f"  tokens={sample.metadata.token_count}")
print(f"  total_chunks={sample.metadata.total_chunks}")

print("\n✅ Phase 3 complete — all chunking strategies working")
