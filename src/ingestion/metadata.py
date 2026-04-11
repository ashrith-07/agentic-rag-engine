# src/ingestion/metadata.py
import uuid
import hashlib
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Metadata attached to every chunk produced by any chunking strategy.
    This schema is stored as Qdrant payload and serialized to JSON.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str                          # sha256 of source PDF binary
    source_file: str                     # original filename
    page_number: int
    section_title: str | None = None
    section_depth: int = 0               # 0=top, 1=H1, 2=H2, 3=H3
    chunk_index: int = 0                 # position in document
    total_chunks: int = 0                # filled in after all chunks produced
    strategy_used: str = "fixed"         # fixed | semantic | hierarchical | structure
    token_count: int = 0
    has_table: bool = False
    has_code: bool = False
    parent_chunk_id: str | None = None   # hierarchical only
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_qdrant_payload(self) -> dict:
        """Serialize for Qdrant vector payload (all fields, JSON-safe)."""
        return self.model_dump()


class ParsedDocument(BaseModel):
    """
    Output of the PDF parser — one per document.
    Contains raw text per page plus document-level metadata.
    """
    doc_id: str
    source_file: str
    total_pages: int
    raw_markdown: str                    # full markdown from pymupdf4llm
    pages: list[dict]                    # list of {page_number, text, has_table, has_code}
    file_size_bytes: int


def compute_doc_id(pdf_bytes: bytes) -> str:
    """SHA-256 hash of PDF binary — stable doc identifier."""
    return hashlib.sha256(pdf_bytes).hexdigest()
