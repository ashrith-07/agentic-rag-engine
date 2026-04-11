# src/ingestion/__init__.py
from src.ingestion.metadata import ChunkMetadata, ParsedDocument, compute_doc_id
from src.ingestion.parser import parse_pdf
from src.ingestion.doc_type_detector import detect_doc_type, DocTypeResult
from src.ingestion.chunker import (
    Chunk,
    FixedChunker,
    SemanticChunker,
    HierarchicalChunker,
    StructureChunker,
    AdaptiveChunker,
    chunk_document,
)

__all__ = [
    "ChunkMetadata",
    "ParsedDocument",
    "compute_doc_id",
    "parse_pdf",
    "detect_doc_type",
    "DocTypeResult",
    "Chunk",
    "FixedChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "StructureChunker",
    "AdaptiveChunker",
    "chunk_document",
]
