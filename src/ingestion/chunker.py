# src/ingestion/chunker.py
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import spacy
from loguru import logger

from src.config import settings
from src.ingestion.doc_type_detector import DocTypeResult, detect_doc_type
from src.ingestion.metadata import ChunkMetadata, ParsedDocument
from src.utils.correlation_id import get_correlation_id
from src.utils.tokenizer import chunk_text_by_tokens, count_tokens


@dataclass
class Chunk:
    """A single chunk of text with its metadata."""
    text: str
    metadata: ChunkMetadata


# ── Base class ────────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.overlap = overlap or settings.default_chunk_overlap

    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split a ParsedDocument into chunks."""
        ...

    def _make_metadata(
        self,
        doc: ParsedDocument,
        page_number: int,
        chunk_index: int,
        strategy: str,
        text: str,
        section_title: str | None = None,
        section_depth: int = 0,
        has_table: bool = False,
        has_code: bool = False,
        parent_chunk_id: str | None = None,
    ) -> ChunkMetadata:
        """Helper — build a ChunkMetadata for a chunk."""
        import hashlib
        # Create a deterministic chunk_id
        payload = f"{doc.doc_id}::{strategy}::{chunk_index}".encode()
        chunk_id = hashlib.md5(payload).hexdigest()
        
        return ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            source_file=doc.source_file,
            page_number=page_number,
            section_title=section_title,
            section_depth=section_depth,
            chunk_index=chunk_index,
            strategy_used=strategy,
            token_count=count_tokens(text),
            has_table=has_table,
            has_code=has_code,
            parent_chunk_id=parent_chunk_id,
        )

    @staticmethod
    def _finalize(chunks: list[Chunk]) -> list[Chunk]:
        """Set total_chunks on all chunks after production."""
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata.total_chunks = total
        return chunks


# ── Strategy A: Fixed-size ────────────────────────────────────────────────────

class FixedChunker(BaseChunker):
    """
    Strategy A: Fixed-size token chunking with overlap.

    Uses tiktoken to split on exact token boundaries.
    Best for: uniform technical docs, API references.
    """

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        cid = get_correlation_id()
        logger.debug(f"[{cid}] FixedChunker: chunk_size={self.chunk_size}, overlap={self.overlap}")

        raw_chunks = chunk_text_by_tokens(
            doc.raw_markdown,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        chunks: list[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            # Estimate page number from chunk position
            page = self._estimate_page(idx, len(raw_chunks), doc.total_pages)

            meta = self._make_metadata(
                doc=doc,
                page_number=page,
                chunk_index=idx,
                strategy="fixed",
                text=text,
                has_table="|" in text and text.count("|") >= 4,
                has_code="```" in text,
            )
            chunks.append(Chunk(text=text, metadata=meta))

        logger.info(f"[{cid}] FixedChunker: produced {len(chunks)} chunks")
        return self._finalize(chunks)

    @staticmethod
    def _estimate_page(chunk_idx: int, total_chunks: int, total_pages: int) -> int:
        """Estimate page number by proportional position."""
        if total_chunks == 0:
            return 1
        ratio = chunk_idx / max(total_chunks - 1, 1)
        return max(1, min(total_pages, round(ratio * (total_pages - 1)) + 1))


# ── Strategy B: Semantic ──────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Strategy B: Semantic chunking on sentence/paragraph boundaries.

    Uses spaCy sentence segmentation. Merges sentences until hitting
    the token soft limit, respecting paragraph breaks as hard stops.
    Best for: narrative text, research papers, articles.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        super().__init__(chunk_size, overlap)
        self._nlp: spacy.Language | None = None

    def _get_nlp(self) -> spacy.Language:
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        return self._nlp

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        cid = get_correlation_id()
        nlp = self._get_nlp()

        # Split on paragraph breaks first (hard boundaries)
        paragraphs = [p.strip() for p in doc.raw_markdown.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        chunk_idx = 0
        current_sentences: list[str] = []
        current_tokens = 0
        soft_max = int(self.chunk_size * 0.9)   # 90% of chunk_size

        for para_idx, paragraph in enumerate(paragraphs):
            # spaCy sentence segmentation within paragraph
            spacy_doc = nlp(paragraph)
            sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

            for sentence in sentences:
                sent_tokens = count_tokens(sentence)

                # Single sentence exceeds limit — add as its own chunk
                if sent_tokens > self.chunk_size:
                    if current_sentences:
                        chunks.append(self._make_chunk(
                            doc, current_sentences, chunk_idx, para_idx,
                        ))
                        chunk_idx += 1
                        current_sentences = []
                        current_tokens = 0
                    chunks.append(self._make_chunk(doc, [sentence], chunk_idx, para_idx))
                    chunk_idx += 1
                    continue

                # Adding this sentence would exceed soft max — flush current
                if current_tokens + sent_tokens > soft_max and current_sentences:
                    chunks.append(self._make_chunk(
                        doc, current_sentences, chunk_idx, para_idx,
                    ))
                    chunk_idx += 1
                    # Keep last sentence for overlap
                    current_sentences = current_sentences[-1:]
                    current_tokens = count_tokens(current_sentences[0]) if current_sentences else 0

                current_sentences.append(sentence)
                current_tokens += sent_tokens

        # Flush remaining
        if current_sentences:
            chunks.append(self._make_chunk(
                doc, current_sentences, chunk_idx, len(paragraphs) - 1,
            ))

        logger.info(f"[{cid}] SemanticChunker: produced {len(chunks)} chunks")
        return self._finalize(chunks)

    def _make_chunk(
        self,
        doc: ParsedDocument,
        sentences: list[str],
        chunk_idx: int,
        para_idx: int,
    ) -> Chunk:
        text = " ".join(sentences)
        page = max(1, min(doc.total_pages, (para_idx * doc.total_pages // max(1, len(doc.pages))) + 1))
        meta = self._make_metadata(
            doc=doc,
            page_number=page,
            chunk_index=chunk_idx,
            strategy="semantic",
            text=text,
        )
        return Chunk(text=text, metadata=meta)


# ── Strategy C: Hierarchical ──────────────────────────────────────────────────

class HierarchicalChunker(BaseChunker):
    """
    Strategy C: Hierarchical chunking on document structure.

    Parses H1/H2/H3 headings from markdown. Each section becomes a chunk.
    Child chunks carry parent_chunk_id for context-window retrieval.
    Best for: technical manuals, textbooks, structured docs.
    """

    # Matches markdown headings: # ## ### ####
    _HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        cid = get_correlation_id()
        sections = self._split_by_headings(doc.raw_markdown)
        chunks: list[Chunk] = []
        chunk_idx = 0

        # Track parent at each depth level
        parent_ids: dict[int, str] = {}

        for section in sections:
            depth = section["depth"]
            title = section["title"]
            text = section["text"].strip()

            if not text:
                continue

            # If section text fits in one chunk
            if count_tokens(text) <= self.chunk_size:
                import hashlib
                payload = f"{doc.doc_id}::hierarchical::{chunk_idx}".encode()
                chunk_id = hashlib.md5(payload).hexdigest()
                
                parent_id = parent_ids.get(depth - 1) if depth > 0 else None

                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_file=doc.source_file,
                    page_number=1,
                    section_title=title,
                    section_depth=depth,
                    chunk_index=chunk_idx,
                    strategy_used="hierarchical",
                    token_count=count_tokens(text),
                    has_table="|" in text and text.count("|") >= 4,
                    has_code="```" in text,
                    parent_chunk_id=parent_id,
                )
                chunks.append(Chunk(text=text, metadata=meta))
                parent_ids[depth] = chunk_id
                chunk_idx += 1

            else:
                # Section is too large — split with fixed chunker, keep parent link
                parent_id = parent_ids.get(depth - 1) if depth > 0 else None
                sub_texts = chunk_text_by_tokens(text, self.chunk_size, self.overlap)

                for sub_idx, sub_text in enumerate(sub_texts):
                    import hashlib
                    payload = f"{doc.doc_id}::hierarchical::{chunk_idx}".encode()
                    chunk_id = hashlib.md5(payload).hexdigest()
                    
                    sub_title = f"{title} (part {sub_idx + 1})" if len(sub_texts) > 1 else title

                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        source_file=doc.source_file,
                        page_number=1,
                        section_title=sub_title,
                        section_depth=depth,
                        chunk_index=chunk_idx,
                        strategy_used="hierarchical",
                        token_count=count_tokens(sub_text),
                        has_table="|" in sub_text,
                        has_code="```" in sub_text,
                        parent_chunk_id=parent_id,
                    )
                    chunks.append(Chunk(text=sub_text, metadata=meta))
                    if sub_idx == 0:
                        parent_ids[depth] = chunk_id
                    chunk_idx += 1

        logger.info(f"[{cid}] HierarchicalChunker: produced {len(chunks)} chunks")
        return self._finalize(chunks)

    def _split_by_headings(self, markdown: str) -> list[dict]:
        """
        Split markdown into sections by heading level.
        Returns list of {depth, title, text} dicts.
        """
        sections: list[dict] = []
        lines = markdown.splitlines(keepends=True)
        current_depth = 0
        current_title: str | None = None
        current_lines: list[str] = []

        for line in lines:
            match = self._HEADING_RE.match(line.rstrip())
            if match:
                # Save previous section
                if current_lines:
                    sections.append({
                        "depth": current_depth,
                        "title": current_title,
                        "text": "".join(current_lines).strip(),
                    })
                current_depth = len(match.group(1))
                current_title = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Flush last section
        if current_lines:
            sections.append({
                "depth": current_depth,
                "title": current_title,
                "text": "".join(current_lines).strip(),
            })

        return sections


# ── Strategy D: Structure-aware ───────────────────────────────────────────────

class StructureChunker(BaseChunker):
    """
    Strategy D: Structure-aware chunking for mixed-content PDFs.

    Keeps tables, code blocks, and lists atomic — never splits mid-structure.
    Best for: developer docs, mixed-content PDFs, documents with tables/code.
    """

    _TABLE_RE = re.compile(r"(\|.+\|\n)+", re.MULTILINE)
    _CODE_RE = re.compile(r"```[\s\S]+?```|~~~[\s\S]+?~~~", re.MULTILINE)
    _LIST_RE = re.compile(r"((?:^[ \t]*[-*+][ \t].+\n?)+)", re.MULTILINE)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        cid = get_correlation_id()
        segments = self._segment(doc.raw_markdown)
        chunks: list[Chunk] = []
        chunk_idx = 0
        current_text = ""
        current_tokens = 0

        for segment in segments:
            seg_text = segment["text"]
            seg_tokens = count_tokens(seg_text)
            is_atomic = segment["atomic"]   # tables/code — never split

            if is_atomic:
                # Flush current buffer first
                if current_text.strip():
                    chunks.append(self._build_chunk(doc, current_text, chunk_idx))
                    chunk_idx += 1
                    current_text = ""
                    current_tokens = 0
                # Add atomic block as its own chunk
                chunks.append(self._build_chunk(
                    doc, seg_text, chunk_idx,
                    has_table=segment.get("is_table", False),
                    has_code=segment.get("is_code", False),
                ))
                chunk_idx += 1

            elif current_tokens + seg_tokens > self.chunk_size and current_text.strip():
                # Flush and start new
                chunks.append(self._build_chunk(doc, current_text, chunk_idx))
                chunk_idx += 1
                current_text = seg_text
                current_tokens = seg_tokens

            else:
                current_text += "\n" + seg_text
                current_tokens += seg_tokens

        # Flush remaining
        if current_text.strip():
            chunks.append(self._build_chunk(doc, current_text, chunk_idx))

        logger.info(f"[{cid}] StructureChunker: produced {len(chunks)} chunks")
        return self._finalize(chunks)

    def _segment(self, markdown: str) -> list[dict]:
        """
        Break markdown into segments: atomic (table/code/list) vs plain text.
        Returns list of {text, atomic, is_table, is_code} dicts.
        """
        # Find all atomic blocks with their positions
        atomic_spans: list[tuple[int, int, dict]] = []

        for m in self._TABLE_RE.finditer(markdown):
            atomic_spans.append((m.start(), m.end(), {"is_table": True, "is_code": False}))
        for m in self._CODE_RE.finditer(markdown):
            atomic_spans.append((m.start(), m.end(), {"is_table": False, "is_code": True}))

        # Sort by start position, resolve overlaps
        atomic_spans.sort(key=lambda x: x[0])
        filtered: list[tuple[int, int, dict]] = []
        last_end = 0
        for start, end, meta in atomic_spans:
            if start >= last_end:
                filtered.append((start, end, meta))
                last_end = end

        segments: list[dict] = []
        pos = 0

        for start, end, meta in filtered:
            # Text before this atomic block
            if pos < start:
                plain = markdown[pos:start].strip()
                if plain:
                    segments.append({"text": plain, "atomic": False})

            # Atomic block
            segments.append({
                "text": markdown[start:end].strip(),
                "atomic": True,
                **meta,
            })
            pos = end

        # Remaining text after last atomic block
        if pos < len(markdown):
            plain = markdown[pos:].strip()
            if plain:
                segments.append({"text": plain, "atomic": False})

        return segments

    def _build_chunk(
        self,
        doc: ParsedDocument,
        text: str,
        chunk_idx: int,
        has_table: bool = False,
        has_code: bool = False,
    ) -> Chunk:
        has_table = has_table or ("|" in text and text.count("|") >= 4)
        has_code = has_code or "```" in text
        meta = self._make_metadata(
            doc=doc,
            page_number=1,
            chunk_index=chunk_idx,
            strategy="structure",
            text=text,
            has_table=has_table,
            has_code=has_code,
        )
        return Chunk(text=text, metadata=meta)


# ── AdaptiveChunker — the auto-selector ──────────────────────────────────────

class AdaptiveChunker:
    """
    Automatically selects the best chunking strategy for a document.

    Runs doc_type_detector on the first 3 pages, then delegates to the
    appropriate strategy class. This is the main entry point.

    Usage:
        chunker = AdaptiveChunker()
        chunks = chunker.chunk(parsed_doc)
        print(chunker.last_detection)  # see what strategy was chosen + why
    """

    _STRATEGY_MAP: dict[str, type[BaseChunker]] = {
        "fixed": FixedChunker,
        "semantic": SemanticChunker,
        "hierarchical": HierarchicalChunker,
        "structure": StructureChunker,
    }

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.overlap = overlap or settings.default_chunk_overlap
        self.last_detection: DocTypeResult | None = None

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Detect document type and chunk accordingly."""
        cid = get_correlation_id()

        detection = detect_doc_type(doc)
        self.last_detection = detection

        logger.info(
            f"[{cid}] AdaptiveChunker selected: {detection.strategy} "
            f"(doc_type={detection.doc_type}, confidence={detection.confidence:.2f})"
        )

        chunker_cls = self._STRATEGY_MAP[detection.strategy]
        chunker = chunker_cls(chunk_size=self.chunk_size, overlap=self.overlap)
        return chunker.chunk(doc)

    def chunk_with_strategy(
        self, doc: ParsedDocument, strategy: str
    ) -> list[Chunk]:
        """Force a specific strategy (useful for comparison notebooks)."""
        if strategy not in self._STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self._STRATEGY_MAP)}")

        chunker_cls = self._STRATEGY_MAP[strategy]
        chunker = chunker_cls(chunk_size=self.chunk_size, overlap=self.overlap)
        return chunker.chunk(doc)


# ── Convenience function ──────────────────────────────────────────────────────

def chunk_document(
    doc: ParsedDocument,
    strategy: str = "auto",
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Top-level convenience function for chunking a ParsedDocument.

    Args:
        doc: Parsed document from parser.py
        strategy: "auto" | "fixed" | "semantic" | "hierarchical" | "structure"
        chunk_size: Override default chunk size
        overlap: Override default overlap

    Returns:
        List of Chunk objects with full metadata
    """
    if strategy == "auto":
        return AdaptiveChunker(chunk_size, overlap).chunk(doc)
    else:
        return AdaptiveChunker(chunk_size, overlap).chunk_with_strategy(doc, strategy)
