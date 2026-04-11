# src/ingestion/parser.py
import re
from pathlib import Path
from loguru import logger

import pymupdf4llm
import pymupdf

from src.ingestion.metadata import ParsedDocument, compute_doc_id
from src.utils.correlation_id import get_correlation_id


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """
    Parse a PDF file into structured markdown using pymupdf4llm.

    Returns a ParsedDocument with:
    - Full markdown string (tables preserved, headings detected)
    - Per-page breakdown with table/code presence flags
    - doc_id (sha256 of binary) for deduplication

    Args:
        pdf_path: Path to the PDF file

    Returns:
        ParsedDocument ready for chunking
    """
    cid = get_correlation_id()
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected .pdf file, got: {path.suffix}")

    logger.info(f"[{cid}] Parsing PDF: {path.name}")

    # Read raw bytes for doc_id
    pdf_bytes = path.read_bytes()
    doc_id = compute_doc_id(pdf_bytes)

    # Extract full markdown (preserves tables as markdown tables)
    raw_markdown = pymupdf4llm.to_markdown(str(path))

    # Open with pymupdf for per-page info
    doc = pymupdf.open(str(path))
    total_pages = len(doc)
    pages = []

    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text("text")

        # Detect tables: look for pipe characters (markdown table indicator)
        has_table = "|" in page_text and page_text.count("|") >= 4

        # Detect code blocks: look for code fence markers or indented blocks
        has_code = "```" in page_text or "    " in page_text

        pages.append({
            "page_number": page_num + 1,
            "text": page_text.strip(),
            "has_table": has_table,
            "has_code": has_code,
            "char_count": len(page_text),
        })

    doc.close()

    logger.info(
        f"[{cid}] Parsed {path.name}: "
        f"{total_pages} pages, {len(raw_markdown)} chars, "
        f"doc_id={doc_id[:12]}..."
    )

    return ParsedDocument(
        doc_id=doc_id,
        source_file=path.name,
        total_pages=total_pages,
        raw_markdown=raw_markdown,
        pages=pages,
        file_size_bytes=len(pdf_bytes),
    )
