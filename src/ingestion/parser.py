# src/ingestion/parser.py
import re
from pathlib import Path

import pymupdf
import pymupdf4llm
from loguru import logger

from src.ingestion.metadata import ParsedDocument, compute_doc_id
from src.utils.correlation_id import get_correlation_id

# Tesseract + OCR imports (optional — only used for image-based PDFs)
try:
    import pytesseract
    from pdf2image import convert_from_path
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


_POPPLER_PATH = "/opt/homebrew/bin"   # macOS Homebrew default


def _is_image_pdf(doc: pymupdf.Document) -> bool:
    """
    Return True if the PDF has no extractable text (scanned/image-only).

    Samples up to 10 pages spread evenly across the document. A PDF is
    considered image-based only if EVERY sampled page has < 20 chars of
    text. This avoids misclassifying large text-heavy documents (books,
    reports) that happen to have image-only cover / decorative pages.
    """
    total = len(doc)
    # Spread sample indices evenly so we cover the whole document
    sample_count = min(10, total)
    step = max(1, total // sample_count)
    indices = list(dict.fromkeys(range(0, total, step)))[:sample_count]  # unique, ordered
    char_counts = [len(doc[i].get_text("text").strip()) for i in indices]
    return all(c < 20 for c in char_counts)


def _ocr_pdf(pdf_path: str, total_pages: int) -> tuple[str, list[dict]]:
    """
    OCR a scanned PDF using Tesseract (via pdf2image).

    Returns:
        (full_markdown_text, per_page_list)
    """
    if not _OCR_AVAILABLE:
        raise RuntimeError(
            "pytesseract / pdf2image not installed. "
            "Run: pip install pytesseract pdf2image  &&  brew install tesseract poppler"
        )

    # 150 DPI balances OCR accuracy vs speed/memory for large documents.
    # 300 DPI on 1000+ pages causes multi-hour waits and potential OOM.
    dpi = 150
    logger.info(
        f"Image-based PDF detected — running Tesseract OCR at {dpi} DPI "
        f"({total_pages} pages, this may take several minutes)"
    )
    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=_POPPLER_PATH)

    pages: list[dict] = []
    full_text_parts: list[str] = []

    for idx, img in enumerate(images):
        page_text = pytesseract.image_to_string(img, lang="eng")
        page_text = page_text.strip()
        full_text_parts.append(page_text)

        has_table = "|" in page_text and page_text.count("|") >= 4
        has_code = "```" in page_text or bool(re.search(r"\bdef \b|\bclass \b|\bimport \b", page_text))

        pages.append({
            "page_number": idx + 1,
            "text": page_text,
            "has_table": has_table,
            "has_code": has_code,
            "char_count": len(page_text),
        })

    # Build a simple markdown representation from OCR text
    full_markdown = "\n\n---\n\n".join(
        f"<!-- page {i + 1} -->\n{t}" for i, t in enumerate(full_text_parts)
    )
    return full_markdown, pages


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """
    Parse a PDF file into structured markdown.

    For text-based PDFs: uses pymupdf4llm (preserves tables, headings).
    For image/scanned PDFs: falls back to Tesseract OCR automatically.

    Returns a ParsedDocument with:
    - Full markdown string
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

    # Read raw bytes for doc_id computation (stable identifier)
    pdf_bytes = path.read_bytes()
    doc_id = compute_doc_id(pdf_bytes)

    # Open with pymupdf to probe for text vs image pages
    doc = pymupdf.open(str(path))
    total_pages = len(doc)

    if _is_image_pdf(doc):
        # ── Scanned / image-based PDF → OCR fallback ─────────────────────────
        logger.warning(
            f"[{cid}] {path.name} appears to be a scanned PDF "
            f"(no embedded text). Falling back to Tesseract OCR."
        )
        doc.close()
        raw_markdown, pages = _ocr_pdf(str(path), total_pages)

    else:
        # ── Text-based PDF → pymupdf4llm (fast, table-preserving) ────────────
        raw_markdown = pymupdf4llm.to_markdown(str(path))
        pages = []

        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text("text")

            has_table = "|" in page_text and page_text.count("|") >= 4
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
