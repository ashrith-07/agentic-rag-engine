# src/ingestion/doc_type_detector.py
import re
from dataclasses import dataclass
from loguru import logger

from src.ingestion.metadata import ParsedDocument


# Chunking strategy names
STRATEGY_FIXED = "fixed"
STRATEGY_SEMANTIC = "semantic"
STRATEGY_HIERARCHICAL = "hierarchical"
STRATEGY_STRUCTURE = "structure"


@dataclass
class DocTypeResult:
    """Result of document type detection."""
    strategy: str               # which chunking strategy to use
    doc_type: str               # human label: "technical" | "narrative" | "structured" | "mixed"
    confidence: float           # 0.0–1.0
    signals: dict               # the raw heuristic scores that drove the decision


def detect_doc_type(doc: ParsedDocument) -> DocTypeResult:
    """
    Analyse the first 3 pages of a document and determine the optimal
    chunking strategy using heuristics.

    Decision tree (in priority order):
      table_density > 0.3       → Strategy D (structure-aware)
      header_density > 0.2      → Strategy C (hierarchical)
      avg_sentence_length > 25  → Strategy B (semantic)
      default                   → Strategy A (fixed-size)

    Args:
        doc: ParsedDocument from parser.py

    Returns:
        DocTypeResult with chosen strategy and diagnostic signals
    """
    # Analyse first 3 pages only (fast heuristic)
    sample_pages = doc.pages[:3]
    sample_text = " ".join(p["text"] for p in sample_pages)
    sample_markdown = _get_first_n_chars(doc.raw_markdown, 5000)

    signals = {
        "table_density": _compute_table_density(sample_pages, sample_markdown),
        "header_density": _compute_header_density(sample_markdown),
        "avg_sentence_length": _compute_avg_sentence_length(sample_text),
        "has_code_blocks": _has_code_blocks(sample_markdown),
        "total_pages": doc.total_pages,
    }

    # Decision tree
    if signals["table_density"] > 0.3 or signals["has_code_blocks"]:
        strategy = STRATEGY_STRUCTURE
        doc_type = "mixed"
        confidence = min(0.9, 0.6 + signals["table_density"])

    elif signals["header_density"] > 0.2:
        strategy = STRATEGY_HIERARCHICAL
        doc_type = "structured"
        confidence = min(0.9, 0.5 + signals["header_density"])

    elif signals["avg_sentence_length"] > 25:
        strategy = STRATEGY_SEMANTIC
        doc_type = "narrative"
        confidence = 0.75

    else:
        strategy = STRATEGY_FIXED
        doc_type = "technical"
        confidence = 0.70

    logger.info(
        f"DocType detected: strategy={strategy}, type={doc_type}, "
        f"confidence={confidence:.2f}, signals={signals}"
    )

    return DocTypeResult(
        strategy=strategy,
        doc_type=doc_type,
        confidence=confidence,
        signals=signals,
    )


# ── Private heuristic functions ───────────────────────────────────────────────

def _get_first_n_chars(text: str, n: int) -> str:
    return text[:n]


def _compute_table_density(pages: list[dict], markdown: str) -> float:
    """
    Table density = (pages with tables) / total pages examined.
    Also checks for markdown pipe tables in the raw markdown.
    """
    if not pages:
        return 0.0

    pages_with_tables = sum(1 for p in pages if p.get("has_table", False))
    page_ratio = pages_with_tables / len(pages)

    # Count markdown table rows (lines with |...|...|)
    table_lines = len(re.findall(r"^\|.+\|", markdown, re.MULTILINE))
    total_lines = max(len(markdown.splitlines()), 1)
    markdown_ratio = min(table_lines / total_lines, 1.0)

    return round(max(page_ratio, markdown_ratio), 3)


def _compute_header_density(markdown: str) -> float:
    """
    Header density = header lines / total lines.
    Headers are lines starting with # ## ###.
    """
    lines = markdown.splitlines()
    if not lines:
        return 0.0

    header_lines = sum(1 for line in lines if re.match(r"^#{1,4}\s+\w", line))
    return round(header_lines / len(lines), 3)


def _compute_avg_sentence_length(text: str) -> float:
    """
    Average sentence length in words.
    Splits on common sentence terminators.
    """
    if not text or not text.strip():
        return 0.0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    word_counts = [len(s.split()) for s in sentences]
    return round(sum(word_counts) / len(word_counts), 2)


def _has_code_blocks(markdown: str) -> bool:
    """Detect fenced code blocks (``` or ~~~)."""
    return bool(re.search(r"```[\s\S]+?```|~~~[\s\S]+?~~~", markdown))
