import re
from dataclasses import dataclass

from loguru import logger


@dataclass
class Citation:
    """A single citation linking answer text to a source chunk."""

    chunk_id: str
    chunk_id_short: str
    source_file: str
    page_number: int | None
    section_title: str | None


@dataclass
class CitedAnswer:
    """Answer with citations extracted and resolved."""

    answer_text: str
    citations: list[Citation]
    cited_chunk_ids: list[str]


class CitationEngine:
    """
    Resolves chunk ID references in LLM answers back to source metadata.
    """

    # Matches [CHUNK abc12345] or [CHUNK_abc12345] or [abc12345]
    _CITATION_RE = re.compile(
        r"\[(?:CHUNK[_ ]?)?([a-f0-9\-]{6,36})\]",
        re.IGNORECASE,
    )

    def extract(
        self,
        answer_text: str,
        retrieval_results: list[dict],
    ) -> CitedAnswer:
        """
        Extract and resolve citations from an answer.
        """
        # Build lookup: short_id (first 8 chars) → full result dict
        id_lookup: dict[str, dict] = {}
        for result in retrieval_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                id_lookup[chunk_id[:8].lower()] = result
                id_lookup[chunk_id.lower()] = result

        # Extract all citation tags from answer
        matches = self._CITATION_RE.findall(answer_text)
        seen_ids: set[str] = set()
        citations: list[Citation] = []
        cited_chunk_ids: list[str] = []

        for match in matches:
            short_id = match[:8].lower()

            if short_id in seen_ids:
                continue
            seen_ids.add(short_id)

            result = id_lookup.get(short_id) or id_lookup.get(match.lower())
            if not result:
                logger.debug(f"Citation not resolved: [{match}]")
                continue

            meta = result.get("metadata", {})
            full_id = result.get("chunk_id", match)

            citation = Citation(
                chunk_id=full_id,
                chunk_id_short=full_id[:8],
                source_file=meta.get("source_file", "unknown"),
                page_number=meta.get("page_number"),
                section_title=meta.get("section_title"),
            )
            citations.append(citation)
            cited_chunk_ids.append(full_id)

        logger.debug(f"Citation engine: {len(matches)} tags found, {len(citations)} resolved")

        return CitedAnswer(
            answer_text=answer_text,
            citations=citations,
            cited_chunk_ids=cited_chunk_ids,
        )

    def format_citations(self, cited_answer: CitedAnswer) -> str:
        """
        Format citations as a readable reference list.
        """
        if not cited_answer.citations:
            return ""

        lines = ["Sources:"]
        for c in cited_answer.citations:
            page_str = f", page {c.page_number}" if c.page_number else ""
            section_str = f" — {c.section_title}" if c.section_title else ""
            lines.append(f"  [{c.chunk_id_short}] {c.source_file}{page_str}{section_str}")

        return "\n".join(lines)


# Module-level singleton
citation_engine = CitationEngine()
