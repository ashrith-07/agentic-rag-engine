import pytest

from src.ingestion.chunker import (
    AdaptiveChunker,
    FixedChunker,
    HierarchicalChunker,
    SemanticChunker,
    StructureChunker,
    chunk_document,
)
from src.ingestion.doc_type_detector import detect_doc_type
from src.ingestion.metadata import ChunkMetadata, ParsedDocument, compute_doc_id
from src.utils.correlation_id import set_correlation_id


@pytest.fixture(autouse=True)
def set_cid():
    set_correlation_id()


class TestChunkMetadata:
    def test_default_chunk_id_is_uuid(self, sample_chunk):
        assert len(sample_chunk.metadata.chunk_id) == 36
        assert sample_chunk.metadata.chunk_id.count("-") == 4

    def test_to_qdrant_payload_includes_all_fields(self, sample_chunk):
        payload = sample_chunk.metadata.to_qdrant_payload()
        required = {
            "chunk_id",
            "doc_id",
            "source_file",
            "page_number",
            "strategy_used",
            "token_count",
            "has_table",
            "has_code",
        }
        assert required.issubset(payload.keys())

    def test_compute_doc_id_is_deterministic(self):
        data = b"hello world"
        assert compute_doc_id(data) == compute_doc_id(data)

    def test_compute_doc_id_different_for_different_content(self):
        assert compute_doc_id(b"aaa") != compute_doc_id(b"bbb")


class TestDocTypeDetector:
    def test_returns_valid_strategy(self, sample_parsed_doc):
        result = detect_doc_type(sample_parsed_doc)
        assert result.strategy in {"fixed", "semantic", "hierarchical", "structure"}

    def test_confidence_is_between_0_and_1(self, sample_parsed_doc):
        result = detect_doc_type(sample_parsed_doc)
        assert 0.0 <= result.confidence <= 1.0

    def test_signals_dict_has_expected_keys(self, sample_parsed_doc):
        result = detect_doc_type(sample_parsed_doc)
        assert "table_density" in result.signals
        assert "header_density" in result.signals
        assert "avg_sentence_length" in result.signals

    def test_table_heavy_doc_selects_structure(self):
        """A markdown doc with many tables should select structure strategy."""
        table_markdown = "\n".join(
            [
                "# Report",
                "| Col1 | Col2 | Col3 |",
                "|------|------|------|",
                "| A    | B    | C    |",
                "| D    | E    | F    |",
            ]
            * 10
        )
        doc = ParsedDocument(
            doc_id="x" * 48,
            source_file="tables.pdf",
            total_pages=1,
            raw_markdown=table_markdown,
            pages=[
                {
                    "page_number": 1,
                    "text": table_markdown,
                    "has_table": True,
                    "has_code": False,
                    "char_count": len(table_markdown),
                }
            ],
            file_size_bytes=1024,
        )
        result = detect_doc_type(doc)
        assert result.strategy == "structure"


class TestFixedChunker:
    def test_produces_chunks(self, sample_parsed_doc):
        chunker = FixedChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(sample_parsed_doc)
        assert len(chunks) > 0

    def test_all_chunks_have_metadata(self, sample_parsed_doc):
        chunks = FixedChunker(chunk_size=200).chunk(sample_parsed_doc)
        for c in chunks:
            assert isinstance(c.metadata, ChunkMetadata)
            assert c.metadata.strategy_used == "fixed"
            assert c.metadata.token_count > 0

    def test_total_chunks_is_consistent(self, sample_parsed_doc):
        chunks = FixedChunker(chunk_size=200).chunk(sample_parsed_doc)
        for c in chunks:
            assert c.metadata.total_chunks == len(chunks)

    def test_chunk_size_respected(self, sample_parsed_doc):
        chunk_size = 100
        chunks = FixedChunker(chunk_size=chunk_size, overlap=1).chunk(sample_parsed_doc)
        for c in chunks:
            assert c.metadata.token_count <= chunk_size + 5


class TestSemanticChunker:
    def test_produces_chunks(self, sample_parsed_doc):
        chunks = SemanticChunker(chunk_size=200).chunk(sample_parsed_doc)
        assert len(chunks) > 0

    def test_strategy_label(self, sample_parsed_doc):
        chunks = SemanticChunker().chunk(sample_parsed_doc)
        assert all(c.metadata.strategy_used == "semantic" for c in chunks)

    def test_chunks_have_text(self, sample_parsed_doc):
        chunks = SemanticChunker().chunk(sample_parsed_doc)
        assert all(len(c.text.strip()) > 0 for c in chunks)


class TestHierarchicalChunker:
    def test_produces_chunks(self, sample_parsed_doc):
        chunks = HierarchicalChunker().chunk(sample_parsed_doc)
        assert len(chunks) > 0

    def test_section_titles_preserved(self, sample_parsed_doc):
        chunks = HierarchicalChunker().chunk(sample_parsed_doc)
        titles = [c.metadata.section_title for c in chunks if c.metadata.section_title]
        assert len(titles) > 0

    def test_strategy_label(self, sample_parsed_doc):
        chunks = HierarchicalChunker().chunk(sample_parsed_doc)
        assert all(c.metadata.strategy_used == "hierarchical" for c in chunks)


class TestStructureChunker:
    def test_produces_chunks(self, sample_parsed_doc):
        chunks = StructureChunker().chunk(sample_parsed_doc)
        assert len(chunks) > 0

    def test_strategy_label(self, sample_parsed_doc):
        chunks = StructureChunker().chunk(sample_parsed_doc)
        assert all(c.metadata.strategy_used == "structure" for c in chunks)

    def test_detects_tables(self, sample_parsed_doc):
        chunks = StructureChunker().chunk(sample_parsed_doc)
        table_chunks = [c for c in chunks if c.metadata.has_table]
        assert len(table_chunks) >= 1


class TestAdaptiveChunker:
    def test_selects_a_valid_strategy(self, sample_parsed_doc):
        adaptive = AdaptiveChunker()
        chunks = adaptive.chunk(sample_parsed_doc)
        assert len(chunks) > 0
        assert adaptive.last_detection is not None
        assert adaptive.last_detection.strategy in {
            "fixed",
            "semantic",
            "hierarchical",
            "structure",
        }

    def test_chunk_with_strategy_override(self, sample_parsed_doc):
        adaptive = AdaptiveChunker()
        for strategy in ["fixed", "semantic", "hierarchical", "structure"]:
            chunks = adaptive.chunk_with_strategy(sample_parsed_doc, strategy)
            assert len(chunks) > 0

    def test_invalid_strategy_raises(self, sample_parsed_doc):
        adaptive = AdaptiveChunker()
        with pytest.raises(ValueError):
            adaptive.chunk_with_strategy(sample_parsed_doc, "nonexistent")

    def test_chunk_document_convenience_fn(self, sample_parsed_doc):
        chunks = chunk_document(sample_parsed_doc, strategy="fixed")
        assert len(chunks) > 0
