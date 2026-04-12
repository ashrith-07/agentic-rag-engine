# src/evaluation/test_dataset_generator.py
"""
Reproducible ground truth dataset generator.

Uses Groq (llama-3.3-70b-versatile) to generate query–answer pairs
from ingested chunks. Output saved to data/evaluation/test_dataset.json.

Run via:
    make generate-dataset
    # or directly:
    python -m src.evaluation.test_dataset_generator
"""
import json
import uuid
from pathlib import Path

from groq import Groq
from loguru import logger

from src.config import settings
from src.ingestion.chunker import Chunk

_OUTPUT_PATH = Path("data/evaluation/test_dataset.json")

_GENERATOR_PROMPT = """\
You are building a retrieval evaluation dataset.

Given the following document chunk, generate {count} diverse questions that can be \
answered using ONLY the information in this chunk.

For each question, also provide:
- difficulty: "easy" | "medium" | "hard"
- query_type: "factual" | "analytical" | "comparative" | "multi_hop"
- expected_keywords: 2-4 keywords that should appear in a correct answer

Chunk ID: {chunk_id}
Chunk text:
\"\"\"
{chunk_text}
\"\"\"

Respond with valid JSON only — an array of objects, no preamble:
[
  {{
    "query": "...",
    "difficulty": "easy|medium|hard",
    "query_type": "factual|analytical|comparative|multi_hop",
    "expected_keywords": ["kw1", "kw2"]
  }}
]"""


def generate_test_dataset(
    chunks: list[Chunk],
    target_size: int = 100,
    output_path: Path = _OUTPUT_PATH,
) -> list[dict]:
    """
    Generate a ground truth test dataset from a list of chunks.

    Distributes generation across chunks to get variety.
    Each chunk contributes ~(target_size / len(chunks)) questions.

    Args:
        chunks: Ingested chunks to generate questions from
        target_size: Target number of query-answer pairs (default 100)
        output_path: Where to save the JSON output

    Returns:
        List of dataset entries
    """
    if not chunks:
        raise ValueError("No chunks provided — ingest documents first.")

    client = Groq(api_key=settings.groq_api_key)

    # How many questions per chunk (at least 1, at most 5)
    questions_per_chunk = max(1, min(5, target_size // len(chunks)))
    logger.info(
        f"Generating dataset: {len(chunks)} chunks × "
        f"{questions_per_chunk} questions = "
        f"~{len(chunks) * questions_per_chunk} pairs"
    )

    dataset: list[dict] = []

    for idx, chunk in enumerate(chunks):
        if len(dataset) >= target_size:
            break

        if len(chunk.text.split()) < 20:
            logger.debug(f"Skipping chunk {idx} — too short ({len(chunk.text.split())} words)")
            continue

        prompt = _GENERATOR_PROMPT.format(
            count=questions_per_chunk,
            chunk_id=chunk.metadata.chunk_id,
            chunk_text=chunk.text[:2000],  # cap to avoid huge prompts
        )

        try:
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            generated: list[dict] = json.loads(raw)

            for item in generated:
                entry = {
                    "query_id": str(uuid.uuid4()),
                    "query": item.get("query", ""),
                    "difficulty": item.get("difficulty", "medium"),
                    "query_type": item.get("query_type", "factual"),
                    "relevant_chunk_ids": [chunk.metadata.chunk_id],
                    "expected_keywords": item.get("expected_keywords", []),
                    "source_doc": chunk.metadata.source_file,
                    "generated_by": settings.groq_model,
                    "reviewed": False,
                }
                if entry["query"]:
                    dataset.append(entry)

            logger.info(
                f"Chunk {idx + 1}/{len(chunks)}: "
                f"generated {len(generated)} questions — total so far: {len(dataset)}"
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Chunk {idx}: JSON parse error — {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Chunk {idx}: Groq error — {e}. Skipping.")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Dataset saved: {output_path} "
        f"({len(dataset)} pairs from {len(chunks)} chunks)"
    )

    # Print difficulty breakdown
    difficulties = {}
    for entry in dataset:
        d = entry["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1
    logger.info(f"Difficulty breakdown: {difficulties}")

    return dataset


def load_dataset(path: Path = _OUTPUT_PATH) -> list[dict]:
    """Load existing test dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}\n"
            "Run: make generate-dataset"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    """
    Entry point for: python -m src.evaluation.test_dataset_generator
    Ingests all PDFs in data/raw/ and generates the dataset.
    """
    from src.ingestion.parser import parse_pdf
    from src.ingestion.chunker import chunk_document
    from src.utils.correlation_id import set_correlation_id

    set_correlation_id()

    raw_dir = Path("data/raw")
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDFs found in data/raw/. "
            "Add at least one PDF before generating the dataset."
        )

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_files:
        logger.info(f"Ingesting: {pdf_path.name}")
        doc = parse_pdf(pdf_path)
        chunks = chunk_document(doc, strategy="auto")
        all_chunks.extend(chunks)
        logger.info(f"  → {len(chunks)} chunks")

    logger.info(f"Total chunks across all docs: {len(all_chunks)}")
    dataset = generate_test_dataset(all_chunks, target_size=100)
    print(f"\n✅ Generated {len(dataset)} query-answer pairs")
    print(f"   Saved to: {_OUTPUT_PATH}")
