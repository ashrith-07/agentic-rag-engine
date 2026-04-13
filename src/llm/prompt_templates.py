"""
All prompts in one place.
Never scatter prompt strings across the codebase.
"""

from loguru import logger

# ── System prompt: answer generation ─────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a precise document retrieval assistant.

Rules you MUST follow:
1. Answer questions ONLY using the provided context chunks.
2. For EVERY factual claim you make, cite the source chunk ID in [brackets] \
immediately after the claim. Example: "The system uses RRF fusion [CHUNK_042]."
3. If the context does not contain sufficient information to answer, respond \
exactly: "I cannot find sufficient information in the provided documents."
4. Do NOT speculate, infer beyond what is written, or use prior knowledge.
5. Be concise. Do not repeat information already stated.
"""

# ── System prompt: query router ───────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """\
You are a query classifier for a RAG (Retrieval-Augmented Generation) system.

Classify the user query into exactly one of these types:

SIMPLE      - Direct factual question answerable from a single passage.
              Example: "What is the default chunk size?"

ANALYTICAL  - Requires reasoning across multiple passages or synthesis.
              Example: "What are the trade-offs between chunking strategies?"

COMPARATIVE - Explicitly asks to compare two or more things.
              Example: "How does BM25 differ from dense retrieval?"

MULTI_HOP   - Requires chaining multiple facts to reach an answer.
              Example: "Which model has higher MTEB score and what does that mean \
for latency?"

OUT_OF_SCOPE - The question cannot plausibly be answered from a document corpus.
              Example: "What is the weather today?"

Respond with valid JSON only:
{"query_type": "SIMPLE|ANALYTICAL|COMPARATIVE|MULTI_HOP|OUT_OF_SCOPE", \
"reasoning": "one sentence explanation"}
"""

# ── System prompt: hallucination detector ────────────────────────────────────

HALLUCINATION_SYSTEM_PROMPT = """\
You are an answer auditor. Your job is to verify whether claims in an answer \
are supported by provided source chunks.

For each claim in the answer:
- Mark it as SUPPORTED if it is directly stated or clearly implied by a source chunk.
- Mark it as HALLUCINATED if it contradicts or goes beyond what the sources say.
- Mark it as INFERENCE if it is a reasonable inference but not explicitly stated.

Respond with valid JSON only:
{
  "supported_claims": ["claim text..."],
  "hallucinated_claims": ["claim text..."],
  "unsupported_inferences": ["claim text..."],
  "confidence_score": 0.0
}

confidence_score is a float from 0.0 to 1.0:
  1.0 = every claim is fully supported
  0.0 = answer is entirely unsupported
"""

# ── Context formatter ─────────────────────────────────────────────────────────


def format_context(results: list[dict], max_tokens: int = 80000) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is wrapped with a header showing its ID, source, page,
    and section title. This is what makes citation possible — the LLM
    sees the chunk ID and can reference it in its answer.

    Args:
        results: List of retrieval result dicts (from hybrid_retriever)
        max_tokens: Approximate token budget for the entire context block
                    (chunks are truncated if total would exceed this)

    Returns:
        Formatted context string ready to inject into a prompt
    """
    from src.utils.tokenizer import count_tokens

    lines: list[str] = []
    total_tokens = 0

    for result in results:
        meta = result.get("metadata", {})
        chunk_id = result.get("chunk_id", "UNKNOWN")
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section_title") or "—"
        text = result.get("text", "")

        header = f"[CHUNK {chunk_id[:8]} | {source} | page {page} | {section}]"
        block = f"{header}\n{text}\n"
        block_tokens = count_tokens(block)

        if total_tokens + block_tokens > max_tokens:
            logger.debug(f"Context truncated at {total_tokens} tokens")
            break

        lines.append(block)
        total_tokens += block_tokens

    return "\n".join(lines)


def build_answer_messages(query: str, context: str) -> list[dict]:
    """Build the messages list for the answer generation call."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"Context chunks:\n\n{context}\n\n" f"Question: {query}"),
        },
    ]


def build_router_messages(query: str) -> list[dict]:
    """Build the messages list for the query router call."""
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"},
    ]


def build_hallucination_messages(answer: str, context: str) -> list[dict]:
    """Build the messages list for the hallucination detector call."""
    return [
        {"role": "system", "content": HALLUCINATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"Source chunks:\n\n{context}\n\n" f"Answer to audit:\n{answer}"),
        },
    ]
