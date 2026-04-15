from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from src.api.models import EvalResponse
from src.evaluation.benchmark_runner import run_retrieval_benchmark
from src.utils.correlation_id import set_correlation_id

router = APIRouter(prefix="/eval", tags=["evaluation"])

_REPORT_PATH = Path("data/evaluation/benchmark_report.json")


@router.get("/run", response_model=EvalResponse)
async def run_eval(background_tasks: BackgroundTasks) -> EvalResponse:
    """
    Run the full retrieval benchmark suite.

    Loads the test dataset from data/evaluation/test_dataset.json,
    runs hybrid retrieval for each query, computes all metrics,
    and saves results to data/evaluation/benchmark_report.json.

    This is a synchronous operation that may take 30–120 seconds
    depending on dataset size. For large datasets consider running
    `make eval` from the CLI instead.
    """
    cid = set_correlation_id()
    logger.info(f"[{cid}] Benchmark triggered via API")

    dataset_path = Path("data/evaluation/test_dataset.json")
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Test dataset not found. "
                "Run `make generate-dataset` first."
            ),
        )

    try:
        report = run_retrieval_benchmark(report_path=_REPORT_PATH)
    except Exception as e:
        logger.error(f"[{cid}] Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}")

    return EvalResponse(
        benchmark_config=report.get("benchmark_config", {}),
        retrieval_metrics=report.get("retrieval_metrics", {}),
        report_path=str(_REPORT_PATH),
    )


@router.post("/generate")
async def generate_eval_dataset(target_size: int = 15) -> dict:
    """
    Generate an evaluation dataset from the uploaded documents in data/raw/.
    """
    cid = set_correlation_id()
    logger.info(f"[{cid}] Generating new test dataset via API")

    raw_dir = Path("data/raw")
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(
            status_code=400,
            detail="No PDFs found in data/raw/. Upload a document first."
        )

    from src.ingestion.chunker import chunk_document, Chunk
    from src.ingestion.parser import parse_pdf
    from src.evaluation.test_dataset_generator import generate_test_dataset

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_files:
        try:
            doc = parse_pdf(pdf_path)
            chunks = chunk_document(doc, strategy="auto")
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"[{cid}] Skipping {pdf_path.name} due to parse error: {e}")

    if not all_chunks:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract any text chunks from the uploaded documents."
        )

    try:
        dataset = generate_test_dataset(all_chunks, target_size=target_size)
    except Exception as e:
        logger.error(f"[{cid}] Dataset generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {e}")

    return {
        "message": "Dataset generated successfully",
        "num_pairs": len(dataset),
        "target_size": target_size,
    }


@router.get("/report")
async def get_report() -> dict:
    """
    Return the most recent benchmark report without re-running.
    """
    if not _REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report found. Run GET /eval/run first.",
        )

    import json

    with open(_REPORT_PATH) as f:
        return json.load(f)
