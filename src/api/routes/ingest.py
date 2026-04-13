from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from src.api.models import IngestResponse
from src.pipeline import pipeline
from src.utils.correlation_id import set_correlation_id

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    strategy: str = Form(default="auto"),
) -> IngestResponse:
    """
    Upload and ingest a PDF document.

    Parses, chunks, embeds, and indexes the document into Qdrant + BM25.
    Returns doc_id and chunk count.

    Supports strategy override: auto | fixed | semantic | hierarchical | structure
    """
    cid = set_correlation_id()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    valid_strategies = {"auto", "fixed", "semantic", "hierarchical", "structure"}
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{strategy}'. Choose from: {valid_strategies}",
        )

    # Save uploaded file to data/raw/
    upload_dir = Path("data/raw")
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / file.filename

    try:
        content = await file.read()
        save_path.write_bytes(content)
        logger.info(f"[{cid}] Uploaded: {file.filename} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Ingest
    try:
        result = pipeline.ingest(str(save_path), strategy=strategy)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[{cid}] Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return IngestResponse(
        doc_id=result.doc_id,
        source_file=result.source_file,
        total_chunks=result.total_chunks,
        strategy_used=result.strategy_used,
    )


@router.post("/path", response_model=IngestResponse)
async def ingest_by_path(
    pdf_path: str = Form(..., description="Absolute or relative path to PDF"),
    strategy: str = Form(default="auto"),
) -> IngestResponse:
    """
    Ingest a PDF already on the server filesystem.
    Used by `make ingest PDF=./data/raw/doc.pdf`.
    """
    cid = set_correlation_id()

    path = Path(pdf_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

    try:
        result = pipeline.ingest(str(path), strategy=strategy)
    except Exception as e:
        logger.error(f"[{cid}] Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return IngestResponse(
        doc_id=result.doc_id,
        source_file=result.source_file,
        total_chunks=result.total_chunks,
        strategy_used=result.strategy_used,
    )
