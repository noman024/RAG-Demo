import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config import DATA_DIR
from rag.query import RagQueryEngine
from rag.ingest import ingest_uploaded_files


# Basic rotating file logger for the FastAPI backend
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler = RotatingFileHandler(
    LOG_DIR / "backend.log",
    maxBytes=5_000_000,  # ~5MB
    backupCount=3,
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Attach handler both to a dedicated backend logger and to the root logger so that
# logs from other modules (e.g. rag.ingest) also end up in backend.log.
logger = logging.getLogger("backend")
root_logger = logging.getLogger()

if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False

if not root_logger.handlers:
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


app = FastAPI(title="Internal Knowledge Assistant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 8
    llm_fallback: bool = True


class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    chunks: List[RetrievedChunk]


class IngestResponse(BaseModel):
    chunks_added: int
    error: Optional[str] = None


rag_engine = RagQueryEngine()


@app.get("/health")
async def health():
    logger.info("Health check requested")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    logger.info(
        "Received query",
        extra={"question": payload.question, "top_k": payload.top_k, "llm_fallback": payload.llm_fallback},
    )
    answer, chunks = rag_engine.answer_question(
        payload.question,
        top_k=payload.top_k,
        allow_llm_fallback=payload.llm_fallback,
    )
    logger.info(
        "Query answered",
        extra={"question": payload.question, "chunk_count": len(chunks)},
    )
    return QueryResponse(
        answer=answer,
        chunks=[
            RetrievedChunk(
                content=c["content"],
                source=c.get("source"),
                page=c.get("page"),
                score=c.get("score"),
            )
            for c in chunks
        ],
    )


@app.post("/ingest_files", response_model=IngestResponse)
async def ingest_files_endpoint(files: List[UploadFile] = File(...)):
    logger.info("Ingestion request received", extra={"file_count": len(files)})
    contents = []
    for f in files:
        data = await f.read()
        # Persist uploads under data/uploads for transparency & debugging.
        uploads_dir = DATA_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        target_path = uploads_dir / f.filename
        try:
            target_path.write_bytes(data)
            logger.info("Saved uploaded file to %s", target_path)
        except OSError:
            logger.warning("Failed to persist uploaded file to disk: %s", target_path)
        contents.append((f.filename, data))

    try:
        added = ingest_uploaded_files(contents)
        return IngestResponse(chunks_added=added)
    except SystemExit as exc:
        # Wrap CLI-style SystemExit from ingest helpers into a structured response
        return IngestResponse(chunks_added=0, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return IngestResponse(chunks_added=0, error=str(exc))
