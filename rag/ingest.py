from __future__ import annotations

import argparse
import base64
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pypdf import PdfReader

from .config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, VISION_MODEL


logger = logging.getLogger(__name__)


if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please configure it in your environment.")


openai_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
)

vision_client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    page: int | None = None


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf_reader(reader: PdfReader) -> List[str]:
    """Return a list of per-page text blocks from a PdfReader."""
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        pages.append(f"[Page {idx + 1}]\n{text}")
    return pages


def _augment_pdf_with_vision_from_path(path: Path, base_pages: List[str]) -> str:
    """Combine text-layer extraction with vision-based OCR for image-heavy pages."""
    logger.info("Augmenting PDF %s with vision OCR for low-text pages", path.name)
    text_blocks = base_pages[:]
    try:
        import pypdfium2 as pdfium  # type: ignore[import]
    except Exception:  # noqa: BLE001
        # If pdf rendering is unavailable, fall back to text-only extraction.
        return "\n\n".join(text_blocks)

    pdf = pdfium.PdfDocument(str(path))
    for idx, page in enumerate(pdf):
        # Heuristic: if this page already has plenty of text, skip vision OCR.
        existing = base_pages[idx] if idx < len(base_pages) else ""
        if len(existing.strip()) > 200:
            logger.debug(
                "Skipping vision OCR for %s page %s (already %d chars of text)",
                path.name,
                idx + 1,
                len(existing.strip()),
            )
            continue

        try:
            # Render page to image and run through OpenAI vision OCR.
            pil_image = page.render(scale=2).to_pil()
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            logger.info("Running vision OCR for PDF page %s of %s", idx + 1, path.name)
            response = vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an OCR and information extraction assistant. "
                            "Extract all visible text from the page image in reading order. "
                            "For tables, linearize each row into clear sentences such as "
                            "'Data type: ... Data source: ... Timeline: ...'. "
                            f"Prefix your output with '[[Page {idx + 1} Vision OCR]]'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this PDF page image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0,
            )
            vision_text = response.choices[0].message.content or ""
            if vision_text.strip():
                logger.info(
                    "Vision OCR produced %d chars for %s page %s",
                    len(vision_text.strip()),
                    path.name,
                    idx + 1,
                )
                text_blocks.append(vision_text)
        except Exception:  # noqa: BLE001
            logger.exception("Vision OCR failed for PDF page %s of %s", idx + 1, path.name)
            continue

    return "\n\n".join(text_blocks)


def load_pdf(path: Path) -> str:
    """Load a PDF from disk, combining text layer + vision OCR for image-heavy pages."""
    logger.info("Loading PDF from disk: %s", path)
    reader = PdfReader(str(path))
    base_pages = _load_pdf_reader(reader)
    return _augment_pdf_with_vision_from_path(path, base_pages)


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Best-effort text extraction from raw bytes based on file extension."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        logger.info("Extracting text (bytes) from uploaded PDF: %s", filename)
        reader = PdfReader(BytesIO(data))
        base_pages = _load_pdf_reader(reader)
        # Try to augment with vision OCR for image-heavy pages using BytesIO.
        try:
            import pypdfium2 as pdfium  # type: ignore[import]
        except Exception:  # noqa: BLE001
            return "\n\n".join(base_pages)

        pdf = pdfium.PdfDocument(BytesIO(data))
        text_blocks = base_pages[:]
        for idx, page in enumerate(pdf):
            existing = base_pages[idx] if idx < len(base_pages) else ""
            if len(existing.strip()) > 200:
                logger.debug(
                    "Skipping vision OCR for uploaded PDF %s page %s (already %d chars)",
                    filename,
                    idx + 1,
                    len(existing.strip()),
                )
                continue
            try:
                pil_image = page.render(scale=2).to_pil()
                buf = BytesIO()
                pil_image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                logger.info("Running vision OCR for uploaded PDF page %s of %s", idx + 1, filename)
                response = vision_client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an OCR and information extraction assistant. "
                                "Extract all visible text from the page image in reading order. "
                                "For tables, linearize each row into clear sentences such as "
                                "'Data type: ... Data source: ... Timeline: ...'. "
                                f"Prefix your output with '[[Page {idx + 1} Vision OCR]]'."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all text from this PDF page image."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}",
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0,
                )
                vision_text = response.choices[0].message.content or ""
                if vision_text.strip():
                    logger.info(
                        "Vision OCR produced %d chars for uploaded PDF %s page %s",
                        len(vision_text.strip()),
                        filename,
                        idx + 1,
                    )
                    text_blocks.append(vision_text)
            except Exception:  # noqa: BLE001
                logger.exception("Vision OCR failed for uploaded PDF page %s of %s", idx + 1, filename)
                continue

        return "\n\n".join(text_blocks)

    # Image types: use OpenAI vision model to perform OCR / table extraction.
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            b64 = base64.b64encode(data).decode("utf-8")
            logger.info("Running vision OCR for image file: %s", filename)
            response = vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an OCR and information extraction assistant. "
                            "Extract all visible text from the image in reading order. "
                            "For tables, linearize each row into clear sentences such as "
                            "'Data type: ... Data source: ... Timeline: ...'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            return text
        except Exception:  # noqa: BLE001 - fall back to naive decode
            logger.exception("Vision OCR failed for %s; falling back to UTF-8 decode", filename)
            return data.decode("utf-8", errors="ignore")

    # Treat common text/markdown extensions as UTF-8 text
    if suffix in {".md", ".markdown", ".txt"}:
        return data.decode("utf-8", errors="ignore")

    # Fallback: try to decode as UTF-8; non-text types will likely yield little or no content.
    return data.decode("utf-8", errors="ignore")


def iter_files(source_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(source_dir):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
                yield p


def simple_chunk(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """Naive token proxy using whitespace-split words."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_document(path: Path) -> List[DocumentChunk]:
    logger.info("Chunking document from disk: %s", path)
    text = load_pdf(path) if path.suffix.lower() == ".pdf" else load_markdown(path)
    chunks = simple_chunk(text)
    results: List[DocumentChunk] = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        results.append(
            DocumentChunk(
                id=f"{path.name}-{idx}",
                content=chunk,
                source=str(path.relative_to(DATA_DIR)),
            )
        )
    logger.info("Chunked %s into %d chunks", path, len(results))
    return results


def chunk_bytes(filename: str, data: bytes, source_label: str | None = None) -> List[DocumentChunk]:
    """Chunk an in-memory file; used for uploads from the UI."""
    logger.info("Chunking uploaded file %s", filename)
    text = extract_text_from_bytes(filename, data)
    chunks = simple_chunk(text)
    results: List[DocumentChunk] = []
    label = source_label or filename
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        results.append(
            DocumentChunk(
                id=f"{filename}-{idx}",
                content=chunk,
                source=label,
            )
        )
    logger.info("Chunked %s into %d chunks", filename, len(results))
    return results


def _upsert_chunks(all_chunks: Sequence[DocumentChunk]) -> int:
    """Shared helper to embed and store chunks in Chroma."""
    if not all_chunks:
        return 0

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=openai_embedding_fn,
    )

    # Chroma's metadata fields do not accept None; build dicts without None values.
    metadatas = []
    for c in all_chunks:
        meta: dict[str, object] = {"source": c.source}
        if c.page is not None:
            meta["page"] = int(c.page)
        metadatas.append(meta)

    try:
        logger.info("Upserting %d chunks into Chroma collection 'documents'", len(all_chunks))
        # Use upsert so repeated ingests for the same file do not fail on duplicate IDs.
        collection.upsert(
            ids=[c.id for c in all_chunks],
            documents=[c.content for c in all_chunks],
            metadatas=metadatas,
        )
    except Exception as exc:  # noqa: BLE001 - entrypoint, keep error readable
        logger.exception("Failed to add embeddings to Chroma")
        raise SystemExit(
            "Failed to add embeddings to Chroma. "
            "This may be due to OpenAI authentication/network issues or invalid metadata types. "
            "Check your OPENAI_API_KEY and ensure metadata fields are serializable. "
            f"Details: {exc}"
        ) from exc

    logger.info("Finished upserting %d chunks into Chroma", len(all_chunks))
    return len(all_chunks)


def ingest_directory(source_dir: Path) -> None:
    all_chunks: List[DocumentChunk] = []
    for file_path in iter_files(source_dir):
        logger.info("Ingesting file from directory: %s", file_path)
        chunks = chunk_document(file_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No supported documents found to ingest in %s", source_dir)
        return

    added = _upsert_chunks(all_chunks)
    logger.info("Ingested %d chunks into Chroma at %s", added, CHROMA_DIR)


def ingest_uploaded_files(files: Sequence[Tuple[str, bytes]]) -> int:
    """
    Ingest a list of in-memory files (name, data).

    This is used by the FastAPI upload endpoint, typically called
    from the Streamlit UI. Returns the number of chunks added.
    """
    all_chunks: List[DocumentChunk] = []
    for filename, data in files:
        logger.info("Ingesting uploaded in-memory file: %s", filename)
        all_chunks.extend(chunk_bytes(filename, data, source_label=filename))

    if not all_chunks:
        return 0

    return _upsert_chunks(all_chunks)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Ingest Markdown/PDF docs into Chroma.")
    parser.add_argument(
        "--source_dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing Markdown/PDF documents.",
    )
    args = parser.parse_args()
    source = Path(args.source_dir)
    if not source.exists():
        raise SystemExit(f"Source directory {source} does not exist.")
    ingest_directory(source)


if __name__ == "__main__":
    main()
