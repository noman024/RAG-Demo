import json
import logging
import os
from typing import Any, Dict, List

import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "ui.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ui")


def call_health() -> Dict[str, Any] | None:
    try:
        logger.info("Calling /health from UI")
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backend health check failed")
        st.error(f"Backend health check failed: {exc}")
        return None


def call_query(question: str, top_k: int = 4, llm_fallback: bool = True) -> Dict[str, Any] | None:
    try:
        payload = {"question": question, "top_k": top_k, "llm_fallback": llm_fallback}
        logger.info("Sending query from UI", extra={"top_k": top_k, "llm_fallback": llm_fallback})
        resp = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query from UI failed")
        st.error(f"Query failed: {exc}")
        return None


def call_ingest_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Dict[str, Any] | None:
    if not uploaded_files:
        st.warning("Please upload at least one file before ingesting.")
        return None

    try:
        files_param = []
        for uf in uploaded_files:
            # `uf.getvalue()` reads the full in-memory file
            files_param.append(
                (
                    "files",
                    (uf.name, uf.getvalue(), uf.type or "application/octet-stream"),
                )
            )

        logger.info("Sending ingestion request from UI", extra={"file_count": len(uploaded_files)})
        resp = requests.post(f"{API_BASE}/ingest_files", files=files_param, timeout=600)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion from UI failed")
        st.error(f"Ingestion failed: {exc}")
        return None


def render_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    with st.expander("Show retrieved context chunks"):
        for idx, ch in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {idx}**")
            meta = []
            if ch.get("source"):
                meta.append(f"source: `{ch['source']}`")
            if ch.get("page") is not None:
                meta.append(f"page: {ch['page']}")
            if ch.get("score") is not None:
                meta.append(f"score: {ch['score']:.4f}")
            if meta:
                st.caption(" | ".join(meta))
            st.code(ch.get("content", ""), language="markdown")


def main() -> None:
    st.set_page_config(page_title="Internal Knowledge Assistant", layout="wide")
    st.title("📚 Internal Knowledge Assistant (RAG Demo)")
    st.write(
        "Upload local documents, build a vector index, and ask questions. "
        "If no documents are available, the assistant can fall back to its own general knowledge."
    )

    with st.sidebar:
        st.header("Backend")
        if st.button("Check /health"):
            health = call_health()
            if health is not None:
                st.success(f"Health: {json.dumps(health)}")
        top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=8)
        llm_fallback = st.checkbox(
            "Allow LLM to answer without context (fallback to its own knowledge)",
            value=True,
        )

    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload any files (Markdown, text, PDF, etc.)",
        accept_multiple_files=True,
    )

    if st.button("Build / update knowledge base from uploads"):
        with st.spinner("Ingesting uploaded files into Chroma..."):
            result = call_ingest_files(uploaded_files or [])
        if result is not None:
            if result.get("error"):
                st.error(f"Ingestion error: {result['error']}")
            else:
                st.success(f"Ingestion complete. Chunks added: {result.get('chunks_added', 0)}")

    st.subheader("2. Ask a question")
    question = st.text_area("Your question", height=100, placeholder="e.g. What does this system do?")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Querying knowledge assistant..."):
            result = call_query(question.strip(), top_k=top_k, llm_fallback=llm_fallback)

        if result is None:
            return

        st.subheader("Answer")
        st.write(result.get("answer", "No answer returned."))
        render_chunks(result.get("chunks", []))


if __name__ == "__main__":
    main()
