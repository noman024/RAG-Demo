# Internal Knowledge Assistant Demo

This is a tiny demo document to validate the RAG pipeline.

## What this system does

- It ingests local Markdown and PDF documents.
- It chunks them into smaller pieces.
- It embeds each chunk using OpenAI embeddings.
- It stores the embeddings in a Chroma vector database.
- At query time, it retrieves the most relevant chunks and asks an LLM to answer using only that context.

## Tech stack

- OpenAI for both embeddings and LLM.
- Chroma as the vector database.
- FastAPI for the backend API.
- Streamlit for the web UI.
