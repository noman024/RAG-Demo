## Internal Knowledge Assistant RAG Demo

**Stack**:
- **Backend**: `FastAPI` (`app/main.py`)
- **RAG core**: `rag.ingest` + `rag.query` using **Chroma** + **OpenAI** (embeddings + chat)
- **Frontend**: `Streamlit` (`ui/app.py`)

The UX is:
- Upload one or more files from the UI (Markdown, text, PDF, or other text-like files)
- Click **“Build / update knowledge base from uploads”** to embed them into Chroma
- Ask questions; the system:
  - Uses RAG over your uploaded docs **when context is available**
  - Falls back to a pure LLM answer (its own knowledge) when there is no context or retrieval fails

### 1. Setup

```bash
cd /home/noman/llm_rag_assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file next to `requirements.txt`:

```bash
OPENAI_API_KEY=your_openai_key_here
```

> **Note**: The tests in this repo currently surface an authentication error because the key in this environment does not have access to the referenced organization. With a valid key, ingestion and querying should succeed.

### 2. Run the FastAPI backend

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Endpoints:
- `GET /health` – simple health check
- `POST /query` – body `{"question": "...", "top_k": 4}` → JSON answer + retrieved chunks
- `POST /ingest_files` – multipart form upload (`files=...`) → `{ "chunks_added": int, "error": str | null }`

On embedding / LLM failures (e.g. invalid API key), `/query` will still return HTTP 200 with a JSON payload describing the error in the `answer` field.

> **Note**: After adding `python-multipart` for uploads, restart `uvicorn` so FastAPI picks it up.

### 3. Run the Streamlit UI

In another terminal:

```bash
cd /home/noman/llm_rag_assistant
source .venv/bin/activate
streamlit run ui/app.py
```

The UI:
- Lets you upload multiple files (any extension; text/markdown/PDF are parsed best)
- Calls the FastAPI `/ingest_files` endpoint to embed them into Chroma
- Lets you toggle **“Allow LLM to answer without context”** (pure LLM fallback)
- Lets you ask a question and visualize:
  - The final answer
  - Retrieved context chunks (source, optional page, similarity score)

### 4. Optional CLI ingestion from a directory

You can still ingest a folder of local docs without the UI:

```bash
source .venv/bin/activate
python -m rag.ingest
```

This will:
- Walk `data/` and load `.md`, `.markdown`, `.txt`, and `.pdf` files
- Chunk them with a simple overlapping window
- Embed each chunk with `text-embedding-3-small`
- Store them in a persistent Chroma collection under `chroma_store/`

### 5. Smoke tests

With the backend running and a valid OpenAI key:

```bash
source .venv/bin/activate
python - << 'PY'
import requests, json

print("Health:", requests.get("http://127.0.0.1:8000/health", timeout=3).json())
resp = requests.post(
    "http://127.0.0.1:8000/query",
    json={"question": "What does this internal knowledge assistant do?", "top_k": 3},
    timeout=60,
)
print("Query response:", json.dumps(resp.json(), indent=2)[:800])
PY
```
