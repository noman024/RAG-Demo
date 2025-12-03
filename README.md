# Internal Knowledge Assistant RAG Demo

**Stack**:

- **Backend**: `FastAPI` (`app/main.py`)
- **RAG core**: `rag.ingest` + `rag.query` using **Chroma** + **OpenAI** (embeddings + chat + vision OCR)
- **Frontend**: `Streamlit` (`ui/app.py`)

**Key Features**:

- **Multimodal ingestion**: Handles Markdown, PDFs, and images (`.png`, `.jpg`, `.jpeg`, `.webp`)
- **Vision OCR**: Extracts text from image-heavy PDF pages and standalone images using OpenAI's vision model (`gpt-4o`)
- **Semantic search**: Uses OpenAI embeddings (`text-embedding-3-small`) for meaning-based retrieval
- **Grounded answers**: All answers include citations showing source documents and pages
- **Improved prompts**: Handles partial name matches and extracts information from tables
- **Structured logging**: Detailed logs in `logs/backend.log` and `logs/ui.log` for observability

The UX is:

- Upload one or more files from the UI (Markdown, text, PDF, or images)
- Click **"Build / update knowledge base from uploads"** to embed them into Chroma
- Ask questions; the system:
  - Uses RAG over your uploaded docs **when context is available** (default `top_k=8`)
  - Falls back to a pure LLM answer (its own knowledge) when there is no context or retrieval fails

## 1. Setup

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
- `POST /query` – body `{"question": "...", "top_k": 8, "llm_fallback": true}` → JSON answer + retrieved chunks
- `POST /ingest_files` – multipart form upload (`files=...`) → `{ "chunks_added": int, "error": str | null }`

**Note**: The `/ingest_files` endpoint has a 10-minute timeout to accommodate large PDFs with vision OCR processing.

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

- Lets you upload multiple files (Markdown, text, PDF, or images - `.png`, `.jpg`, `.jpeg`, `.webp`)
- Calls the FastAPI `/ingest_files` endpoint to embed them into Chroma
- Shows ingestion progress (check `logs/backend.log` for detailed step-by-step logs)
- Lets you adjust `top_k` (default: 8) via sidebar slider
- Lets you toggle **"Allow LLM to answer without context"** (pure LLM fallback)
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
- For PDFs: extract text layer + run vision OCR on image-heavy pages (if `pypdfium2` is installed)
- Chunk documents with a simple overlapping window (~400 words, ~50 word overlap)
- Embed each chunk with `text-embedding-3-small`
- Store them in a persistent Chroma collection under `chroma_store/`

**Note**: For image files (`.png`, `.jpg`, etc.), use the UI upload feature as the CLI currently focuses on text/PDF files.

### 5. Logging

The system generates detailed logs for observability:

- **Backend logs**: `logs/backend.log` - Ingestion progress, query handling, API calls
- **UI logs**: `logs/ui.log` - User interactions, API calls from frontend

To monitor ingestion progress in real-time:

```bash
tail -f logs/backend.log
```

### 6. Smoke tests

With the backend running and a valid OpenAI key:

```bash
source .venv/bin/activate
python - << 'PY'
import requests, json

print("Health:", requests.get("http://127.0.0.1:8000/health", timeout=3).json())
resp = requests.post(
    "http://127.0.0.1:8000/query",
    json={"question": "What does this internal knowledge assistant do?", "top_k": 8},
    timeout=60,
)
print("Query response:", json.dumps(resp.json(), indent=2)[:800])
PY
```

### 7. Example Use Cases

**Multimodal ingestion**:

- Upload a PDF with tables → Vision OCR extracts table content as text
- Upload a screenshot of a document → OCR extracts all visible text
- Upload Markdown files → Direct text extraction

**Query examples**:

- "What was the timeline for computer-composed data submission?" (requires vision OCR for table extraction)
- "Who is Fakhruddin?" (handles partial name matching)
- "What does this system do?" (general knowledge from documents)
