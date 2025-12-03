# Title Slide: Building an Internal RAG Knowledge Assistant

- **Title**: Building an Internal Knowledge Assistant with Retrieval-Augmented Generation (RAG)
- **Speaker**: Machine Learning Engineer
- **Audience**: ML / Software Engineers, Technical Leaders
- **Goal**: Give a clear mental model of RAG and walk through our concrete implementation.

**Speaking Points:**

- **Title**: "Good morning/afternoon everyone. Today I'm going to walk you through building an Internal Knowledge Assistant using Retrieval-Augmented Generation, or RAG for short."
- **Speaker**: "I'm a Machine Learning Engineer, and I've been working on this project to solve a real problem we face in our organization."
- **Audience**: "This talk is designed for ML and Software Engineers, as well as technical leaders who want to understand how to build practical RAG systems."
- **Goal**: "By the end of this presentation, you'll have a clear mental model of what RAG is, how it works, and you'll see a complete working implementation that you can adapt for your own use cases."
- **Introduction**: "The motivation here is simple: internal knowledge is scattered everywhere - in Markdown files, PDFs, wikis, tickets, email threads, Slack messages. Finding the right information when you need it is a constant challenge."
- **Structure**: "We'll move from high-level concepts to architecture, then dive into implementation details, and finally touch on advanced patterns and improvements. The focus is on integration and system design rather than deep model internals."

---

## Agenda

1. Motivation: Why plain LLMs aren't enough
2. RAG Basics: Concepts and core components
3. Our Assistant: Architecture and data flow
4. Implementation Walkthrough: How the code fits together
5. Advanced RAG Variants & Improvements
6. Q&A

**Speaking Points:**

- **Point 1**: "First, we'll discuss why plain large language models aren't sufficient for answering questions about your internal documentation. This sets up the problem we're solving."
- **Point 2**: "Then we'll cover RAG basics - the core concepts of embeddings, vector databases, and how retrieval-augmented generation works at a high level."
- **Point 3**: "Next, I'll show you the architecture of our assistant - how all the pieces fit together, from ingestion to query time."
- **Point 4**: "We'll do a deep dive into the implementation - actual code walkthroughs so you can see how we built this end-to-end."
- **Point 5**: "We'll discuss advanced RAG variants and potential improvements - things like Graph RAG, Hybrid RAG, and production considerations."
- **Point 6**: "Finally, we'll have time for Q&A where we can dive deeper into any specific areas you're interested in."
- **Emphasis**: "Throughout this talk, we're focusing on integration and system design rather than model internals. We're using OpenAI's APIs, but the patterns apply to any LLM provider."

---

## Problem: Knowledge Is Trapped

- **Scattered information**:
  - Markdown, PDFs, wikis, tickets, email threads, Slack messages.
- **Traditional search is brittle**:
  - Exact keyword matching; struggles with synonyms and paraphrasing.
- **LLMs alone hallucinate**:
  - Great at language, not at being an authoritative source of truth.
- **Goal**:
  - Ask natural language questions over internal docs and get **grounded, auditable answers**.

**Speaking Points:**

- **Scattered information**: "Think about where your team's knowledge lives. It's in Markdown files in GitHub repos, PDFs in shared drives, wikis like Confluence or Notion, JIRA tickets, email threads, Slack conversations. There's no single source of truth, and finding information requires knowing where to look."
- **Traditional search is brittle**: "Traditional search engines work on exact keyword matching. If you search for 'incident response' but the document says 'outage handling', you might miss it. They struggle with synonyms, paraphrasing, and understanding intent. You need to know the exact words the document uses."
- **LLMs alone hallucinate**: "Large language models are incredible at generating fluent, natural language. But they're trained on internet-scale data, not your internal docs. They don't know your specific processes, your team's decisions, or your recent changes. And they'll confidently make up answers that sound right but are wrong - that's hallucination."
- **Goal**: "What we want is simple: ask a natural language question like 'What's our incident on-call process?' and get an answer that's grounded in our actual documentation, with citations showing where the information came from. That's what RAG enables."
- **Concrete example**: "For example, imagine asking 'What's our incident on-call process?' The answer might be spread across three different documents - a runbook, a wiki page, and a recent postmortem. Traditional search would require you to find all three and piece them together. An LLM alone might make up a plausible-sounding process. RAG retrieves the relevant chunks from all three sources and generates an answer grounded in that context."

---

## Limitations of Plain LLM Chatbots

- **Static knowledge**:
  - Trained on a snapshot; anything after the training cutoff is invisible to the model.
- **No private data access by default**:
  - The model doesn't know your internal docs, systems, or policies.
- **No personalization**:
  - It can't know "What is my mother's name?" or "What did our last SEV-1 postmortem conclude?".
- **Hallucinations & no citations**:
  - Answers can be fluent but wrong, with no easy way to trace the source.

**Speaking Points:**

- **Static knowledge**: "LLMs are trained on a snapshot of data up to a certain date - their knowledge cutoff. For GPT-4, that's April 2023. Anything that happened after that, any new information, any recent changes to your processes - the model has no idea about it. It's frozen in time."
- **No private data access**: "These models are trained on public internet data. They don't have access to your internal documentation, your private wikis, your company policies, your architecture diagrams. They can't answer questions about your specific systems because they've never seen them."
- **No personalization**: "The model can't know personal information like 'What is my mother's name?' because it doesn't have access to your personal data. Similarly, it can't know about your team's specific events - 'What did our last SEV-1 postmortem conclude?' - because those are private to your organization."
- **Hallucinations & no citations**: "This is the dangerous part. LLMs are so good at generating fluent language that they'll confidently answer questions even when they don't know the answer. They'll make up plausible-sounding responses. And there's no way to trace where that information came from - no citations, no sources. You can't verify if it's correct."
- **Real-world example**: "We've all seen ChatGPT confidently give wrong answers about recent events or internal processes. It sounds authoritative, but it's just making things up based on patterns it learned during training. For internal knowledge systems, this is unacceptable - we need answers we can trust and verify."

---

## From Plain Chatbot to RAG

- **Plain chatbot**:
  - User question → LLM answers from its pre-training only.
  - Works for general internet knowledge, fails for org-specific or up-to-date info.
- **With RAG**:
  - User question → **retrieve relevant internal documents** → LLM reasons over that context.
  - LLM shifts from "oracle" to **reasoning engine over your curated knowledge base**.
- Core idea:
  - Keep using LLMs for reasoning and language, but **separate storage and retrieval of knowledge**.

**Speaking Points:**

- **Plain chatbot flow**: "In a plain chatbot, the flow is simple: user asks a question, and the LLM answers based solely on what it learned during training. This works great for general knowledge questions - 'What is Python?' or 'Explain quantum computing.' But it completely fails for organization-specific information or anything that happened after the training cutoff."
- **RAG flow**: "With RAG, we insert a retrieval step. The user asks a question, we first retrieve relevant chunks from our internal documents, then we feed both the question and those retrieved chunks to the LLM. The LLM now reasons over your actual documentation, not just its training data."
- **LLM as reasoning engine**: "This is a key mental shift. Instead of treating the LLM as an oracle that knows everything, we treat it as a reasoning engine. It's still incredibly powerful at understanding language and synthesizing information, but now it's working over your curated knowledge base, not its own training data."
- **Core idea**: "The fundamental insight is to separate concerns. Keep using LLMs for what they're great at - reasoning and natural language generation. But handle knowledge storage and retrieval separately. Store your documents in a searchable index, retrieve the relevant pieces at query time, and let the LLM reason over that context."
- **Visual aid**: "If I were drawing this, I'd show: Plain chatbot is User → LLM. RAG is User → Retriever → LLM. That retrieval step is what makes all the difference."

---

## What Is Retrieval-Augmented Generation (RAG)?

- **Definition**:
  - RAG = **Retrieval** of relevant external knowledge + **Generation** by an LLM using that knowledge.
- **High-level flow**:
  1. Store your documents in a **vector index**.
  2. At query time, embed the user question.
  3. Retrieve the top‑k most similar chunks.
  4. Feed those chunks to the LLM as **context** in the prompt.
- **Benefits**:
  - Access to **fresh and proprietary** data.
  - Reduced hallucinations via grounding in retrieved context.
  - Better **traceability** and control over what the model sees.

**Speaking Points:**

- **Definition**: "RAG stands for Retrieval-Augmented Generation. It's a two-part process: first, we retrieve relevant external knowledge from our documents, then we use an LLM to generate an answer based on that retrieved knowledge. The retrieval part finds the information, the generation part formulates the answer."
- **Step 1 - Vector index**: "First, we take all our documents and store them in a vector index. This is an offline process - we break documents into chunks, convert each chunk into a vector representation called an embedding, and store those in a vector database. Think of it as creating a searchable index of your knowledge base."
- **Step 2 - Embed the question**: "When a user asks a question, we convert that question into the same kind of vector representation. This allows us to search for semantically similar content, not just keyword matches."
- **Step 3 - Retrieve top-k**: "We search the vector index to find the most similar chunks to the user's question. We typically retrieve the top k chunks - maybe 4, 8, or 10 - that are most relevant to answering the question."
- **Step 4 - Feed to LLM**: "Finally, we take those retrieved chunks and feed them to the LLM as context in the prompt. The prompt says something like 'Here's the relevant context from our documentation, now answer the user's question using only this information.'"
- **Benefit 1 - Fresh data**: "The first major benefit is access to fresh and proprietary data. Your documents can be updated daily, and the system will use the latest information. It's not limited to what the model saw during training."
- **Benefit 2 - Reduced hallucinations**: "By grounding the answer in retrieved context, we dramatically reduce hallucinations. The model is instructed to only use the provided context, and if the answer isn't there, to say 'I don't know' rather than making something up."
- **Benefit 3 - Traceability**: "Finally, we get traceability. Every answer can cite its sources - we know exactly which document chunks were used. This is crucial for internal knowledge systems where you need to verify information."
- **Broad applicability**: "This pattern isn't just for documentation. You can use RAG for logs, tickets, knowledge bases, code repositories, customer support - anywhere you have structured or unstructured text that you want to query in natural language."

---

## RAG: Four Core Steps

1. **Ingestion**  
   - Load documents (Markdown, PDFs, etc.).
   - Clean, chunk, and embed into vectors.
2. **Indexing**  
   - Store vectors + metadata in a **vector database**.
3. **Retrieval**  
   - Convert user question → query embedding.
   - Retrieve top‑k similar chunks from the vector DB.
4. **Generation**  
   - Combine the question + retrieved chunks into a prompt.
   - LLM generates an answer grounded in this context.

**Speaking Points:**

- **Overview**: "Let me break down RAG into four core steps. Everything we build in our system is an implementation of these four steps, so understanding them is crucial."
- **Step 1 - Ingestion**: "Ingestion is the offline process where we prepare our documents. We load them from various sources - Markdown files, PDFs, images. We clean them up, break them into smaller chunks - maybe 400 words each with some overlap. Then we convert each chunk into a vector embedding using an embedding model. This is typically done once or whenever documents are updated."
- **Step 2 - Indexing**: "Indexing is where we store those vectors in a vector database along with metadata. The metadata includes things like the source file name, page number, timestamp. The vector database is optimized for similarity search - finding vectors that are close to each other in the embedding space."
- **Step 3 - Retrieval**: "Retrieval happens at query time. The user asks a question, we convert that question into an embedding using the same model we used during ingestion. Then we search the vector database for the top k most similar chunks. 'Similar' here means semantically similar - not just keyword matching, but meaning-based matching."
- **Step 4 - Generation**: "Finally, generation. We take the user's question and the retrieved chunks, combine them into a prompt, and send it to the LLM. The prompt instructs the model to answer using only the provided context. The LLM generates a natural language answer grounded in that context."
- **Repetition**: "I'll keep coming back to these four steps throughout the talk - ingestion, indexing, retrieval, generation. Every RAG system implements these steps, though the details vary. Our implementation is one concrete example of how to do it."

---

## Key Concept: Embeddings & Vector Search

- **Embeddings**:
  - Map text \(\rightarrow\) dense vectors in \(\mathbb{R}^d\).
  - Semantically similar texts have **nearby vectors**.
- **Vector search**:
  - Instead of keyword match, search for **closest vectors** (e.g., cosine similarity).
  - Enables "meaning-based" retrieval: "How do we handle outages?" ≈ "Incident response procedure".
- In our project:
  - Embedding model: **OpenAI `text-embedding-3-small`**.
  - Vector database: **Chroma**, storing vectors + text + metadata.

**Speaking Points:**

- **Embeddings definition**: "Embeddings are the key to semantic search. An embedding model takes a piece of text - a sentence, a paragraph, a document chunk - and converts it into a dense vector, which is just a list of numbers in a high-dimensional space, typically 1536 dimensions for OpenAI's models."
- **Semantic similarity**: "The magic is that semantically similar texts end up with vectors that are close together in this space. If two sentences mean the same thing, even if they use different words, their vectors will be nearby. This is learned during the model's training on vast amounts of text."
- **Vector search advantage**: "Vector search is fundamentally different from keyword matching. Instead of looking for exact word matches, we compute the distance between vectors - typically using cosine similarity. The closest vectors are the most semantically similar chunks."
- **Meaning-based example**: "This enables meaning-based retrieval. If you ask 'How do we handle outages?' the system can find a document that says 'Incident response procedure' even though it doesn't contain the word 'outage'. The vectors are similar because the meanings are similar."
- **Our embedding model**: "In our project, we use OpenAI's text-embedding-3-small model. It's cost-effective, produces 1536-dimensional vectors, and is optimized for retrieval tasks. There are other options - Cohere, Voyage, open-source models - but OpenAI's embeddings work well for our use case."
- **Our vector database**: "We use Chroma as our vector database. It's simple, runs locally, and stores vectors along with the original text and metadata. The metadata is crucial - it lets us track which document and page each chunk came from, which we need for citations."
- **Analogy**: "Think of embedding space like a map of a city. Similar sentences live in the same neighborhood. Questions about outages cluster together, questions about authentication cluster together. When you ask a question, we find which neighborhood it belongs to and retrieve the nearby chunks."

---

## Our Use Case: Internal Knowledge Assistant

- **Goal**:
  - Let engineers query internal documentation in natural language.
- **Data sources**:
  - Local **Markdown** (`.md`, `.txt`), **PDF** files, and **images** (`.png`, `.jpg`, `.jpeg`, `.webp`).
- **Users**:
  - Engineers, SREs, PMs asking about architecture, runbooks, processes, and decisions.
- **Requirements**:
  - Runs locally on a laptop.
  - Simple, inspectable Python code.
  - Transparent answers with supporting snippets (citations).
  - **Multimodal understanding**: extract text from both text layers and image-only content (tables, screenshots, scanned pages).

**Speaking Points:**

- **Goal**: "Our specific use case is building an internal knowledge assistant. The goal is simple: let engineers query our internal documentation using natural language. Instead of searching through wikis or reading through multiple documents, they can just ask a question and get an answer."
- **Data sources**: "We support multiple data formats. Markdown and text files are straightforward - they're already text. PDFs are more complex because they can have both text layers and image content. We also support standalone images - screenshots, diagrams, scanned documents. This multimodal support is crucial because real documents often mix text and images."
- **Users**: "The primary users are engineers, SREs, and product managers. They're asking questions about system architecture, runbooks for incident response, team processes, and past decisions. These are people who need quick, accurate answers to do their jobs effectively."
- **Requirement 1 - Local**: "One key requirement is that this runs locally on a laptop. We don't want to set up complex infrastructure or cloud services. It should be something you can run on your machine for demos or small teams."
- **Requirement 2 - Simple code**: "The code should be simple and inspectable. We're not using heavy frameworks that abstract away the details. You should be able to read the code and understand exactly what's happening at each step."
- **Requirement 3 - Citations**: "Transparency is crucial. Every answer should come with citations - we show which document chunks were used. This lets users verify the information and dive deeper if needed."
- **Requirement 4 - Multimodal**: "Finally, multimodal understanding is essential. Real documents have tables, screenshots, scanned pages. We need to extract text from both the text layer of PDFs and from image content using OCR. This is what makes the system truly useful for real-world documents."
- **Mapping to your org**: "You can map this to whatever your team uses - Confluence, Notion, GitHub wikis, SharePoint. The pattern is the same: take your documents, index them, and make them queryable."

---

## Tech Stack Overview

- **LLM & Embeddings**:
  - OpenAI Chat Completions (`gpt-4o-mini`) + Embeddings (`text-embedding-3-small`).
- **Vision model (for images & image-heavy PDFs)**:
  - OpenAI multimodal model (`gpt-4o` by default) for OCR and table extraction.
- **Vector Database**:
  - **Chroma** with a persistent collection `documents`.
- **Backend**:
  - **FastAPI** providing `/query`, `/health`, and ingestion endpoints.
- **Frontend**:
  - **Streamlit** web app for chat UI and file upload.
- **Config & Environment**:
  - `.env` + `python-dotenv` to manage `OPENAI_API_KEY` and paths.

**Speaking Points:**

- **LLM choice**: "For the language model, we use OpenAI's gpt-4o-mini. It's cost-effective, fast, and provides good quality for our use case. For embeddings, we use text-embedding-3-small, which produces 1536-dimensional vectors optimized for retrieval."
- **Vision model**: "For handling images and image-heavy PDF pages, we use OpenAI's gpt-4o model, which is multimodal - it can understand both text and images. We use it to perform OCR and extract text from tables, screenshots, and scanned documents."
- **Vector database**: "We chose Chroma as our vector database. It's simple, runs locally, and persists data to disk. It handles the similarity search efficiently and stores metadata alongside vectors. There are other options - Qdrant, Pinecone, pgvector - but Chroma is great for getting started."
- **Backend**: "The backend is FastAPI, which is a modern Python web framework. It provides three main endpoints: a health check, a query endpoint that takes questions and returns answers, and an ingestion endpoint for uploading files. FastAPI gives us automatic API documentation and type validation."
- **Frontend**: "The frontend is Streamlit, which is perfect for rapid prototyping. It gives us a chat interface, file upload capability, and controls for adjusting parameters. It's not a production-grade UI, but it's excellent for demos and internal tools."
- **Config**: "Configuration is handled through environment variables using python-dotenv. The OpenAI API key, model names, and file paths are all configurable without changing code."
- **Local-first**: "The key point here is that all of this runs locally. Chroma runs on your machine, FastAPI runs locally, Streamlit runs locally. The only external service is OpenAI's API. There's no Kubernetes, no cloud infrastructure, no heavy orchestration. This makes it accessible and easy to understand."

---

## High-Level Architecture

- **Offline / Batch (Ingestion)**:
  - Read documents from `data/` or uploads via UI.
  - Extract text (from text layers) + run vision OCR (for images and image-heavy PDF pages).
  - Chunk, embed, and store in Chroma (`chroma_store/`).
- **Online / Query Path**:
  - Streamlit UI sends question to FastAPI `/query`.
  - `RagQueryEngine` retrieves top‑k chunks from Chroma.
  - Builds a grounded prompt and calls OpenAI.
  - Returns answer + supporting chunks back to the UI.

**Speaking Points:**

- **Two-phase system**: "The architecture has two distinct phases: offline ingestion and online querying. These are separate concerns, which makes the system easier to understand and maintain."
- **Offline phase - Read documents**: "The ingestion phase is offline and batch-oriented. Documents can come from a `data/` directory on disk, or they can be uploaded through the UI. This is a one-time or periodic process - you run it when documents are added or updated."
- **Offline phase - Extract text**: "For each document, we extract text. For Markdown and text files, this is straightforward. For PDFs, we use pypdf to extract the text layer. But PDFs often have content rendered as images - tables, scanned pages, diagrams. For those, we use OpenAI's vision model to perform OCR and extract the text."
- **Offline phase - Chunk and embed**: "Once we have the text, we chunk it into smaller pieces - typically 400 words with 50-word overlap. Each chunk is then embedded into a vector using OpenAI's embedding model. These vectors are what enable semantic search."
- **Offline phase - Store**: "Finally, we store everything in Chroma. Each chunk gets stored with its vector, the original text, and metadata like source file and page number. This creates our searchable knowledge base."
- **Online phase - User question**: "The query path is online and interactive. A user enters a question in the Streamlit UI, which sends a POST request to the FastAPI backend's `/query` endpoint."
- **Online phase - Retrieve**: "The backend uses our RagQueryEngine class, which queries Chroma to retrieve the top k most similar chunks. Chroma handles the embedding of the question and the similarity search."
- **Online phase - Generate**: "The retrieved chunks are combined with the user's question into a prompt, which is sent to OpenAI's chat completion API. The prompt instructs the model to answer using only the provided context."
- **Online phase - Return**: "The answer, along with the supporting chunks for citations, is returned to the UI and displayed to the user."
- **Visual flow**: "If I were to draw this, it would be: Docs → Ingestion → Chroma → FastAPI → OpenAI → Streamlit. Documents flow left to right through ingestion into storage, then queries flow from the UI through the backend to the LLM and back."

---

## Project Structure (Repository View)

- **Top-level**:
  - `README.md` – how to run the project.
  - `requirements.txt` – Python dependencies.
  - `.env` – environment variables (e.g., `OPENAI_API_KEY`).
- **Data & Storage**:
  - `data/` – Markdown & PDF documents to index.
  - `chroma_store/` – Chroma's persistent vector store.
- **Code**:
  - `rag/` – ingestion and core query engine.
  - `app/` – FastAPI backend.
  - `ui/` – Streamlit frontend.

**Speaking Points:**

- **Top-level files**: "At the top level, we have the standard project files. README.md explains how to set up and run the project. requirements.txt lists all Python dependencies. .env contains environment variables, most importantly the OpenAI API key."
- **Data directory**: "The `data/` directory is where you put documents to be indexed - Markdown files, PDFs, images. This is the input to the ingestion process."
- **Chroma store**: "The `chroma_store/` directory is where Chroma persists its vector database. This is created automatically when you first run ingestion. It contains the embedded vectors, original text, and metadata."
- **RAG module**: "The `rag/` directory contains the core RAG logic - the ingestion code that processes documents and the query engine that retrieves and generates answers. This is the heart of the system, and it's independent of the web framework."
- **App module**: "The `app/` directory contains the FastAPI backend. It's a thin layer that exposes HTTP endpoints and delegates to the RAG module. This separation means you could swap FastAPI for Flask or Django without changing the core logic."
- **UI module**: "The `ui/` directory contains the Streamlit frontend. Again, this is just a UI layer - you could replace it with React, Vue, or any other frontend framework, and the backend API would work the same."
- **Mental map**: "Keep this structure in mind as we dive into the code. The separation of concerns - data, storage, core logic, API, UI - makes the system modular and easy to understand. Each component has a clear responsibility."

---

## Ingestion: Responsibilities

- Discover supported files in `data/`.
- Extract raw text from:
  - Markdown / text files (UTF-8 decode).
  - PDFs via `pypdf` (text layer).
  - Image-heavy PDF pages and standalone images via **OpenAI vision OCR**.
- Chunk documents into overlapping segments.
- Embed each chunk using OpenAI embeddings.
- Upsert chunks into the Chroma `documents` collection with:
  - `id`, `content`, `source`, optional `page` metadata.

**Speaking Points:**

- **Discover files**: "The ingestion process starts by discovering supported files in the data directory. We walk through the directory tree looking for Markdown files, text files, PDFs, and images. This is a simple file system traversal."
- **Extract text - Markdown**: "For Markdown and text files, extraction is straightforward - we just read them as UTF-8. These files are already in text format, so there's no parsing needed."
- **Extract text - PDFs**: "For PDFs, we use the pypdf library to extract the text layer. Most PDFs have a text layer that contains the actual text content. However, some PDFs - especially scanned documents or PDFs with complex layouts - have content rendered as images, which pypdf can't read."
- **Extract text - Vision OCR**: "For image-heavy PDF pages and standalone images, we use OpenAI's vision model to perform OCR. We render PDF pages to images and send them to the vision model, which extracts all visible text and linearizes tables into readable sentences."
- **Chunk documents**: "Once we have the text, we chunk it into smaller segments. We use a simple approach - split on whitespace and create overlapping windows. This ensures that important information isn't split across chunk boundaries."
- **Embed chunks**: "Each chunk is then embedded using OpenAI's embedding model. This converts the text into a vector representation that captures semantic meaning. The same model is used for both ingestion and query time to ensure consistency."
- **Upsert to Chroma**: "Finally, we upsert the chunks into Chroma. We use 'upsert' rather than 'add' so that re-ingesting the same document doesn't fail - it just updates the existing chunks. Each chunk gets a unique ID, the original content, source file information, and optional page metadata."
- **Idempotent and repeatable**: "A crucial property of ingestion is that it's idempotent and repeatable. You can run it multiple times, and it will update the index with any changes. This means you can re-run ingestion whenever documents are added or updated without worrying about duplicates or stale data."

---

## Chunking Strategy

- **Why chunk?**
  - Whole documents may be too long for the context window.
  - Smaller chunks improve retrieval granularity and relevance.
- **Our simple approach**:
  - Split on whitespace into "token-like" units.
  - Build windows of ~400 tokens with ~50-token overlap.
  - Treat each vision-OCR output (e.g., a table rendered as sentences) as part of the text to be chunked.
- **Trade-offs**:
  - Simple and fast, good enough for a demo.
  - Can be improved with **semantic** or **structure-aware** chunking later.

**Speaking Points:**

- **Why chunk - Context window**: "You might wonder why we don't just embed entire documents. The first reason is context window limits. LLMs have maximum context lengths - for gpt-4o-mini, that's 128k tokens, but for many models it's much smaller. A single large document might exceed this."
- **Why chunk - Granularity**: "More importantly, smaller chunks improve retrieval granularity and relevance. If you embed a 100-page document as one chunk, a question about page 50 will have to retrieve the entire document. With chunks, you can retrieve just the relevant section."
- **Our approach - Simple splitting**: "Our chunking strategy is intentionally simple. We split text on whitespace into word-like units, then build windows of approximately 400 words with 50 words of overlap between windows. This is a naive approach, but it's fast and works reasonably well."
- **Our approach - Vision OCR**: "When we extract text from images using vision OCR, that text is treated the same as any other text. If a table is linearized into sentences like 'Data type: Computer composed paper. Timeline: 40% by March 2020', that becomes part of the text stream and gets chunked normally."
- **Overlap importance**: "The overlap is crucial. If we have chunks of 400 words with no overlap, a sentence that spans the boundary between chunks might get split. With 50 words of overlap, we ensure that important information near boundaries appears in multiple chunks, reducing the chance it gets missed."
- **Trade-off - Simplicity**: "This approach is simple and fast, which makes it good for demos and getting started. It doesn't require any special libraries or complex logic."
- **Trade-off - Future improvements**: "However, there are more sophisticated approaches. Semantic chunking uses embeddings to find natural boundaries. Structure-aware chunking respects document structure - splitting on headings, paragraphs, or sections. These can improve retrieval quality but add complexity."

---

## Multimodal Ingestion: Text + Vision

- **Motivation**:
  - Real documents mix **text, tables, and images** (screenshots, scanned PDFs).
  - Plain text extraction misses image-only content (e.g., timelines in tables).
- **Our solution**:
  - For each PDF page:
    - Use `pypdf` to grab any existing text.
    - If the page has little text, render it to an image and send it to **OpenAI vision (`gpt-4o`)**.
  - For standalone images:
    - Send them directly to the vision model.
  - The vision model:
    - Performs OCR.
    - Linearizes tables into sentences like  
      "Data type: Computer composed paper. Timeline: 40% data by 10 March 2020; 80% data by 10 August 2020."
- **Result**:
  - Both text and image-only content end up as plain text chunks in Chroma, making questions answerable via RAG.

**Speaking Points:**

- **Motivation - Real documents**: "Real-world documents are messy. They mix text, tables, images, screenshots, scanned pages. A PDF might have a text layer for some content, but tables and diagrams are often rendered as images. Plain text extraction using pypdf will miss all of that image content."
- **Motivation - Missing content**: "This is a real problem. We had a PDF with a timeline table that was completely invisible to text extraction. When we asked 'What was the timeline for computer-composed data submission?', the system couldn't answer because the information was in an image, not in the text layer."
- **Solution - PDF pages**: "Our solution is multimodal. For each PDF page, we first try to extract text using pypdf. If the page has substantial text - say more than 200 characters - we use that. But if it has little or no text, we render the page to an image and send it to OpenAI's vision model."
- **Solution - Standalone images**: "For standalone images uploaded directly - PNGs, JPGs, screenshots - we send them straight to the vision model. There's no text layer to extract, so vision OCR is the only option."
- **Vision model - OCR**: "The vision model, gpt-4o, performs OCR - it reads all the text visible in the image. But it does more than just OCR - it understands structure."
- **Vision model - Table linearization**: "For tables, the vision model linearizes them into readable sentences. Instead of trying to preserve table structure, it converts each row into a sentence. For example, a table row becomes 'Data type: Computer composed paper. Data source: A4 whitepaper. Timeline: 40% data by 10 March 2020; 80% data by 10 August 2020.' This makes the information searchable and answerable."
- **Result - Unified text**: "The result is that both text-layer content and image content end up as plain text chunks in Chroma. From the RAG system's perspective, it's all just text. Questions about content that was originally in images can now be answered because that content is in the vector database."
- **Real example**: "In our testing, we had a question 'Who is Fakhruddin?' that worked because the name was in the text layer. But 'What was the timeline for computer-composed data submission?' only worked after we added vision OCR, because that timeline was in a table rendered as an image. Vision OCR extracted it, and now the question works perfectly."

---

## What the Vector Store Contains

- For each chunk, Chroma stores:
  - **id**: e.g., `my_doc.md-3`.
  - **document**: the chunk text.
  - **embedding**: high-dimensional vector from OpenAI.
  - **metadata**: e.g., `{ "source": "my_doc.md", "page": 2 }`.
- Conceptually like a table:
  - Columns for text, embedding, and metadata.
  - Row per chunk.
- Retrieval:
  - Given a query, Chroma embeds it and returns the most similar rows.

**Speaking Points:**

- **ID field**: "Each chunk gets a unique ID. We use a simple naming scheme: the filename followed by a dash and the chunk index. For example, 'my_doc.md-3' means the third chunk from my_doc.md. This makes it easy to identify chunks and handle updates."
- **Document field**: "The document field stores the original chunk text. This is crucial - we need the actual text to include in the prompt sent to the LLM. The embedding is just for search; the text is what gets used for generation."
- **Embedding field**: "The embedding is the high-dimensional vector - 1536 numbers for OpenAI's models. This is what enables semantic search. Chroma stores these vectors in an optimized format for fast similarity search."
- **Metadata field**: "Metadata stores additional information about each chunk. At minimum, we store the source file name. For PDFs, we also store the page number. This metadata is essential for citations - when we show the user where an answer came from, we use this metadata."
- **Table analogy**: "Conceptually, you can think of Chroma as a table. Each row is a chunk. The columns are: ID, document text, embedding vector, and metadata. It's like a database, but optimized for vector similarity search rather than SQL queries."
- **Retrieval process**: "When you query Chroma, it embeds your question using the same embedding model, then searches for the rows with the most similar embedding vectors. It returns the top k rows along with their similarity scores."
- **Concrete example**: "Let me give you a concrete example. Say we have two chunks: 'my_doc.md-0' with text 'The system uses Python and FastAPI' and 'my_doc.md-1' with text 'Authentication is handled via API keys'. If you ask 'What technology does the system use?', Chroma will find that the first chunk is more similar and return it. The metadata tells us it came from my_doc.md, page 1."

---

## Query Engine: Responsibilities

- Wrap Chroma + OpenAI behind a simple interface:
  - `retrieve(question, top_k)`
  - `build_prompt(question, chunks)`
  - `generate_answer(prompt)`
  - `answer_question(question, top_k, allow_llm_fallback)`
- Handle:
  - Empty or missing documents.
  - Chroma or OpenAI errors.
  - Optional fallback to "LLM-only" answers when retrieval fails.

**Speaking Points:**

- **Interface design**: "The Query Engine wraps Chroma and OpenAI behind a simple, clean interface. It exposes four main methods that handle the core RAG operations."
- **Retrieve method**: "The retrieve method takes a question and a top_k parameter, queries Chroma for the most similar chunks, and returns them. This is pure retrieval - no generation yet."
- **Build prompt method**: "The build_prompt method takes the question and retrieved chunks and constructs the prompt that will be sent to the LLM. This includes system instructions, the context chunks, and the user's question."
- **Generate answer method**: "The generate_answer method takes the constructed prompt and calls OpenAI's chat completion API. It handles the API call and extracts the answer text from the response."
- **Answer question method**: "The answer_question method is the high-level interface that orchestrates the whole process. It calls retrieve, then build_prompt, then generate_answer, and handles errors gracefully."
- **Error handling - Empty documents**: "The engine handles edge cases. If there are no documents in Chroma, it returns a helpful message rather than crashing. This is important for a good user experience."
- **Error handling - API errors**: "If Chroma or OpenAI APIs fail - network issues, authentication problems, rate limits - the engine catches these errors and returns user-friendly error messages rather than exposing stack traces."
- **Fallback mode**: "The engine supports an optional LLM-only fallback. If retrieval fails or returns no chunks, and fallback is enabled, it can still answer using the LLM's own knowledge. This prevents a completely broken experience."
- **Separation of concerns**: "A key design principle here is separation of concerns. The Query Engine is pure RAG logic - it doesn't know about HTTP, FastAPI, or Streamlit. This makes it testable, reusable, and easy to understand. You could use this same engine with a CLI, a Slack bot, or any other interface."

---

## Retrieval Phase in Detail

- Given a user question:
  - `RagQueryEngine` calls Chroma's `query(...)`:
    - `query_texts=[question]`
    - `n_results=top_k` (defaults to 10 in our engine; UI slider defaults to 8).
  - Chroma:
    - Embeds the question using the same embedding model as ingestion.
    - Computes similarity scores vs stored embeddings.
    - Returns:
      - `documents` (chunk texts)
      - `metadatas` (source, page)
      - `distances` (similarity scores)
- The engine converts this into a list of chunk dicts:
  - `content`, `source`, `page`, `score`.

**Speaking Points:**

- **User question input**: "The retrieval phase starts when a user asks a question. This is just plain text - something like 'What was the timeline for computer-composed data submission?' or 'Who is Fakhruddin?'"
- **Chroma query call**: "Our RagQueryEngine calls Chroma's query method. We pass the question as query_texts - it's a list because Chroma supports batch queries, though we typically just pass one question. We also specify n_results, which is how many chunks we want back."
- **Top-k defaults**: "We default to retrieving 10 chunks in the engine, but the UI slider defaults to 8. Why 8 or 10? It's a balance. Too few chunks and you might miss relevant information. Too many and you add noise and increase token costs. We found 8-10 works well for most questions."
- **Chroma embedding**: "Chroma takes the question and embeds it using the same embedding model we used during ingestion. This is crucial - you must use the same model for both ingestion and query, otherwise the vectors won't be comparable."
- **Similarity computation**: "Chroma then computes similarity scores between the query embedding and all stored embeddings. It uses cosine similarity by default, which measures the angle between vectors. Vectors pointing in similar directions have high similarity."
- **Return values**: "Chroma returns three things: the document texts themselves, the metadata for each chunk, and the distance scores. Distance is the inverse of similarity - smaller distance means higher similarity, more relevant chunks."
- **Distance vs similarity**: "Let me clarify distance versus similarity. Distance measures how far apart vectors are - smaller distance means vectors are closer together, which means the texts are more semantically similar. A distance of 0.6 means the chunks are fairly similar; 0.9 means they're less similar. Think of it like physical distance - closer objects are more similar."
- **Engine conversion**: "The engine converts Chroma's raw response into a cleaner format - a list of dictionaries. Each dict has content (the chunk text), source (file name), page (if available), and score (the similarity score). This normalized format makes it easier to work with in the rest of the pipeline."

---

## Prompt Construction & Generation

- `build_prompt(question, chunks)`:
  - Creates labeled context blocks, e.g.:
    - `[Document chunk 1 | source=runbook.md | page=3]`
  - Assembles all chunks into a single **Context** section.
  - Adds system instructions / guidelines:
    - Use **only** the provided context as the source of truth.
    - Treat partial name/term matches as the same entity when reasonable (e.g., "Fakhruddin" vs "A.K.M Fakruddin Mahamud").
    - For tables and lists, read the relevant row(s) and restate them clearly in natural language.
    - Prefer short, direct answers; say "I don't know" only if the answer is genuinely absent.
- `generate_answer(prompt)`:
  - Calls OpenAI Chat Completions with low temperature (e.g., 0.2).
  - Returns the model's answer text.

**Speaking Points:**

- **Build prompt - Labeled blocks**: "The build_prompt method creates labeled context blocks for each retrieved chunk. Each chunk gets a header like '[Document chunk 1 | source=runbook.md | page=3]' followed by the chunk text. This labeling helps the model understand where information came from and helps with citations."
- **Build prompt - Context assembly**: "All chunks are assembled into a single Context section. This becomes part of the prompt sent to the LLM. The format is: system instructions, then the Context section with all chunks, then the user's question."
- **System instruction - Only context**: "The first and most important system instruction is: use ONLY the provided context as the source of truth. This is crucial for preventing hallucinations. The model should not make up information or use its training data - only what's in the retrieved chunks."
- **System instruction - Partial matches**: "We explicitly tell the model to treat partial name or term matches as the same entity. This is important because documents might say 'A.K.M Fakruddin Mahamud' but users might ask 'Who is Fakhruddin?'. The model should recognize these refer to the same person."
- **System instruction - Tables**: "For tables and lists extracted via vision OCR, we instruct the model to read the relevant rows and restate them clearly in natural language. Tables are linearized during ingestion, but the model should present the information in a readable way."
- **System instruction - Direct answers**: "We prefer short, direct answers. The model should get to the point quickly. And it should only say 'I don't know' if the answer is genuinely absent from the context - not if it's just unsure or if the answer requires inference."
- **Generate answer - API call**: "The generate_answer method takes the constructed prompt and calls OpenAI's chat completions API. We use gpt-4o-mini for cost-effectiveness, though you could use any model."
- **Generate answer - Temperature**: "We set temperature to 0.2, which is quite low. This makes the model more deterministic and factual, less creative. For knowledge retrieval tasks, we want consistency and accuracy, not creativity."
- **Example prompt**: "Let me show you what a prompt looks like. System: 'You are an internal knowledge assistant. Answer using ONLY the provided context.' Context: '[Document chunk 1 | source=doc.pdf] The timeline is 40% by March 2020, 80% by August 2020.' Question: 'What was the timeline?' Answer: 'The timeline for computer-composed data submission was 40% by March 2020 and 80% by August 2020.'"

---

## LLM-Only Fallback Mode

- **Why we have it**:
  - Avoid a completely broken experience when:
    - Chroma is empty or unavailable.
    - OpenAI embedding calls fail.
  - Still allow the model to answer using its **own knowledge**.
- Behavior:
  - If retrieval fails or returns no chunks:
    - Optionally call a separate prompt that does **not** include retrieved context.
    - Tell the user implicitly that this is a general answer (UI can indicate no sources).

**Speaking Points:**

- **Why fallback - Broken experience**: "We include an LLM-only fallback mode to avoid a completely broken user experience. There are several scenarios where retrieval might fail: Chroma might be empty if no documents have been ingested yet, Chroma might be unavailable due to a disk issue, or OpenAI embedding API calls might fail due to network or authentication problems."
- **Why fallback - Still useful**: "In these cases, we don't want to just show an error. The LLM still has useful general knowledge. If someone asks 'What is Python?' and we have no documents, the LLM can still answer from its training data. This keeps the system useful even when the knowledge base isn't set up."
- **Behavior - Conditional**: "The fallback is optional and conditional. If retrieval fails or returns no chunks, and the allow_llm_fallback flag is true, we call a separate prompt that does NOT include any retrieved context."
- **Behavior - Different prompt**: "This fallback prompt is different from the RAG prompt. It tells the model: 'You are a general-purpose assistant. Answer using your own knowledge. If you don't know, say so.' This is the standard LLM behavior, not RAG behavior."
- **UI indication**: "The UI can indicate when an answer came from fallback mode by showing no source citations. This tells the user that this is a general answer, not grounded in their documents. It's transparent about the source of information."
- **UX choice**: "This is a design choice. For some domains - like medical or legal - you might want to disable fallback entirely and require that all answers be grounded in retrieved documents. For internal knowledge assistants, having a fallback provides a better user experience during setup or when documents aren't available."

---

## FastAPI Backend: API Surface

- **Endpoints**:
  - `GET /health` – returns `{ "status": "ok" }`.
  - `POST /query` – body: `{ "question": str, "top_k": int, "llm_fallback": bool }`.
  - `POST /ingest_files` – multipart file upload for on-the-fly ingestion.
- **Responsibilities**:
  - Validate inputs with Pydantic models.
  - Delegate to `RagQueryEngine` and ingestion helpers.
  - Return structured JSON with:
    - `answer` (string).
    - `chunks` (list of `{ content, source, page, score }`).

**Speaking Points:**

- **Health endpoint**: "The health endpoint is simple - it just returns a status OK. This is useful for monitoring and for the UI to check if the backend is running. It's a standard pattern in microservices."
- **Query endpoint**: "The query endpoint is the main API. It accepts a POST request with a JSON body containing the question, top_k parameter for how many chunks to retrieve, and a flag for whether to allow LLM fallback. This is where users ask questions."
- **Ingest endpoint**: "The ingest_files endpoint accepts multipart file uploads. Users can upload documents through the UI, and the backend processes them on-the-fly. This makes the system interactive - you don't need to pre-populate a data directory."
- **Pydantic validation**: "FastAPI uses Pydantic models for automatic request validation. If someone sends invalid data - wrong types, missing fields - FastAPI automatically returns a 422 error with details. This is built-in and requires no extra code."
- **Delegation**: "The backend is intentionally thin. It doesn't contain RAG logic - that's all in the RAG module. The backend just validates inputs, calls the appropriate functions, and formats responses. This separation makes the code cleaner and more testable."
- **Response structure**: "Every response is structured JSON. The answer is a string, and chunks is a list of objects with content, source, page, and score. This consistent structure makes it easy for any frontend to consume the API."
- **Swappable frontend**: "This clean API contract means you can swap the frontend easily. We use Streamlit for demos, but you could build a React app, a Slack bot, a CLI tool, or integrate it into any system that can make HTTP requests. The backend doesn't care."

---

## Streamlit Frontend: UX Goals

- **Simple chat interface**:
  - Text area for questions.
  - Button to send the query.
  - Display answer and supporting chunks.
- **Observability for demos**:
  - Sidebar controls for `top_k` and toggling LLM fallback.
  - Button to call `/health` and show backend status.
  - UI for uploading documents and triggering ingestion.

**Speaking Points:**

- **Chat interface - Text area**: "The main interface is a simple chat-style UI. There's a text area where users type their questions. It's intentionally simple - no fancy chat bubbles or conversation history, just a straightforward Q&A interface."
- **Chat interface - Send button**: "A button sends the query to the backend. When clicked, it shows a spinner while waiting for the response, then displays the answer."
- **Chat interface - Display chunks**: "The answer is shown prominently, and below it we display the supporting chunks in an expandable section. Each chunk shows its source, page number if available, similarity score, and the actual text. This transparency is crucial for trust."
- **Observability - Sidebar controls**: "In the sidebar, we have controls for observability and experimentation. Users can adjust top_k - how many chunks to retrieve - and toggle LLM fallback on or off. This lets people experiment and understand how the system works."
- **Observability - Health check**: "There's a health check button that calls the backend's /health endpoint and displays the status. This is useful for debugging and ensuring the backend is running."
- **Observability - File upload**: "The UI includes a file upload widget where users can select multiple files - PDFs, Markdown, images - and trigger ingestion. This makes the system self-contained - you don't need to manually place files in a directory."
- **Prototyping tool**: "Streamlit is perfect for prototyping and demos. You can build a functional UI in minutes with just Python. It's not a production-grade framework - it's single-threaded and not optimized for scale - but for internal tools and demos, it's excellent."
- **Production UI**: "For production, you'd likely want a more sophisticated frontend - React, Vue, or a proper chat framework. But the beauty is that the backend API doesn't change. You can build a production UI that calls the same endpoints, and all your RAG logic stays the same."

---

## End-to-End Flow: From Bytes to Answer

1. **Ingestion**:
   - Files in `data/` or uploaded via UI → **multimodal extraction** (text layer + vision OCR for images/tables) → chunking → embedding → Chroma.
2. **User Question**:
   - Question entered in Streamlit → POST `/query` to FastAPI.
3. **Retrieval**:
   - `RagQueryEngine` queries Chroma and gets top‑k chunks (default top_k=8).
4. **Prompting & Generation**:
   - Builds grounded prompt with improved guidelines (partial name matching, table extraction) and calls OpenAI LLM.
5. **Response**:
   - FastAPI returns answer + chunks → Streamlit renders both answer and supporting context.

**Speaking Points:**

- **Step 1 - Ingestion overview**: "Let's trace the complete flow from a document file to an answer. Step one is ingestion. Files can come from the data directory or be uploaded through the UI. We perform multimodal extraction - getting text from text layers and using vision OCR for images and tables. Then we chunk, embed, and store in Chroma."
- **Step 1 - Multimodal detail**: "The multimodal extraction is key. For a PDF, we extract the text layer first. If a page has little text, we render it to an image and send it to the vision model. For standalone images, we go straight to vision OCR. All of this text - from text layers and OCR - gets chunked and embedded together."
- **Step 2 - User question**: "Step two is when a user asks a question. They type it in the Streamlit UI, which sends a POST request to FastAPI's /query endpoint with the question and parameters."
- **Step 3 - Retrieval detail**: "Step three is retrieval. The RagQueryEngine takes the question, embeds it, and queries Chroma. Chroma returns the top k most similar chunks - we default to 8, but this is configurable. These chunks are the most semantically relevant pieces of our knowledge base for answering the question."
- **Step 4 - Prompting detail**: "Step four is prompting and generation. We build a prompt that includes system instructions - use only context, handle partial matches, extract from tables - then all the retrieved chunks as context, then the user's question. This prompt goes to OpenAI's LLM, which generates the answer."
- **Step 5 - Response detail**: "Step five is the response. FastAPI returns structured JSON with the answer and all the supporting chunks. Streamlit renders the answer prominently and shows the chunks in an expandable section so users can see where the information came from."
- **Trace the code**: "I encourage you to trace this path in the codebase after the talk. Start with rag/ingest.py for ingestion, then rag/query.py for retrieval and generation, then app/main.py for the API, and ui/app.py for the frontend. Following the code will make everything concrete."

---

## Limitations of Our Simple RAG

- **Single-step retrieval**:
  - One query → one retrieval; complex questions may need multi-step reasoning or query rewriting.
- **Basic chunking**:
  - Fixed-size, word-based windows; does not respect headings or semantic boundaries.
- **No reranking layer**:
  - We trust Chroma's top‑k as-is; we could add a dedicated reranker model.
- **Local-only vector store**:
  - Great for demos; at scale we might need managed or distributed vector DBs.
- **Small model for generation**:
  - `gpt-4o-mini` is cost-effective; more complex domains might require larger models.

**Speaking Points:**

- **Limitation 1 - Single-step**: "Our RAG system is intentionally simple, and that means it has limitations. First, we do single-step retrieval - one query, one retrieval, one answer. Complex questions that require multi-step reasoning - like 'What did we learn from the last three postmortems?' - might need multiple retrievals or query rewriting. Our system doesn't do that."
- **Limitation 2 - Basic chunking**: "Second, our chunking is basic. We use fixed-size, word-based windows that don't respect document structure. If a document has clear headings or sections, we might split a section across chunks, losing context. Semantic chunking or structure-aware chunking would be better."
- **Limitation 3 - No reranking**: "Third, we don't have a reranking layer. We trust Chroma's top-k results as-is. But sometimes the most similar chunks by embedding distance aren't the most relevant for answering the question. A dedicated reranker model could reorder the results for better quality."
- **Limitation 4 - Local vector store**: "Fourth, we use a local vector store - Chroma running on your machine. This is great for demos and small teams, but it doesn't scale. For production with millions of documents or high query volume, you'd want a managed service like Pinecone or a distributed system."
- **Limitation 5 - Small model**: "Fifth, we use gpt-4o-mini for generation. It's cost-effective and fast, but for complex domains with nuanced reasoning, you might need a larger model like gpt-4 or Claude. The trade-off is cost and latency versus quality."
- **Tie to improvements**: "Each of these limitations has a corresponding improvement we can make. Let's look at those next."

---

## Potential Improvements & Extensions

- **Better chunking**:
  - Semantic or structure-aware splitting (e.g., heading-based, section-based).
- **Reranking**:
  - Use cross-encoder rerankers (e.g., `bge-reranker`) to reorder Chroma's top‑k.
- **Hybrid retrieval**:
  - Combine vector search with keyword/BM25 filters and metadata constraints.
- **Scaling the vector store**:
  - Swap Chroma for Qdrant, pgvector, Pinecone, or Elasticsearch as data grows.
- **Advanced UX**:
  - Conversation history, follow-up questions, and source-level filtering (by team, system, date).

**Speaking Points:**

- **Improvement 1 - Better chunking**: "The first improvement is better chunking. Instead of fixed-size windows, we could use semantic chunking - using embeddings to find natural boundaries where topics change. Or structure-aware chunking that respects headings, paragraphs, or sections. This would improve retrieval quality by keeping related information together."
- **Improvement 2 - Reranking**: "Second, we could add a reranking layer. After Chroma returns the top-k chunks, we could use a cross-encoder reranker like bge-reranker. These models are trained specifically to score query-document pairs and can reorder results for better relevance. It's an extra step, but it often improves answer quality significantly."
- **Improvement 3 - Hybrid retrieval**: "Third, hybrid retrieval combines multiple search strategies. We could combine vector search with keyword-based BM25 search, or add metadata filters - only search in documents from a specific team or date range. This gives you the best of both worlds - semantic understanding and precise filtering."
- **Improvement 4 - Scaling**: "Fourth, for scaling, you'd swap Chroma for a managed service. Pinecone is a popular cloud vector database. Qdrant and Weaviate are open-source options you can self-host. pgvector lets you use Postgres as a vector store. Elasticsearch has vector search capabilities. Each has different trade-offs for scale, cost, and features."
- **Improvement 5 - Advanced UX**: "Fifth, the UX could be much richer. Add conversation history so users can ask follow-up questions. Add source-level filtering so users can restrict searches to specific teams or time periods. Add threading so related questions are grouped. These features make the system more useful for real workflows."
- **Production readiness**: "These improvements take the demo toward production readiness. You don't need all of them to start, but as you scale and get more users, these become important for quality and performance."

---

## RAG Architecture Variants

- **Graph RAG**:
  - Knowledge is represented as a graph (entities + relationships).
  - Retrieval involves traversing the graph to collect connected facts before generation.
  - Helpful when relational structure (who owns what, causal chains) is critical.
- **Hybrid RAG**:
  - Combines vector search with symbolic/keyword-based retrieval (e.g., knowledge graphs, BM25).
  - Often uses multiple retrievers and optional reranking.
- **Modular / Orchestrated RAG**:
  - "Retrieve-then-generate" becomes a **workflow**:
    - Routing, branching, loops, and tool calls.
  - Enables complex pipelines for multi-step, knowledge-intensive tasks.

**Speaking Points:**

- **Our system - Linear RAG**: "Before we dive into variants, let me clarify: our system is a linear, basic RAG. It's retrieve-then-generate in a single pass. This is the simplest form and a great starting point. But there are more advanced patterns worth knowing about."
- **Graph RAG - Concept**: "Graph RAG represents knowledge as a graph with entities as nodes and relationships as edges. For example, 'Person X works on System Y' or 'Incident A caused Change B'. This captures relational structure that pure text embeddings might miss."
- **Graph RAG - Retrieval**: "Retrieval in Graph RAG involves traversing the graph. You might start with an entity, follow relationships to connected entities, and collect a subgraph of related facts. This is powerful when understanding relationships is critical - like organizational structure or causal chains."
- **Hybrid RAG - Concept**: "Hybrid RAG combines multiple retrieval strategies. You might use vector search for semantic similarity, BM25 for keyword matching, and knowledge graph traversal for relationships. Then you combine or rerank the results."
- **Hybrid RAG - Benefits**: "The benefit is that different retrieval methods catch different types of information. Vector search finds semantically similar content, keyword search finds exact matches, and graph traversal finds related entities. Combining them gives you comprehensive coverage."
- **Modular RAG - Workflow**: "Modular or Orchestrated RAG turns the simple retrieve-then-generate into a workflow. You might have routing logic that decides which retriever to use based on the question type. You might have loops that do multiple retrievals. You might call tools or external APIs as part of the process."
- **Modular RAG - Complexity**: "This enables complex, multi-step tasks. For example: retrieve initial context, generate a refined query, retrieve again with that query, call an external API for additional data, then generate the final answer. It's like building a reasoning pipeline."
- **When to use variants**: "These variants are for advanced use cases. Start with basic RAG like we've built. Once you understand the fundamentals and have specific needs - like needing to understand relationships or requiring multi-step reasoning - then explore these patterns."

---

## Advanced Options by Phase (High-Level)

- **LLMs & Embeddings**:
  - Providers: OpenAI, Anthropic, Google, open-source models (Llama, Mistral, Qwen, etc.).
  - Embeddings: OpenAI, Cohere, Voyage, Jina, `bge-*`, etc.
- **Ingestion & Processing**:
  - Rich document loaders: `unstructured`, Apache Tika, code-aware loaders.
  - Orchestration: Airflow, Prefect, Dagster.
- **Vector Databases**:
  - Qdrant, Pinecone, Weaviate, Milvus, pgvector, Elasticsearch/OpenSearch.
- **Frameworks & Tooling**:
  - LangChain, LlamaIndex, Semantic Kernel, DSPy, RAG evaluation tools (Ragas, TruLens).

**Speaking Points:**

- **LLMs - Providers**: "For LLMs, you have many options beyond OpenAI. Anthropic's Claude models are excellent. Google has Gemini. And there's a growing ecosystem of open-source models - Llama from Meta, Mistral, Qwen from Alibaba. Each has different strengths in cost, quality, and capabilities."
- **Embeddings - Options**: "For embeddings, OpenAI is popular, but Cohere has strong multilingual models, Voyage AI specializes in retrieval, Jina AI offers good performance, and the bge family from BAAI is open-source and competitive. Different models work better for different languages or domains."
- **Ingestion - Loaders**: "For ingestion, there are rich document loaders beyond basic PDF parsing. The unstructured library handles complex PDFs, Office documents, HTML. Apache Tika is a Java-based content analysis toolkit. Code-aware loaders understand programming languages and can parse repositories."
- **Ingestion - Orchestration**: "For orchestrating ingestion pipelines, you might use Airflow for complex workflows, Prefect for Python-native orchestration, or Dagster for data-aware pipelines. These are useful when you have many documents, need scheduling, or have complex dependencies."
- **Vector DBs - Options**: "For vector databases, we use Chroma, but there are many options. Qdrant is fast and open-source. Pinecone is managed and scales well. Weaviate has graph capabilities. Milvus is designed for scale. pgvector lets you use Postgres. Elasticsearch has vector search. Each has different trade-offs."
- **Frameworks - High-level**: "There are high-level frameworks that abstract away RAG complexity. LangChain and LlamaIndex are the most popular - they provide pre-built components and patterns. Semantic Kernel is Microsoft's framework. DSPy is for programmatic prompting. These can speed development but add abstraction."
- **Frameworks - Evaluation**: "For evaluation, tools like Ragas and TruLens help you measure RAG quality - faithfulness, answer relevance, context utilization. These are crucial for production systems where you need to monitor and improve quality over time."
- **Roadmap approach**: "This slide is a roadmap - names to recognize, not concepts to master immediately. Start simple with what we've built. As you encounter specific needs - better document parsing, scale requirements, evaluation needs - you'll know what tools to explore."

---

## Summary

- **RAG** bridges the gap between static LLMs and your evolving internal knowledge.
- Our project demonstrates a **minimal but realistic RAG stack**:
  - OpenAI (embeddings + LLM + vision model for OCR) + Chroma + FastAPI + Streamlit.
- We walked through:
  - **Multimodal ingestion** (text + vision OCR for images/tables), indexing, retrieval, prompt construction, and answer generation.
- Key features:
  - Handles PDFs, Markdown, and images; extracts text from tables and scanned content.
  - Grounded answers with citations; improved prompt engineering for partial matches and table extraction.
- Next steps:
  - Experiment with better chunking, reranking, hybrid retrieval, and richer UX.

**Speaking Points:**

- **RAG's value**: "To summarize: RAG bridges the gap between static LLMs frozen at training time and your evolving internal knowledge. It lets you query your documents in natural language while maintaining traceability and reducing hallucinations."
- **Our stack**: "Our project demonstrates a minimal but realistic RAG stack. We use OpenAI for embeddings, LLM generation, and vision OCR. Chroma as our vector database. FastAPI for the backend API. Streamlit for the frontend. It's simple enough to understand completely, but realistic enough to be useful."
- **What we covered**: "We walked through the complete pipeline: multimodal ingestion that handles both text and images, indexing into a vector database, semantic retrieval, prompt construction with improved guidelines, and answer generation. Each step is important and builds on the previous ones."
- **Key features - Multimodal**: "Key features include multimodal understanding - we handle PDFs, Markdown, and images. We extract text from tables and scanned content using vision OCR. This makes the system work with real-world documents, not just perfect text files."
- **Key features - Grounded answers**: "We provide grounded answers with citations. Every answer shows where it came from. And we've improved prompt engineering to handle partial name matches and extract information from tables. These details matter for real-world usability."
- **Next steps - Experimentation**: "For next steps, I encourage you to experiment. Try different chunk sizes - maybe 200 words, maybe 600. Adjust top_k - see how it affects answer quality. Try different embedding models or LLMs. The system is designed to be tweakable."
- **Next steps - Extensions**: "Consider the improvements we discussed: better chunking, reranking, hybrid retrieval, richer UX. Pick one that addresses a pain point you're experiencing and implement it. The codebase is structured to make these additions straightforward."
- **Invitation**: "I invite you to explore the repository, read the code, tweak parameters, and extend the system to your own documents. The best way to understand RAG is to build with it. Start simple, iterate, and add complexity as you need it. Thank you, and I'm happy to take questions."

---

## Q&A

- **Questions?**
- Possible topics:
  - Trade-offs between different vector DBs.
  - How to handle security and permissions.
  - When RAG is the right tool vs. fine-tuning or pure LLM solutions.

**Speaking Points:**

- **Opening**: "Thank you for your attention. I'm now happy to take questions. Please feel free to ask about any aspect of RAG, our implementation, or how you might adapt this for your own use cases."
- **Possible topic 1 - Vector DBs**: "Some questions you might have: What are the trade-offs between different vector databases? When should you use Chroma versus Pinecone versus pgvector? The answer depends on scale, cost, hosting preferences, and feature needs."
- **Possible topic 2 - Security**: "How do you handle security and permissions? This is crucial for internal systems. You might need to filter documents by user permissions, encrypt data at rest, or add authentication layers. These are important production considerations we didn't cover in detail."
- **Possible topic 3 - RAG vs alternatives**: "When is RAG the right tool versus fine-tuning or pure LLM solutions? RAG is great when you have evolving knowledge, need citations, or have domain-specific documents. Fine-tuning is better when you need the model to learn specific patterns or styles. Pure LLMs work for general knowledge questions."
- **Other topics**: "Other topics we could discuss: evaluation strategies, cost optimization, handling very large documents, multilingual support, or integrating with existing systems. I'm open to any questions."
- **Closing**: "If you have questions after the talk, feel free to reach out. The code is available in the repository, and I'm happy to help you get started with your own RAG system. Thank you!"
