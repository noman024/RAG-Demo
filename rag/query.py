from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from .config import CHAT_MODEL, CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY


logger = logging.getLogger("rag")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please configure it in your environment.")


client = OpenAI(api_key=OPENAI_API_KEY)

openai_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
)


class RagQueryEngine:
    def __init__(self, collection_name: str = "documents") -> None:
        logger.info("Initializing RagQueryEngine", extra={"collection_name": collection_name})
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_embedding_fn,
        )

    def retrieve(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        logger.info("Retrieving context", extra={"question": question, "top_k": top_k})
        results = self._collection.query(
            query_texts=[question],
            n_results=top_k,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "content": doc,
                    "source": (meta or {}).get("source") if isinstance(meta, dict) else None,
                    "page": (meta or {}).get("page") if isinstance(meta, dict) else None,
                    "score": float(dist),
                }
            )

        logger.info("Retrieved chunks", extra={"question": question, "returned": len(chunks)})
        return chunks

    def build_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a grounded prompt for the LLM from the retrieved chunks.

        The prompt is engineered to:
        - Treat partial name matches (e.g. \"Fakruddin\" vs \"A.K.M Fakruddin Mahamud\") as the same entity.
        - Prefer extraction / lookup over speculation, especially for tables and lists.
        - Answer concisely and explicitly say \"I don't know\" when the answer is not present.
        """
        context_blocks = []
        for idx, ch in enumerate(chunks):
            header = f"[Document chunk {idx + 1} | source={ch.get('source')} | page={ch.get('page')}]"
            context_blocks.append(f"{header}\n{ch['content']}")
        context = "\n\n".join(context_blocks)

        system = (
            "You are an internal knowledge assistant that answers questions using ONLY the provided context. "
            "Treat the context as the source of truth.\n\n"
            "Guidelines:\n"
            "1. If a question refers to a person, role, project, table row, or other entity, look for that name "
            "   or a close variant in the context. Treat partial or shortened forms as the same entity when it is "
            "   reasonable (for example, 'Fakruddin' vs 'A.K.M Fakruddin Mahamud').\n"
            "2. When the answer is contained in a table or list (e.g. 'Name – Position'), read the relevant row and "
            "   restate the information clearly in natural language.\n"
            "3. Prefer short, direct answers over long explanations unless the question explicitly asks for detail.\n"
            "4. If the answer is genuinely not present anywhere in the context, say that you don't know instead of "
            "   guessing or inventing information.\n"
            "5. If multiple relevant pieces of information exist, summarize them clearly.\n"
        )

        user = (
            "You will be given context chunks from internal documents, followed by a user question.\n"
            "Use the guidelines above and answer based only on the context.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        prompt = f"{system}\n{user}"
        return prompt

    def generate_answer(self, prompt: str) -> str:
        logger.info("Calling OpenAI for grounded answer")
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for internal documentation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _llm_only_answer(self, question: str) -> str:
        """Answer using the model's own knowledge (no retrieved context)."""
        logger.info("Calling OpenAI in LLM-only mode", extra={"question": question})
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable general-purpose assistant. "
                        "Answer the user's question using your own knowledge. "
                        "If you genuinely do not know, say that you don't know."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content or ""

    def answer_question(
        self,
        question: str,
        top_k: int = 8,
        allow_llm_fallback: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        High-level helper that runs retrieval + generation.

        Any errors from the vector DB or LLM are caught and surfaced as a
        human-readable message rather than a 500 from the API layer.

        If `allow_llm_fallback` is True and retrieval yields no context or fails,
        we fall back to a pure LLM answer using its own knowledge.
        """
        logger.info(
            "Answering question",
            extra={"question": question, "top_k": top_k, "llm_fallback": allow_llm_fallback},
        )
        try:
            chunks = self.retrieve(question, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - we want to be defensive at this boundary
            logger.exception("Error during retrieval from the vector store")
            if allow_llm_fallback:
                answer = self._llm_only_answer(question)
                return answer, []
            return (
                "Error during retrieval from the vector store. "
                "Check your OpenAI credentials and Chroma configuration. "
                f"Details: {exc}",
                [],
            )

        if not chunks:
            logger.info("No chunks retrieved for question", extra={"question": question})
            if allow_llm_fallback:
                answer = self._llm_only_answer(question)
                return answer, []
            return "I couldn't find any relevant context in the knowledge base.", []

        prompt = self.build_prompt(question, chunks)

        try:
            answer = self.generate_answer(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during answer generation with the LLM")
            if allow_llm_fallback:
                answer = self._llm_only_answer(question)
                return answer, chunks
            return (
                "Error during answer generation with the LLM. "
                "Check your OpenAI API key, model name, and network. "
                f"Details: {exc}",
                chunks,
            )

        logger.info("Answer generation completed", extra={"question": question, "chunk_count": len(chunks)})
        return answer, chunks
