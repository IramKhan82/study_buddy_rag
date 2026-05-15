"""
rag_engine.py
─────────────
Core RAG pipeline:
  1. PDF text extraction  (PyMuPDF)
  2. Chunking             (character-level with overlap)
  3. Embeddings           (Google Generative AI embeddings)
  4. Vector store         (FAISS)
  5. Answer generation    (Google Gemini 1.5 Flash)
  6. Out-of-scope guard   (confidence check in the prompt)
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

import fitz                          # PyMuPDF
import numpy as np
import faiss                         # pip install faiss-cpu
import google.generativeai as genai  # pip install google-generativeai


# ─────────────────────────────────────────────────────────────────────────────
# RAGEngine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """End-to-end RAG pipeline backed by FAISS + Google Gemini."""

    EMBED_MODEL   = "models/embedding-001"
    GENERATE_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        gemini_api_key: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        genai.configure(api_key=gemini_api_key)
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[dict[str, Any]] = []   # {"text": ..., "page": ...}
        self._index: faiss.IndexFlatL2 | None = None
        self._dim: int | None = None

    # ── 1. Ingestion ──────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str) -> int:
        """Extract, chunk, embed, and index the PDF. Returns number of chunks."""
        pages = self._extract_text(pdf_path)
        self._chunks = self._chunk_pages(pages)
        if not self._chunks:
            raise ValueError("No text could be extracted from the PDF.")
        self._build_index(self._chunks)
        return len(self._chunks)

    # ── 2. Query ──────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 4) -> dict[str, Any]:
        """
        Retrieve top_k chunks and generate an answer.
        Returns {"answer": str, "out_of_scope": bool, "chunks": list[dict]}.
        """
        if self._index is None:
            raise RuntimeError("No document indexed. Call ingest() first.")

        q_emb    = self._embed([question])[0]
        chunks   = self._search(q_emb, top_k)
        context  = "\n\n---\n\n".join(
            f"[Page {c['page']}] {c['text']}" for c in chunks
        )
        answer, out_of_scope = self._generate(question, context)
        return {"answer": answer, "out_of_scope": out_of_scope, "chunks": chunks}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: str) -> list[dict[str, Any]]:
        """Return list of {page: int, text: str} dicts."""
        pages = []
        doc   = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append({"page": i, "text": text})
        doc.close()
        return pages

    def _chunk_pages(self, pages: list[dict]) -> list[dict]:
        """Split each page's text into overlapping fixed-size chunks."""
        chunks = []
        for page_dict in pages:
            text = page_dict["text"]
            page = page_dict["page"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "page": page})
                start += self.chunk_size - self.chunk_overlap
        return chunks

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) float32 embedding matrix."""
        result = genai.embed_content(
            model=self.EMBED_MODEL,
            content=texts,
            task_type="retrieval_document",
        )
        vectors = np.array(result["embedding"], dtype=np.float32)
        # embed_content returns a flat list when given a single string,
        # or a list-of-lists for multiple strings — normalise shape.
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        return vectors

    def _build_index(self, chunks: list[dict]) -> None:
        """Build FAISS flat L2 index from chunk embeddings."""
        texts    = [c["text"] for c in chunks]
        # Embed in small batches to respect API limits
        batch    = 100
        all_embs = []
        for i in range(0, len(texts), batch):
            all_embs.append(self._embed(texts[i : i + batch]))
        matrix = np.vstack(all_embs)
        self._dim   = matrix.shape[1]
        self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(matrix)

    def _search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Return top_k most-similar chunks."""
        q = query_vec[np.newaxis, :].astype(np.float32)
        k = min(top_k, len(self._chunks))
        _, indices = self._index.search(q, k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    def _generate(self, question: str, context: str) -> tuple[str, bool]:
        """Call Gemini to answer using only the context provided."""
        prompt = textwrap.dedent(f"""
            You are Study Buddy, a helpful academic assistant.
            You ONLY answer questions based on the document context below.
            If the answer is not present in the context, respond with exactly:
            OUT_OF_SCOPE: <brief reason why you cannot answer>

            === DOCUMENT CONTEXT ===
            {context}
            ========================

            Question: {question}

            Instructions:
            - Answer concisely and accurately using ONLY the context above.
            - If the answer is not in the context, start your reply with OUT_OF_SCOPE:.
            - Do NOT use any external knowledge.
            - Cite the page number(s) when possible, e.g. (Page 3).
        """).strip()

        model    = genai.GenerativeModel(self.GENERATE_MODEL)
        response = model.generate_content(prompt)
        answer   = response.text.strip()

        out_of_scope = answer.upper().startswith("OUT_OF_SCOPE")
        if out_of_scope:
            # Clean up the tag for display
            answer = re.sub(r"(?i)^out_of_scope\s*:\s*", "", answer).strip()
            if not answer:
                answer = "This question is outside the scope of the uploaded document."
        return answer, out_of_scope
