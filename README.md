# 📚 Study Buddy RAG

A **Retrieval-Augmented Generation** chatbot that answers questions from your
textbook PDFs or handwritten notes — and politely refuses anything outside the
document.

---

## Architecture

```
PDF Upload
   │
   ▼
Text Extraction  ──►  PyMuPDF (fitz)
   │
   ▼
Chunking         ──►  Character-level sliding window (configurable size + overlap)
   │
   ▼
Embeddings       ──►  Google Generative AI  (models/embedding-001)
   │
   ▼
Vector Store     ──►  FAISS (IndexFlatL2, in-memory)
   │
   ▼
Query Pipeline:
  User question → embed → FAISS search → top-K chunks
                                              │
                                              ▼
                                    Gemini 1.5 Flash (answer gen)
                                              │
                                              ▼
                                 Out-of-scope guard (prompt-level)
```

---

## Project Structure

```
study_buddy_rag/
├── app.py            ← Streamlit UI
├── rag_engine.py     ← Core RAG pipeline
├── test_rag.py       ← Automated tests
├── requirements.txt
└── README.md
```

---

## Step-by-Step Setup

### 1. Prerequisites

- Python **3.10+**
- A **Google Gemini API key** — get one free at  
  <https://aistudio.google.com/app/apikey>

---

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note — FAISS on Windows:**  If `faiss-cpu` fails, try:
> ```bash
> pip install faiss-cpu --prefer-binary
> ```

---

### 4. Run the app

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`.

---

### 5. Using the app

1. Paste your **Gemini API key** in the sidebar.
2. Upload any **PDF** (textbook, scanned notes, subject notes).
3. Adjust **chunk size / overlap / top-K** if needed (defaults work well).
4. Click **🚀 Process Document** — wait for "✅ Ready — N chunks indexed".
5. Type questions in the chat box.
6. Expand **📎 Retrieved source chunks** to see exactly which passages were used.

---

### 6. Run the tests

```bash
export GEMINI_API_KEY="AIza..."   # Windows: set GEMINI_API_KEY=AIza...
python test_rag.py
```

Expected output:
```
============================================================
Study Buddy RAG — Test Suite
============================================================
[PASS] Ingestion — 6 chunks indexed
[PASS] In-scope Q — answer: Photosynthesis is the process...
[PASS] Out-of-scope Q — correctly refused
[PASS] Top-K retrieval — 2 chunks returned
============================================================
Results: 4 passed, 0 failed
============================================================
```

---

## Configuration Reference

| Sidebar Setting    | Default | Effect |
|--------------------|---------|--------|
| Chunk size (chars) | 800     | Larger = more context per chunk, fewer chunks |
| Chunk overlap      | 100     | Overlap prevents losing info at boundaries |
| Top-K chunks       | 4       | More = richer context, slower & more tokens |

---

## How Out-of-Scope Detection Works

The Gemini prompt explicitly instructs the model:
> *"If the answer is not present in the context, respond with exactly: `OUT_OF_SCOPE: <reason>`"*

The engine checks whether the response starts with `OUT_OF_SCOPE` and flags it
accordingly — no separate classifier needed.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `google-generativeai` | Gemini embeddings + generation |
| `faiss-cpu` | Fast vector similarity search |
| `pymupdf` | PDF text extraction |
| `numpy` | Embedding matrix operations |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `RESOURCE_EXHAUSTED` from Gemini | You hit the free-tier RPM limit; wait 60 s |
| No text extracted from PDF | PDF may be image-only; use an OCR tool first |
| `ModuleNotFoundError: fitz` | `pip install pymupdf` |
| FAISS install fails | `pip install faiss-cpu --prefer-binary` |
